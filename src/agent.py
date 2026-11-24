"""A two-tool agent with a visible reasoning trace.

    python src/agent.py                    # offline, scripted model, 0 calls
    python src/agent.py --gemini "..."     # real planning loop

    Reason -> Act -> Observe -> Reason again   (the ReAct pattern)

Full agent build is a Year 2 skill. The point of this file is to make the
loop small enough to read in one sitting, so the DESIGN decisions in
worksheets/architecture-decision.md are made with the mechanism in view.

Two tools, chosen because they fail differently:

    search_notes   can return nothing. The agent must cope with an empty
                   observation rather than inventing one.
    calculate      can raise. The agent must cope with a tool that errors.

An agent whose tools always succeed teaches nothing, because the whole
difficulty of agents is what happens on the step after something goes wrong.
"""

import json
import re
import sys
from pathlib import Path

from embeddings import LocalEmbedder
from vector_store import VectorStore, chunk_notes

ROOT = Path(__file__).parent.parent
NOTES = (ROOT / "data" / "notes.txt").read_text(encoding="utf-8")

# Hard stop. Without it, an agent that keeps choosing the same tool runs
# until your quota does. "Infinite loop" was one of the four failure modes
# in Week 1, and this single constant is the whole defence against it.
MAX_STEPS = 4


# --- the tools -----------------------------------------------------------

class Tools:
    """The two things the agent can do."""

    def __init__(self):
        self.store = VectorStore(LocalEmbedder()).add(chunk_notes(NOTES))
        self.calls = []

    def search_notes(self, query):
        """Find relevant lecture notes. May legitimately return nothing."""
        hits = self.store.search(query, top_k=2, min_score=0.05)
        if not hits:
            return "No matching notes found."
        return " | ".join(f"{hit.chunk.text}" for hit in hits)

    def calculate(self, expression):
        """Evaluate simple arithmetic. Raises on anything it does not like.

        Only digits and operators are permitted. `eval` on model-generated
        text is a remote code execution bug waiting for the day someone
        prompt-injects your agent, and "it is only our own model" is not a
        security boundary.
        """
        if not re.fullmatch(r"[\d\s()+\-*/.]+", expression or ""):
            raise ValueError(f"refusing to evaluate {expression!r}: unexpected characters")
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307 - guarded above
        return str(result)

    def run(self, name, argument):
        """Dispatch, recording the call. Tool errors become observations."""
        self.calls.append((name, argument))
        try:
            if name == "search_notes":
                return self.search_notes(argument)
            if name == "calculate":
                return self.calculate(argument)
            return f"Error: no tool called {name!r}. Available: search_notes, calculate."
        except Exception as exc:
            # A failing tool must not kill the loop. The agent gets told what
            # went wrong and decides what to do next - which is the whole
            # difference between an agent and a script.
            return f"Error: {exc}"


TOOL_INSTRUCTIONS = """You are an agent that answers questions using tools.

Available tools:
  search_notes(query)      - search the lecture notes. May return nothing.
  calculate(expression)    - arithmetic only, e.g. "12 / (12 + 4)"

Reply with EXACTLY ONE JSON object per turn, and nothing else:

  {"thought": "...", "action": "search_notes", "action_input": "..."}

or, when you can answer:

  {"thought": "...", "answer": "..."}

Rules:
- One tool call per turn.
- If a tool returns "No matching notes found", do NOT invent the content.
  Say what you could not find.
- If a tool returns an error, do not retry it unchanged.
- Base the final answer only on what the tools returned."""


# --- the loop ------------------------------------------------------------

def run_agent(question, ask, tools=None, max_steps=MAX_STEPS, verbose=True):
    """Reason -> Act -> Observe -> Reason again.

    Args:
        ask: callable(prompt) -> str. The model. Injected so the loop can be
            driven by a script in tests - a planning loop you can only test
            by spending quota is one nobody tests.

    Returns:
        (answer, trace) - trace is every step, which is the artefact the lab
        asks you to inspect. An agent you cannot watch think is an agent you
        cannot debug.
    """
    tools = tools or Tools()
    trace = []
    transcript = f"QUESTION: {question}"

    for step in range(1, max_steps + 1):
        raw = ask(f"{TOOL_INSTRUCTIONS}\n\n{transcript}\n\nYour JSON:")

        try:
            decision = json.loads(raw.strip().strip("`").removeprefix("json").strip())
        except json.JSONDecodeError:
            trace.append({"step": step, "error": "model did not return JSON", "raw": raw})
            if verbose:
                print(f"[{step}] model did not return JSON - stopping")
            return None, trace

        thought = decision.get("thought", "")

        if "answer" in decision:
            trace.append({"step": step, "thought": thought, "answer": decision["answer"]})
            if verbose:
                print(f"[{step}] THOUGHT: {thought}")
                print(f"[{step}] ANSWER:  {decision['answer']}")
            return decision["answer"], trace

        action = decision.get("action")
        action_input = decision.get("action_input", "")
        observation = tools.run(action, action_input)

        trace.append({
            "step": step,
            "thought": thought,
            "action": action,
            "action_input": action_input,
            "observation": observation,
        })

        if verbose:
            print(f"[{step}] THOUGHT: {thought}")
            print(f"[{step}] ACTION:  {action}({action_input!r})")
            print(f"[{step}] OBSERVE: {observation[:110]}")

        transcript += (
            f"\n\nThought: {thought}\nAction: {action}({action_input})"
            f"\nObservation: {observation}"
        )

    # Ran out of steps. This is a FAILURE and is reported as one. An agent
    # that quietly returns its last partial thought as an answer is the
    # "silent failure" mode from Week 1.
    trace.append({"step": max_steps, "error": f"no answer within {max_steps} steps"})
    if verbose:
        print(f"[!] gave up after {max_steps} steps without an answer")
    return None, trace


# --- offline demo --------------------------------------------------------

def scripted_model(replies):
    """A fake model that returns prepared replies in order.

    This is how the loop is tested without an API key. It is also how you
    reproduce an agent failure you saw once and need to fix.
    """
    state = {"n": 0}

    def ask(prompt):
        index = min(state["n"], len(replies) - 1)
        state["n"] += 1
        return replies[index]

    return ask


DEMO = [
    '{"thought": "I need the definition of recall from the notes.", '
    '"action": "search_notes", "action_input": "recall formula"}',
    '{"thought": "I have the formula. Now compute recall for TP=12, FN=4.", '
    '"action": "calculate", "action_input": "12 / (12 + 4)"}',
    '{"thought": "Recall is 0.75.", '
    '"answer": "Recall = TP/(TP+FN) = 12/16 = 0.75. Source: Lecture 3, page 4."}',
]


def main(argv):
    question = next((a for a in argv[1:] if not a.startswith("--")),
                    "What is recall, and what is it for TP=12 and FN=4?")

    if "--gemini" in argv:
        from gemini_client import GeminiClient

        client = GeminiClient()
        ask = lambda prompt: client.ask(prompt)   # noqa: E731
        print("model: gemini (live)\n")
    else:
        ask = scripted_model(DEMO)
        print("model: scripted (offline, 0 API calls)\n")

    print(f"QUESTION: {question}\n")
    answer, trace = run_agent(question, ask)

    print(f"\n--- trace: {len(trace)} step(s) ---")
    print(json.dumps(trace, indent=2)[:900])

    if answer is None:
        print("\nNo answer. That is a reported failure, not a silent one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
