# Week 14 — RAG, Vectors, Agents and Production AI

**CIT AI-Native Engineering Program · Year 1 · Semester 2 · Module M5**

Semester 1 covered these as concepts. This week they are code you can read in
one sitting — small enough that the **architectural decision** at the end is
made with the mechanism in view.

**Lab time: 2 hours. API calls: 12.** See [LAB.md](LAB.md).

> Most of this repo runs **offline, for free**. `LocalEmbedder` is a real
> retriever that costs nothing, so the whole pipeline can be built and
> debugged before a single call is spent.

## Contents

| Path | What it is |
|---|---|
| `src/embeddings.py` | Two embedders, same interface — `LocalEmbedder` (free) and `GeminiEmbedder` |
| `src/vector_store.py` | The vector store and the chunker, in ~100 lines |
| `src/rag.py` | The pipeline, with and without retrieval side by side |
| `src/agent.py` | A two-tool agent with a visible reasoning trace |
| `src/api.py` | `/retrieve` on the Week 13 service, same envelope |
| `worksheets/architecture-decision.md` | RAG, agent, or neither — and what it costs |
| `worksheets/journal-14.md` | Final journal entry before the capstone |
| `tests/` | **37 tests**. No API key needed |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # mac/linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                 # only needed for the --gemini runs
```

## Running

```powershell
python src/rag.py                 # offline, 0 calls
python src/agent.py               # offline, 0 calls
python -m pytest tests -v         # 37 tests, 0 calls
uvicorn src.api:app --reload      # 0 calls

python src/rag.py --gemini        # ~10 calls
python src/agent.py --gemini "..." # ~2-4 calls
```

### Verified in this repo

| | Status |
|---|---|
| 37 tests | **passing** |
| RAG pipeline, all 5 demo questions | **verified offline** |
| Agent loop end to end, with trace | **verified offline** |
| `/health` and `/retrieve`, covered and uncovered queries | **verified over real HTTP** |
| `--gemini` paths (real embeddings, real generation) | **not run** — need a live key |

## The pipeline

```
query -> embed -> similarity search -> top-k -> inject into prompt -> generate
```

```
Q: What is data leakage?
  [0.426] Lecture 6, page 3: Data leakage is when information from the test set...
  [0.378] Lecture 4, page 2: Overfitting is when a model does well on training...

Q: What is the capital of France?
  retrieved 0 chunk(s)
  (nothing above the relevance threshold - this is a RESULT, not a bug)
```

## Retrieving nothing is a feature

The `min_score` floor is the part most tutorials leave out.

Without it, a query about something the corpus does not contain still returns
`top_k` chunks — **the least unrelated ones** — and the model dutifully
answers from them. That is how a RAG system invents an answer *while
appearing to cite a source*, which is worse than having no sources at all,
because the citation makes it more convincing.

A RAG system that cannot say "the corpus does not cover this" will answer
anyway. There is a test for both halves: one that an uncovered query returns
nothing, and one that **without** the floor it returns something.

## Two embedders, one interface

```python
store = VectorStore(LocalEmbedder()).add(chunks)    # free, offline
store = VectorStore(GeminiEmbedder()).add(chunks)   # 1 call, semantic
```

`LocalEmbedder` is bag-of-words with cosine similarity. It genuinely
retrieves — but it matches **words**, not **meaning**. Ask it about "accuracy
of positive predictions" and it will not find the precision chunk, because no
words overlap.

**That gap is the entire argument for real embeddings**, and the lab makes you
watch it happen rather than take it on trust.

## Retrieval quality is a real problem

```
Q: How does a train/test split help with overfitting?
  [0.405] Lecture 5, page 1: Cross-validation repeats the train/test split...
  [0.346] Lecture 4, page 3: A train/test split holds back part of the data...
```

The *wrong* chunk ranks first — bag-of-words over-weights the phrase appearing
in a neighbouring entry. Retrieval sits upstream of everything, and no amount
of prompt engineering downstream fixes a bad top-k.

This is why `/retrieve` returns chunks and scores **without** generating: if
retrieval is wrong, generating anyway just spends a call to confirm it.

## The agent

**Reason → Act → Observe → Reason again.**

```
[1] THOUGHT: I need the definition of recall from the notes.
[1] ACTION:  search_notes('recall formula')
[1] OBSERVE: Lecture 3, page 4: Precision = TP / (TP + FP). Recall = TP / (TP + FN).
[2] THOUGHT: I have the formula. Now compute recall for TP=12, FN=4.
[2] ACTION:  calculate('12 / (12 + 4)')
[2] OBSERVE: 0.75
[3] ANSWER:  Recall = TP/(TP+FN) = 12/16 = 0.75. Source: Lecture 3, page 4.
```

Two tools, chosen because they **fail differently**: `search_notes` can return
nothing, `calculate` can raise. An agent whose tools always succeed teaches
nothing — the whole difficulty of agents is the step *after* something goes
wrong.

Three defences, each against a Week 1 failure mode:

| Code | Failure mode it prevents |
|---|---|
| `MAX_STEPS = 4` | **Infinite loop** — otherwise it runs until your quota does |
| Out of steps returns `None`, not a partial answer | **Silent failure** |
| `calculate` refuses anything but arithmetic | `eval` on model text is an RCE bug waiting for a prompt injection |

And the cost: **each step is a generation call.** A 4-step limit is up to 4
calls per question. Go back and redo the Week 12 cost model.

## Retrieval is not an agent

`/retrieve` calls a search function on every request, in a fixed order.
Nothing *chooses*. That is **rule 4** of the Week 1 matrix — chained calls —
and most "agents" in student projects are rule 4 in a costume.

It becomes rule 2 when the model decides *whether* to search, *what* to search
for, and *whether the result was good enough to search again*.

## The decision this week is for

RAG, fine-tuning, or a bigger context window:

| | Use when | Changes when the data changes |
|---|---|---|
| **Bigger context window** | The corpus fits, and is small | Nothing — paste the new version |
| **RAG** | Too big to send, or changes often | Re-index the changed chunks |
| **Fine-tuning** | You need a *behaviour*, not *facts* | Retrain |

> *"Our docs change weekly so we fine-tuned on them"* is the commonest
> mistake. Fine-tuning teaches behaviour, not facts. The result will sound
> like your docs and still invent the version number.

**If your corpus fits in a prompt and rarely changes, the answer is
"neither" — and saying so is the stronger answer.**

---

## Set B (non-IT background)

**`src/embeddings.py`**
- `tokenise(text)` — splits text into lowercase words and drops common ones
  like "the", which appear everywhere and help you find nothing.
- `cosine_similarity(a, b)` — how closely two vectors point in the same
  direction. 1.0 means identical, 0.0 means unrelated.
- `LocalEmbedder` — turns text into a vector by counting words.
  `fit(documents)` learns the vocabulary; `embed(texts)` makes the vectors.
  Free, offline, deterministic.
- `GeminiEmbedder` — the same two methods, but the vectors come from Google's
  embedding model and capture **meaning** rather than word overlap.

**`src/vector_store.py`**
- `Chunk` — one piece of text plus where it came from.
- `Hit` — a chunk and how well it matched.
- `VectorStore.add(chunks)` — embeds and stores them.
- `VectorStore.search(query, top_k, min_score)` — embeds the query, compares
  it to everything stored, returns the best few — **but only those scoring
  above `min_score`**, so an irrelevant question gets nothing back.
- `chunk_notes(text)` — splits the notes into one chunk per lecture entry.

**`src/rag.py`**
- `build_index(embedder)` — chunk the notes and embed them.
- `build_prompt(question, hits)` — put the retrieved excerpts into the
  prompt, numbered, so the model can cite which one it used.
- `answer(question, store, client)` — the whole pipeline. With no client it
  stops after retrieval, which is the cheap way to debug.

**`src/agent.py`**
- `Tools` — the two things the agent can do, plus `run()`, which calls one
  and turns any error into a message rather than a crash.
- `run_agent(question, ask)` — the loop: ask the model what to do, do it,
  tell the model what happened, repeat until it answers or runs out of steps.
- `scripted_model(replies)` — a fake model for testing, which is how all 37
  tests run without a key.

### What Set B may simplify

- Use `LocalEmbedder` throughout. The `--gemini` runs are **optional**.
- The architectural decision may be one short paragraph in plain language.
- The agent is **read and explained**, not modified.

### What Set B may **not** simplify

- You must run a query that retrieves **nothing** and explain why that is
  correct.
- The architectural decision must name the **rule** from the Week 1 matrix.
- "Neither" is an acceptable answer and must be **defended**, not assumed.
