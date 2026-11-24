# Week 14 Lab — RAG, Vectors, Agents and Production AI

**Time: 2 hours · API calls used: 12**

Most of this lab runs **offline for free**. `LocalEmbedder` is a real
retriever that costs nothing, so you build and debug the whole pipeline
before spending a single call — and then spend them on the comparison that
actually teaches something.

| # | Activity | Time | Calls |
|---|----------|------|------:|
| 1 | Demo 1 — RAG, with and without retrieval | 25 min | 0 |
| 2 | Swap in real embeddings and compare | 20 min | 10 |
| 3 | Connect retrieval to the Week 13 backend | 20 min | 0 |
| 4 | Demo 2 — the agent and its trace | 20 min | 2 |
| 5 | Architectural decision | 25 min | 0 |
| 6 | Journal entry | 10 min | 0 |

---

## 1. Demo 1 — RAG over the domain FAQ (25 min, 0 calls)

```powershell
pip install -r requirements.txt
python src/rag.py
```

Watch the pipeline run: **query → embed → similarity search → top-k →
inject → generate**.

Three things to notice:

**"What is the capital of France?" retrieves nothing.** That is a *result*,
not a bug. Without the `min_score` floor it would return the three least
unrelated chunks and the model would answer from them — which is how a RAG
system invents an answer while appearing to cite a source.

**"How does a train/test split help with overfitting?" ranks the
cross-validation chunk first**, above the actual train/test chunk. Bag-of-words
matching over-weights the phrase appearing in a neighbouring entry. Retrieval
quality is a real problem, and it is upstream of everything else.

**Chunking is the decision that matters most.** Look at `chunk_notes()`.
These notes hand you boundaries ("Lecture N, page M:"). Most documents do not,
and then it gets hard.

---

## 2. Swap in real embeddings (20 min, 10 calls)

```powershell
python src/rag.py --gemini
```

One line changes: `LocalEmbedder()` → `GeminiEmbedder()`. The rest of the
system cannot tell. That is what the interface was for.

**Compare "What is the F1 score?" between the two runs.** Bag-of-words
matches **words**; embeddings match **meaning**. Try a question phrased
entirely in synonyms — "how accurate are my positive predictions?" — and
watch bag-of-words fail where embeddings succeed.

Write down one query where they differ. That difference is the entire
argument for vector embeddings, and it is worth more seen than explained.

---

## 3. Connect retrieval to the backend (20 min, 0 calls)

```powershell
uvicorn src.api:app --reload
curl "http://127.0.0.1:8000/retrieve?q=what+is+data+leakage"
```

Two design decisions in `src/api.py`:

**The index is built at start-up, not per request.** Re-embedding the corpus
on every call would be an API call per request before the request does
anything useful.

**`/retrieve` makes no generation call.** It returns chunks and scores. If
retrieval is wrong, no prompt engineering downstream fixes it — and
generating anyway just spends a call to confirm that.

Same envelope as Weeks 12 and 13. **Adding retrieval changed the service's
capability without changing its contract.**

---

## 4. Demo 2 — the agent (20 min, 2 calls)

```powershell
python src/agent.py                    # scripted, 0 calls
python src/agent.py --gemini "What is recall, and what is it for TP=12, FN=4?"
```

**Reason → Act → Observe → Reason again.** Read the trace.

Two tools, chosen because they fail differently:

- `search_notes` can return **nothing** — the agent must cope with an empty
  observation rather than inventing one
- `calculate` can **raise** — the agent must cope with a tool that errors

An agent whose tools always succeed teaches nothing. The whole difficulty of
agents is the step *after* something goes wrong.

Three things in the code worth your attention:

| | Why |
|---|---|
| `MAX_STEPS = 4` | Week 1's "infinite loop" failure mode. Without it the agent runs until your quota does |
| Running out of steps returns **`None`** | Not a partial answer. That would be Week 1's "silent failure" |
| `calculate` refuses anything but arithmetic | `eval` on model-generated text is a remote-code-execution bug waiting for a prompt injection |

Now count: **each step is a generation call.** A 4-step limit means up to 4
calls per question.

---

## 5. Architectural decision (25 min, 0 calls)

`worksheets/architecture-decision.md`. The question is **not** "how do I add
RAG" — it is whether you should.

**Part A: RAG vs fine-tuning vs a bigger context window.**

If your corpus fits in a prompt and rarely changes, **the answer is
"neither"**. Say so and defend it. Adding retrieval to a system that does not
need it means more moving parts, an extra call per request, and a new way to
be confidently wrong.

**Part B: does it need an agent?**

> **Retrieval is not an agent.** `/retrieve` calls a search function every
> time, in a fixed order. Nothing *chooses*. That is rule 4 — chained calls —
> and most "agents" in student projects are rule 4 in a costume.

**Part D: the cost.** Go back to your Week 12 cost model and redo the
per-user-per-month figure with the new per-request call count. An agent
multiplies it by its step limit.

---

## 6. Journal entry (10 min)

`worksheets/journal-14.md`. Final entry before the capstone.

**What are you giving up?** Every choice costs something, and the last
question asks you to look back at Journal #1 — where you answered "does my
system need an agent?" before you had seen one built.

---

## Before you leave

- [ ] RAG pipeline run offline; watched a query retrieve **nothing**
- [ ] Real embeddings run; **one query found where the two differ**
- [ ] Retrieval endpoint running, same envelope as Week 13
- [ ] Agent trace read end to end
- [ ] Counted the calls per question for an agent
- [ ] Architectural decision written — including "neither" if that is the answer
- [ ] Week 12 cost model updated with the new per-request figure
- [ ] Capstone plan updated if RAG or an agent applies
- [ ] `python -m pytest tests` passes (37 tests)
- [ ] Journal entry written
