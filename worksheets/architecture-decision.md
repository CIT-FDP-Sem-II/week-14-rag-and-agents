# Worksheet — Architectural Decision

**For your domain system: does it need RAG, an agent, or neither?**
One paragraph with justification. (25 minutes, no code)

This is the Week 1 decision matrix again, now that you have seen the
mechanisms rather than just heard about them.

---

## Part A — RAG, fine-tuning, or a bigger context window?

Three ways to get the model to know something it does not know. They are not
interchangeable.

| | Use when | Cost | Changes when the data changes |
|---|---|---|---|
| **Bigger context window** | The whole corpus fits, and is small | Tokens on **every** call | Nothing — you just paste the new version |
| **RAG** | The corpus is too big to send, or changes often | One embedding call per query, plus indexing | Re-index the changed chunks |
| **Fine-tuning** | You need a *behaviour* or *format*, not facts | Training run, then cheaper calls | Retrain |

### The question people get wrong

> *"Our documentation changes weekly, so we fine-tuned on it."*

Fine-tuning teaches **behaviour**, not **facts**. A model fine-tuned on your
docs will sound like your docs and still invent the version number.

### Decide for your system

| | Your answer |
|---|---|
| How big is your corpus, in characters? | |
| Divided by 4 — roughly how many tokens? | |
| Does it fit in one prompt? | |
| How often does it change? | |
| Does the answer need a **citation**? | |
| **Decision** | |

**If it fits in a prompt and rarely changes, the answer is "neither".** Say so
and defend it. Adding retrieval to a system that does not need it is a real
engineering decision in the wrong direction — more moving parts, an extra API
call per request, and a new way to fail.

The citation question is often decisive on its own. Retrieval gives you a
chunk you can quote; a big context window gives you an answer you have to
trust.

---

## Part B — Does it need an agent?

The Week 1 matrix, unchanged. Check top to bottom; first match wins.

| # | If the component… | Then |
|---|---|---|
| 1 | is high-stakes or irreversible | human-in-the-loop, **not** an agent |
| 2 | needs external data or tools | agent with tool use |
| 3 | has steps that depend on earlier output | agent with a planning loop |
| 4 | is a multi-step **fixed** sequence | chained calls |
| 5 | is a single well-defined task | one direct call |

> **Retrieval is not an agent.** `/retrieve` calls a search function on every
> request, in a fixed order. Nothing chooses. That is rule 4, and most
> "agents" in student projects are rule 4 wearing a costume.
>
> It becomes rule 2 when the model decides *whether* to search, *what* to
> search for, and *whether the result was good enough to search again*.

### Decide for your system

| Component | Rule | Verdict |
|---|---|---|
| | | |
| | | |
| | | |

---

## Part C — The one-paragraph decision

> Write it. Name the component, the rule, and what you are giving up.

---

## Part D — If RAG or an agent applies, how does it connect?

### To the backend endpoint from Week 13

| | Your answer |
|---|---|
| Which endpoint? | |
| Is the index built at **start-up** or per request? | |
| What is the **score floor**, and why that number? | |
| What happens when retrieval returns **nothing**? | |

That last row is the one to get right. A RAG system that cannot say "the
corpus does not cover this" will answer anyway, from whatever it retrieved,
and cite it.

### Cost, per request

| | Calls |
|---|---|
| Embedding the query | |
| Generation | |
| Agent: generation **per step** | |
| **Total, worst case** | |

An agent with a 4-step limit costs up to 4 generation calls per question.
Week 12's cost model needs updating — go back and redo the per-user-per-month
figure with this number.

### Failure modes this adds

| New failure | What the user sees | How you would catch it |
|---|---|---|
| Retrieval returns nothing | | |
| Retrieval returns the **wrong** chunk confidently | | |
| Agent loops until the step limit | | |
| A tool errors mid-loop | | |

The second row is the hard one. A retrieval failure that returns *nothing* is
visible. A retrieval failure that returns a plausible but wrong chunk produces
a confident, cited, wrong answer — and the citation makes it **more**
convincing, not less.

> How would you detect that in production? (Week 11's regression dataset is
> the honest answer.)
