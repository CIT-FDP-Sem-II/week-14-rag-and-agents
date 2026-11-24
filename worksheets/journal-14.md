# AI Decision Journal — Final Entry Before Capstone

**Question:** What is your final architecture decision, and what are you
giving up?

Half a page. This is the last journal entry of Year 1.

---

## 1. The decision

> RAG, an agent, both, or neither. Name the component and the rule.

## 2. What you are giving up

> Every choice costs something.
>
> - Chose RAG? You now have an index to keep in step with the source, an
>   extra call per request, and a new failure mode where retrieval is
>   confidently wrong.
> - Chose an agent? Up to N generation calls per question, non-deterministic
>   behaviour, and a trace you have to read to debug anything.
> - Chose neither? Say what the system cannot do as a result.

## 3. What changed your mind this week

> Something should have. The two most common:
>
> - Running the retrieval demo and seeing a query return **nothing** — and
>   realising that is the correct answer, not a bug.
> - Reading the agent trace and noticing how much of it was the model
>   deciding things you could have decided at design time.

## 4. The failure you are most worried about

> And how you would find out it had happened.

## 5. Looking back at Journal #1

> In Week 1 you answered "does my system need an agent?" using the matrix,
> before you had seen one built.
>
> Do you still agree with yourself? What did you not know then?
