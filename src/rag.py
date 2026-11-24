"""The retrieval pipeline, end to end.

    python src/rag.py                      # offline, LocalEmbedder, 0 calls
    python src/rag.py --gemini             # real embeddings + generation
    python src/rag.py --gemini "your question"

    query -> embed -> similarity search -> top-k -> inject into prompt -> generate

The demo runs each question twice, with and without retrieval, so the
difference is something you watch rather than something you are told.
"""

import sys
from pathlib import Path

from context_package import ContextPackage
from embeddings import GeminiEmbedder, LocalEmbedder
from vector_store import VectorStore, chunk_notes

ROOT = Path(__file__).parent.parent
NOTES = (ROOT / "data" / "notes.txt").read_text(encoding="utf-8")

TOP_K = 3

# Below this, a chunk is not evidence. Without a floor, a query about
# something the corpus does not contain still returns the three least
# unrelated chunks and the model answers from them - which is how a RAG
# system invents an answer while appearing to cite a source.
MIN_SCORE = 0.05


RAG_PACKAGE = ContextPackage(
    name="rag-tutor",
    authority=(
        "You are a subject tutor. You answer using only the retrieved "
        "excerpts supplied below, and you say so when they do not cover the "
        "question."
    ),
    exemplar=(
        "ANSWER: Recall is TP/(TP+FN) - of the things that were right, how "
        "many did you flag.\n"
        "SOURCE: Lecture 3, page 4"
    ),
    constraints=[
        "Use only the retrieved excerpts. Do not use outside knowledge.",
        "If the excerpts do not answer the question, say NOT IN THE RETRIEVED "
        "EXCERPTS and stop.",
        "Quote the Lecture/page reference from the excerpt you used.",
        "Never cite an excerpt that was not supplied.",
    ],
    rubric=[
        "Every claim traces to a supplied excerpt.",
        "The SOURCE line matches an excerpt actually given.",
        "If nothing was retrieved, the answer says so rather than guessing.",
        "Under 100 words.",
    ],
    metadata={
        "output format": "ANSWER line, then SOURCE line",
        "source of truth": "only the retrieved excerpts in this message",
        "when nothing is retrieved": "say NOT IN THE RETRIEVED EXCERPTS",
    },
)


def build_index(embedder):
    """Chunk the notes and embed them. One API call for GeminiEmbedder."""
    return VectorStore(embedder).add(chunk_notes(NOTES))


def build_prompt(question, hits):
    """Inject the retrieved excerpts into the prompt.

    The excerpts are numbered and delimited. That is not decoration: the
    model has to be able to tell retrieved evidence from the question, and
    the rubric asks it to cite which excerpt it used.
    """
    if not hits:
        return (
            f"QUESTION: {question}\n\n"
            "RETRIEVED EXCERPTS: none - nothing in the corpus scored above "
            "the relevance threshold."
        )

    excerpts = "\n\n".join(
        f"[{index}] (score {hit.score:.3f}) {hit.chunk.text}"
        for index, hit in enumerate(hits, 1)
    )
    return f"QUESTION: {question}\n\nRETRIEVED EXCERPTS:\n{excerpts}"


def answer(question, store, client=None, top_k=TOP_K):
    """The full pipeline. Returns (answer_text, hits).

    With no client, it stops after retrieval and reports what it found -
    which is the cheap way to debug a RAG system. If retrieval is returning
    the wrong chunks, no amount of prompt engineering downstream will fix it,
    and generating anyway just spends a call to confirm that.
    """
    hits = store.search(question, top_k=top_k, min_score=MIN_SCORE)

    if client is None:
        return None, hits

    text = client.ask(build_prompt(question, hits), system=RAG_PACKAGE.build_prompt())
    return text, hits


DEMO_QUESTIONS = [
    "What is recall?",
    "How does a train/test split help with overfitting?",
    "What is data leakage?",
    "What is the F1 score?",          # not in the corpus - should retrieve nothing
    "What is the capital of France?",  # not in the corpus at all
]


def main(argv):
    use_gemini = "--gemini" in argv
    arguments = [a for a in argv[1:] if not a.startswith("--")]
    questions = arguments or DEMO_QUESTIONS

    embedder = GeminiEmbedder() if use_gemini else LocalEmbedder()
    print(f"embedder: {embedder.name}")

    store = build_index(embedder)
    print(f"indexed {len(store)} chunks\n")

    client = None
    if use_gemini:
        from gemini_client import GeminiClient

        client = GeminiClient()

    for question in questions:
        print("=" * 70)
        print(f"Q: {question}")
        print("=" * 70)

        text, hits = answer(question, store, client=client)

        print(f"retrieved {len(hits)} chunk(s):")
        for hit in hits:
            print(f"  {hit}")

        if not hits:
            print("  (nothing above the relevance threshold - this is a RESULT,")
            print("   not a bug. The right answer here is 'I do not know'.)")

        if text:
            print(f"\nwith retrieval:\n{text}")

            # The comparison the lab asks for: the same question with no
            # retrieved context at all.
            without = client.ask(
                f"QUESTION: {question}\n\nRETRIEVED EXCERPTS: none",
                system=RAG_PACKAGE.build_prompt(),
            )
            print(f"\nwithout retrieval:\n{without}")
        print()

    if not use_gemini:
        print("Run again with --gemini for real embeddings and generation.")
        print("Compare the 'What is the F1 score?' result between the two:")
        print("bag-of-words matches WORDS, embeddings match MEANING.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
