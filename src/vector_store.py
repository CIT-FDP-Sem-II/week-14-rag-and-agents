"""An in-memory vector store: add chunks, search by similarity, return top-k.

A real vector database (Chroma, pinecone, pgvector) adds persistence, scale
and filtering. It does not add a different idea - this is the whole concept
in 70 lines, and seeing it this small is worth more than configuring one.

What a real store buys you, and when:

    persistence   you do not re-embed 10,000 documents on every restart
    scale         exact search is O(n); above ~100k vectors you need an index
    filtering     "only chunks from lecture 4" without scanning everything
    concurrency   more than one process reading and writing

Below those thresholds, this is not a toy - it is the right tool.
"""

import re
from dataclasses import dataclass, field

from embeddings import cosine_similarity

# What the start of a retrievable entry looks like: "Lecture 3, page 4:".
ENTRY_START = re.compile(r"^\s*lecture\s+\d+\s*,\s*page\s+\d+\s*:", re.IGNORECASE)


@dataclass
class Chunk:
    """One retrievable piece of text, with where it came from."""

    text: str
    source: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Hit:
    """A chunk and how well it matched."""

    chunk: Chunk
    score: float

    def __str__(self):
        preview = self.chunk.text[:60].replace("\n", " ")
        return f"[{self.score:.3f}] {self.chunk.source}: {preview}..."


class VectorStore:
    """Chunks plus their vectors, searchable by cosine similarity."""

    def __init__(self, embedder):
        self.embedder = embedder
        self.chunks = []
        self.vectors = []

    def add(self, chunks):
        """Embed and store. One API call for the whole batch, if applicable."""
        if not chunks:
            return self
        self.embedder.fit([chunk.text for chunk in chunks])
        self.vectors = self.embedder.embed([chunk.text for chunk in chunks])
        self.chunks = list(chunks)
        return self

    def search(self, query, top_k=3, min_score=0.0):
        """Return the top_k most similar chunks, best first.

        min_score is the part most tutorials leave out. Without it, a query
        about something the corpus does not contain still returns three
        chunks - the three LEAST unrelated ones - and the model dutifully
        answers from them. That is how a RAG system invents an answer while
        appearing to cite a source.

        Returning nothing is a valid and often correct retrieval result.
        """
        if not self.chunks:
            return []

        query_vector = self.embedder.embed([query])[0]
        scored = [
            Hit(chunk, cosine_similarity(query_vector, vector))
            for chunk, vector in zip(self.chunks, self.vectors)
        ]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return [hit for hit in scored[:top_k] if hit.score >= min_score]

    def __len__(self):
        return len(self.chunks)


def chunk_notes(text, source="notes"):
    """Split the lecture notes into one chunk per line-entry.

    Chunking strategy is the decision that most affects RAG quality, and it
    is almost entirely about where the boundaries fall:

        too small   "Precision = TP / (TP + FP)" retrieves without the
                    context that makes it mean anything
        too large   the whole file is one chunk, every query matches it, and
                    retrieval has achieved nothing
        badly cut   a definition split from its own formula

    These notes have one idea per entry, already marked by the "Lecture N,
    page M:" prefix, so the structure of the document gives us the
    boundaries. When a document does not hand you boundaries like that,
    chunking gets much harder and is worth real thought.
    """
    chunks = []
    current = []

    for line in text.splitlines():
        if ENTRY_START.match(line) and current:
            chunks.append(" ".join(current).strip())
            current = [line.strip()]
        elif line.strip():
            current.append(line.strip())

    if current:
        chunks.append(" ".join(current).strip())

    # Keep only real entries. The file's title line - "LECTURE NOTES -
    # Introduction to Machine Learning" - starts with the word "lecture" but
    # is not a retrievable fact, and leaving it in gives every unanswerable
    # query a chunk to match weakly against.
    return [
        Chunk(text=body, source=source, metadata={"index": index})
        for index, body in enumerate(entry for entry in chunks if ENTRY_START.match(entry))
    ]
