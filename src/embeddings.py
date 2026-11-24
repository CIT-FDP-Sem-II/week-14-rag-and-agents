"""Turning text into vectors. Two embedders, same interface.

    LocalEmbedder   deterministic, offline, FREE - develop and test with this
    GeminiEmbedder  real semantic embeddings - 1 API call per batch

Build the pipeline with LocalEmbedder, get the architecture right, then swap
one line to use GeminiEmbedder. The rest of the system cannot tell the
difference, which is the point of having an interface at all.

LocalEmbedder is a bag-of-words vector with cosine similarity. It genuinely
retrieves - "what is recall" finds the chunk containing "recall" - but it
matches WORDS, not MEANING. Ask it about "accuracy of positive predictions"
and it will not find the precision chunk, because the words do not overlap.

That gap is the entire argument for real embeddings, and you should see it
happen rather than be told about it. The lab makes you run both.
"""

import math
import re
from collections import Counter

MODEL = "models/text-embedding-004"

# Words that appear everywhere and carry no retrieval signal. Without this,
# every chunk matches every query on "the" and "a".
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "how", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this",
    "to", "was", "what", "when", "which", "with", "you", "your",
}


def tokenise(text):
    """Lowercase words, stopwords removed."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [word for word in words if word not in STOPWORDS and len(word) > 1]


def cosine_similarity(a, b):
    """How close two vectors point. 1.0 identical, 0.0 unrelated.

    Cosine rather than distance because it ignores magnitude: a long chunk
    and a short query about the same thing should still match.
    """
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class LocalEmbedder:
    """Bag-of-words vectors. No API calls, deterministic, free.

    The vocabulary is built from the documents, so this must be `fit` on the
    corpus before anything is embedded. That is a real constraint of the
    approach and is why `GeminiEmbedder.fit` exists but does nothing - a real
    embedding model has no vocabulary to learn from you.
    """

    name = "local (bag-of-words)"
    uses_api = False

    def __init__(self):
        self.vocabulary = {}

    def fit(self, documents):
        """Build the vocabulary from the corpus."""
        words = sorted({word for document in documents for word in tokenise(document)})
        self.vocabulary = {word: index for index, word in enumerate(words)}
        return self

    def embed(self, texts):
        """Return one vector per text. Call fit() first."""
        if not self.vocabulary:
            raise RuntimeError("LocalEmbedder.fit(documents) must be called first")

        vectors = []
        for text in texts:
            counts = Counter(tokenise(text))
            vector = [0.0] * len(self.vocabulary)
            for word, count in counts.items():
                index = self.vocabulary.get(word)
                if index is not None:
                    vector[index] = float(count)
            vectors.append(vector)
        return vectors


class GeminiEmbedder:
    """Real semantic embeddings from the Gemini API.

    Costs one API call per `embed` call, not per text - the SDK batches. Eight
    chunks is one call, so indexing the whole notes file is cheap. Queries are
    one call each, which is the number to watch: a RAG endpoint spends one
    embedding call on EVERY request, before it spends a generation call.
    """

    name = "gemini (semantic)"
    uses_api = True

    def __init__(self, client=None):
        self._client = client

    def _get_client(self):
        if self._client is None:
            import os

            from dotenv import load_dotenv
            from google import genai

            load_dotenv()
            key = os.environ.get("GEMINI_API_KEY", "").strip()
            if not key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. Copy .env.example to .env, or "
                    "use LocalEmbedder, which needs no key."
                )
            self._client = genai.Client(api_key=key)
        return self._client

    def fit(self, documents):
        """No-op. A real embedding model has no vocabulary to learn from you.

        It exists so that LocalEmbedder and GeminiEmbedder are swappable
        without the calling code needing to know which one it has.
        """
        return self

    def embed(self, texts):
        response = self._get_client().models.embed_content(model=MODEL, contents=texts)
        return [item.values for item in response.embeddings]
