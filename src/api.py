"""The retrieval endpoint, added to the Week 13 service.

    uvicorn src.api:app --reload
    curl "http://127.0.0.1:8000/retrieve?q=what+is+recall"

Same envelope as Weeks 12 and 13. The whole point of this file is that
adding retrieval changed the service's capability without changing its
contract - a caller written in Week 12 still works.

`/retrieve` makes NO generation call. It returns the chunks and their
scores, which is the cheap way to debug a RAG system: if retrieval is
returning the wrong chunks, no prompt engineering downstream will fix it,
and generating anyway just spends a call to confirm that.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from embeddings import LocalEmbedder
from rag import MIN_SCORE, TOP_K, build_index

VERSION = "3.0.0"

app = FastAPI(
    title="Week 14 - Retrieval Endpoint",
    description="The Week 13 service, with retrieval added. Same envelope.",
    version=VERSION,
)

# Indexed once at start-up, not per request. Re-embedding the corpus on every
# call would be the single most expensive mistake available here - with
# GeminiEmbedder it would mean an API call per request before the request
# does anything useful.
STORE = build_index(LocalEmbedder())


def ok(data):
    return {"ok": True, "data": data, "error": None}


def fail(code, message, field=None):
    error = {"code": code, "message": message}
    if field:
        error["field"] = field
    return {"ok": False, "data": None, "error": error}


@app.get("/health")
def health():
    return ok({"status": "ok", "version": VERSION, "indexed_chunks": len(STORE)})


@app.get("/retrieve", summary="Retrieve relevant chunks. No generation.")
def retrieve(
    q: str = Query(..., min_length=1, max_length=500),
    top_k: int = Query(TOP_K, ge=1, le=10),
    min_score: float = Query(MIN_SCORE, ge=0.0, le=1.0),
):
    """Return the chunks that match, with their scores.

    An empty `results` list is a valid answer, not an error. It means nothing
    in the corpus is relevant - and a RAG system that cannot say that is one
    that will answer anyway.
    """
    hits = STORE.search(q, top_k=top_k, min_score=min_score)

    return ok({
        "query": q,
        "indexed_chunks": len(STORE),
        "retrieved": len(hits),
        "min_score": min_score,
        "results": [
            {
                "score": round(hit.score, 4),
                "text": hit.chunk.text,
                "source": hit.chunk.source,
            }
            for hit in hits
        ],
        "note": (
            "Nothing scored above the threshold. The honest answer to this "
            "question is 'the corpus does not cover it'."
            if not hits else
            "Retrieval only. No generation call was made."
        ),
    })


@app.exception_handler(RequestValidationError)
def handle_validation_error(request: Request, exc: RequestValidationError):
    first = exc.errors()[0]
    field = ".".join(str(part) for part in first["loc"] if part not in ("query", "body"))
    return JSONResponse(
        status_code=422,
        content=fail("validation_error", first["msg"], field=field or None),
    )


@app.exception_handler(StarletteHTTPException)
def handle_http_error(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=fail(f"http_{exc.status_code}", str(exc.detail)),
    )
