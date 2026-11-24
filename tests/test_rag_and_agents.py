"""Tests for retrieval, the endpoint and the agent loop. No API calls.

The agent loop takes `ask` as an argument and the embedder is swappable, so
every path here runs without a key. A planning loop you can only test by
spending quota is a planning loop nobody tests.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import MAX_STEPS, Tools, run_agent, scripted_model
from api import app
from embeddings import LocalEmbedder, cosine_similarity, tokenise
from rag import MIN_SCORE, build_index, build_prompt
from vector_store import VectorStore, chunk_notes

NOTES = (ROOT / "data" / "notes.txt").read_text(encoding="utf-8")
client = TestClient(app, raise_server_exceptions=False)


# --- chunking ------------------------------------------------------------

def test_the_notes_split_into_one_chunk_per_entry():
    chunks = chunk_notes(NOTES)
    assert len(chunks) == 8


def test_the_file_header_is_not_a_chunk():
    """It starts with the word 'lecture' but is not a retrievable fact, and
    leaving it in gives every unanswerable query something to match."""
    texts = [chunk.text for chunk in chunk_notes(NOTES)]
    assert not any(text.startswith("LECTURE NOTES") for text in texts)


def test_every_chunk_keeps_its_citation():
    """A chunk without its source cannot be cited, and an answer that cannot
    be traced is the thing RAG exists to prevent."""
    for chunk in chunk_notes(NOTES):
        assert "page" in chunk.text.lower()


# --- embeddings ----------------------------------------------------------

def test_stopwords_are_removed():
    assert "the" not in tokenise("the model and the data")
    assert "model" in tokenise("the model and the data")


def test_cosine_of_identical_vectors_is_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_of_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_handles_an_empty_vector_without_dividing_by_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine_similarity([], []) == 0.0


def test_the_local_embedder_must_be_fitted_first():
    with pytest.raises(RuntimeError):
        LocalEmbedder().embed(["anything"])


def test_the_local_embedder_is_deterministic():
    embedder = LocalEmbedder().fit(["precision and recall", "overfitting"])
    assert embedder.embed(["recall"]) == embedder.embed(["recall"])


# --- retrieval -----------------------------------------------------------

@pytest.fixture(scope="module")
def store():
    return build_index(LocalEmbedder())


def test_a_covered_question_retrieves_the_right_chunk(store):
    hits = store.search("What is recall?", top_k=3, min_score=MIN_SCORE)
    assert hits
    assert "Recall" in hits[0].chunk.text


def test_data_leakage_retrieves_its_own_chunk(store):
    hits = store.search("What is data leakage?", top_k=1, min_score=MIN_SCORE)
    assert "leakage" in hits[0].chunk.text.lower()


def test_an_uncovered_question_retrieves_nothing(store):
    """The most important retrieval test. Without a score floor the model
    gets three irrelevant chunks and answers from them - which is how a RAG
    system invents an answer while appearing to cite a source."""
    assert store.search("What is the capital of France?", min_score=MIN_SCORE) == []


def test_f1_is_not_in_the_corpus_so_retrieves_nothing(store):
    assert store.search("What is the F1 score?", min_score=MIN_SCORE) == []


def test_without_a_score_floor_irrelevant_chunks_come_back(store):
    """Proves the floor is doing the work, not the query."""
    assert len(store.search("What is the capital of France?", min_score=0.0)) > 0


def test_results_are_ordered_best_first(store):
    hits = store.search("overfitting and cross-validation", top_k=4, min_score=0.0)
    assert [h.score for h in hits] == sorted([h.score for h in hits], reverse=True)


def test_top_k_is_respected(store):
    assert len(store.search("lecture", top_k=2, min_score=0.0)) <= 2


def test_an_empty_store_returns_nothing_rather_than_crashing():
    assert VectorStore(LocalEmbedder()).search("anything") == []


# --- prompt construction -------------------------------------------------

def test_retrieved_chunks_are_numbered_in_the_prompt(store):
    hits = store.search("What is recall?", top_k=2, min_score=0.0)
    prompt = build_prompt("What is recall?", hits)
    assert "[1]" in prompt
    assert "RETRIEVED EXCERPTS" in prompt


def test_the_prompt_says_so_when_nothing_was_retrieved():
    prompt = build_prompt("What is the capital of France?", [])
    assert "none" in prompt.lower()


# --- the endpoint --------------------------------------------------------

def test_the_endpoint_keeps_the_week_12_envelope():
    body = client.get("/retrieve", params={"q": "recall"}).json()
    assert set(body) == {"ok", "data", "error"}


def test_retrieve_returns_scored_results():
    data = client.get("/retrieve", params={"q": "What is recall?"}).json()["data"]
    assert data["retrieved"] >= 1
    assert "Recall" in data["results"][0]["text"]
    assert 0 <= data["results"][0]["score"] <= 1


def test_retrieve_returns_an_empty_list_not_an_error():
    """An empty result is a valid answer, not a failure."""
    response = client.get("/retrieve", params={"q": "capital of France"})
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["results"] == []
    assert "does not cover it" in data["note"]


def test_retrieve_makes_no_generation_call():
    note = client.get("/retrieve", params={"q": "recall"}).json()["data"]["note"]
    assert "No generation call" in note


def test_an_empty_query_is_rejected_with_the_field_named():
    response = client.get("/retrieve", params={"q": ""})
    assert response.status_code == 422
    assert response.json()["error"]["field"] == "q"


def test_top_k_is_bounded():
    assert client.get("/retrieve", params={"q": "x", "top_k": 99}).status_code == 422


def test_health_reports_the_index_size():
    assert client.get("/health").json()["data"]["indexed_chunks"] == 8


# --- the agent -----------------------------------------------------------

ANSWER_NOW = '{"thought": "I know this.", "answer": "42"}'


def test_the_agent_answers_without_a_tool_when_it_can():
    answer, trace = run_agent("q", scripted_model([ANSWER_NOW]), verbose=False)
    assert answer == "42"
    assert len(trace) == 1


def test_the_agent_calls_a_tool_then_answers():
    replies = [
        '{"thought": "look it up", "action": "search_notes", "action_input": "recall"}',
        '{"thought": "got it", "answer": "Recall is TP/(TP+FN)."}',
    ]
    tools = Tools()
    answer, trace = run_agent("q", scripted_model(replies), tools=tools, verbose=False)

    assert answer == "Recall is TP/(TP+FN)."
    assert tools.calls == [("search_notes", "recall")]
    assert "Recall" in trace[0]["observation"]


def test_the_trace_records_every_step():
    """An agent you cannot watch think is an agent you cannot debug."""
    replies = [
        '{"thought": "search", "action": "search_notes", "action_input": "recall"}',
        '{"thought": "compute", "action": "calculate", "action_input": "12 / 16"}',
        '{"thought": "done", "answer": "0.75"}',
    ]
    _, trace = run_agent("q", scripted_model(replies), verbose=False)

    assert len(trace) == 3
    assert trace[0]["action"] == "search_notes"
    assert trace[1]["observation"] == "0.75"
    assert trace[2]["answer"] == "0.75"


def test_a_tool_returning_nothing_is_an_observation_not_a_crash():
    replies = [
        '{"thought": "search", "action": "search_notes", "action_input": "quantum physics"}',
        '{"thought": "not there", "answer": "The notes do not cover that."}',
    ]
    answer, trace = run_agent("q", scripted_model(replies), verbose=False)
    assert "No matching notes found" in trace[0]["observation"]
    assert answer == "The notes do not cover that."


def test_a_failing_tool_is_an_observation_not_a_crash():
    """The whole difference between an agent and a script is what happens on
    the step after something goes wrong."""
    replies = [
        '{"thought": "compute", "action": "calculate", "action_input": "__import__(\'os\')"}',
        '{"thought": "that failed", "answer": "I could not compute that."}',
    ]
    answer, trace = run_agent("q", scripted_model(replies), verbose=False)
    assert trace[0]["observation"].startswith("Error:")
    assert answer == "I could not compute that."


def test_the_calculator_refuses_anything_that_is_not_arithmetic():
    """eval on model-generated text is a remote code execution bug waiting
    for the day someone prompt-injects the agent."""
    with pytest.raises(ValueError):
        Tools().calculate("__import__('os').system('echo hi')")


def test_the_calculator_does_arithmetic():
    assert Tools().calculate("12 / (12 + 4)") == "0.75"


def test_an_unknown_tool_is_reported_not_executed():
    replies = [
        '{"thought": "try", "action": "send_email", "action_input": "x"}',
        '{"thought": "no such tool", "answer": "I cannot do that."}',
    ]
    _, trace = run_agent("q", scripted_model(replies), verbose=False)
    assert "no tool called" in trace[0]["observation"]


def test_the_loop_stops_at_max_steps():
    """Week 1's 'infinite loop' failure mode. Without this the agent runs
    until your quota does."""
    forever = '{"thought": "again", "action": "search_notes", "action_input": "recall"}'
    answer, trace = run_agent("q", scripted_model([forever]), verbose=False)

    assert answer is None, "running out of steps is a failure, not an answer"
    assert len(trace) == MAX_STEPS + 1
    assert "no answer within" in trace[-1]["error"]


def test_a_non_json_reply_stops_the_loop_and_is_recorded():
    answer, trace = run_agent("q", scripted_model(["Sure! Let me help."]), verbose=False)
    assert answer is None
    assert trace[0]["error"] == "model did not return JSON"


def test_a_fenced_json_reply_is_still_parsed():
    """Models add markdown fences despite being told not to."""
    answer, _ = run_agent("q", scripted_model([f"```json\n{ANSWER_NOW}\n```"]), verbose=False)
    assert answer == "42"
