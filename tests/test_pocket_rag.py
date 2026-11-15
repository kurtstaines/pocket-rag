"""Tests run offline: the hashing embedder and a local stub stand in for a real model."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from pocket_rag import cli, core

DOCS = [
    (
        "limits",
        "Each workspace may send 600 requests per minute. Beyond that the API returns E4291.",
    ),
    (
        "rotate",
        "To replace a credential without downtime, create a second key and revoke the old one.",
    ),
    ("upload", "Files are uploaded in parts. Parts may be sent in parallel and in any order."),
]
ORBIT = Path(__file__).resolve().parents[1] / "examples" / "orbit"


@pytest.fixture
def index() -> core.Index:
    idx = core.Index(":memory:", core.hashing_embedder())
    idx.add(DOCS)
    return idx


# --- chunking ----------------------------------------------------------------


def test_chunk_text_keeps_paragraphs_together_and_respects_the_limit():
    text = "one " * 30 + "\n\n" + "two " * 30 + "\n\n" + "three " * 30
    chunks = core.chunk_text(text, max_chars=260)
    assert len(chunks) == 2 and all(len(c) <= 260 for c in chunks)
    assert "one" in chunks[0] and "two" in chunks[0] and "three" in chunks[1]


def test_chunk_text_splits_an_oversized_paragraph_with_overlap():
    chunks = core.chunk_text("x" * 250, max_chars=100, overlap=20)
    assert [len(c) for c in chunks] == [100, 100, 90]
    assert core.chunk_text("  \n\n  ") == []


# --- indexing ----------------------------------------------------------------


def test_add_stores_chunks_and_readding_a_doc_replaces_it(index: core.Index):
    assert index.count() == 3
    index.add([("limits", "The allowance is now 900 calls per minute.")])
    assert index.count() == 3
    assert "900" in index.search("allowance", 1, "keyword")[0]["text"]
    assert index.search("600", 3, "keyword") == []  # the old text is gone from full-text too


def test_long_documents_become_several_chunks_with_stable_ids():
    idx = core.Index(":memory:", core.hashing_embedder())
    assert idx.add([("long", "alpha " * 100 + "\n\n" + "beta " * 100)], max_chars=300) > 1
    ids = [h["id"] for h in idx.search("beta", 20, "keyword")]
    assert ids and all(i.startswith("long#") for i in ids)


def test_index_persists_to_a_file(tmp_path: Path):
    path = tmp_path / "docs.db"
    core.Index(path, core.hashing_embedder()).add(DOCS)
    reopened = core.Index(path, core.hashing_embedder())
    assert reopened.count() == 3 and reopened.search("credential", 1)[0]["doc"] == "rotate"


def test_index_can_be_searched_from_another_thread(index: core.Index):
    found = []
    worker = threading.Thread(target=lambda: found.extend(index.search("credential", 1)))
    worker.start()
    worker.join()
    assert found[0]["doc"] == "rotate"


# --- search ------------------------------------------------------------------


@pytest.mark.parametrize("mode", core.MODES)
def test_each_mode_finds_the_matching_document(index: core.Index, mode: str):
    hits = index.search("credential downtime", 2, mode)
    assert hits[0]["doc"] == "rotate" and hits[0]["rank"] == 1 and hits[0]["id"] == "rotate#0"


def test_keyword_search_finds_exact_identifiers_and_ignores_stop_words(index: core.Index):
    assert index.search("what is E4291", 1, "keyword")[0]["doc"] == "limits"
    assert index.search("what is the", 3, "keyword") == []  # nothing but stop words
    assert index.search("?!", 3, "keyword") == []


def test_keyword_query_cannot_inject_fts_syntax(index: core.Index):
    assert index.search('parts" OR "x NEAR(', 3, "keyword")[0]["doc"] == "upload"


def test_hybrid_fuses_both_rankings():
    """A chunk ranked first by one branch and absent from the other still surfaces."""

    def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "semantic" in t or "meaning" in t else [0.0, 1.0] for t in texts]

    idx = core.Index(":memory:", embed)
    idx.add([("a", "semantic match only"), ("b", "contains the token ZX81"), ("c", "unrelated")])
    docs = [h["doc"] for h in idx.search("meaning of ZX81", 2, "hybrid")]
    assert set(docs) == {"a", "b"}


def test_search_rejects_unknown_mode(index: core.Index):
    with pytest.raises(ValueError):
        index.search("x", 1, "magic")


def test_context_cites_sources_and_tells_the_model_to_stay_grounded(index: core.Index):
    prompt = index.context("credential downtime", k=1)
    assert "[rotate#0]" in prompt and "only the passages" in prompt
    assert prompt.rstrip().endswith("Question: credential downtime")


# --- evaluation --------------------------------------------------------------


def test_evaluate_reports_recall_and_mrr(index: core.Index):
    queries = [
        {"query": "E4291", "relevant": ["limits"]},
        {"query": "credential downtime", "relevant": ["rotate"]},
        {"query": "parallel parts", "relevant": ["nonexistent"]},
    ]
    report = core.evaluate(index, queries, k=3)
    assert set(report) == set(core.MODES)
    assert report["keyword"] == {"recall": 0.6667, "mrr": 0.6667}


def test_sample_corpus_is_consistent_and_keyword_wins_on_identifiers():
    """Every labelled query points at real documents, and exact-identifier queries are
    answered by keyword search without any embedding model."""
    docs = cli.read_docs(ORBIT / "docs.jsonl")
    queries = cli.read_jsonl(ORBIT / "queries.jsonl")
    ids = {doc_id for doc_id, _ in docs}
    assert len(ids) == len(docs) == 30
    assert all(set(q["relevant"]) <= ids and q["kind"] in ("exact", "paraphrase") for q in queries)
    idx = core.Index(":memory:", core.hashing_embedder())
    idx.add(docs)
    exact = [q for q in queries if q["kind"] == "exact"]
    assert core.evaluate(idx, exact, k=5)["keyword"]["recall"] == 1.0


# --- embedders ---------------------------------------------------------------


def test_hashing_embedder_is_deterministic_and_word_based():
    embed = core.hashing_embedder(64)
    a, b, c = embed(["rotate the key", "key rotate the", "something else"])
    assert a == b and a != c and len(a) == 64


def test_ollama_embedder_posts_a_batch_and_returns_vectors():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = json.dumps({"embeddings": [[1.0, 2.0]] * len(seen["body"]["input"])}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        embed = core.ollama_embedder("some-model", f"http://127.0.0.1:{server.server_port}")
        assert embed(["a", "b"]) == [[1.0, 2.0], [1.0, 2.0]]
    finally:
        server.shutdown()
        server.server_close()
    assert seen["path"] == "/api/embed"
    assert seen["body"] == {"model": "some-model", "input": ["a", "b"]}


# --- CLI ---------------------------------------------------------------------


def test_cli_index_search_eval_and_ask(tmp_path: Path, capsys: pytest.CaptureFixture):
    db = str(tmp_path / "orbit.db")
    assert cli.main(["index", db, str(ORBIT / "docs.jsonl"), "--embedder", "hashing"]) == 0
    assert "indexed 30 documents as 30 chunks" in capsys.readouterr().out
    assert cli.main(["search", db, "E4291", "-k", "1", "--mode", "keyword"]) == 0
    assert "[rate-limits#0]" in capsys.readouterr().out
    assert cli.main(["eval", db, str(ORBIT / "queries.jsonl"), "--by", "kind"]) == 0
    out = capsys.readouterr().out
    assert "34 queries" in out and "| hybrid |" in out and "kind = paraphrase (19 queries)" in out
    assert cli.main(["ask", db, "E4291", "-k", "1"]) == 0  # no --llm: prints the prompt
    assert "Question: E4291" in capsys.readouterr().out


def test_cli_indexes_a_folder_and_refuses_a_different_embedder(tmp_path: Path, capsys):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "a.md").write_text("Rotate keys every ninety days.", encoding="utf-8")
    (tmp_path / "notes" / "skip.png").write_bytes(b"not text")
    db = str(tmp_path / "notes.db")
    assert cli.main(["index", db, str(tmp_path / "notes"), "--embedder", "hashing"]) == 0
    assert "indexed 1 documents" in capsys.readouterr().out
    assert cli.main(["search", db, "rotate"]) == 0  # reuses the embedder stored in the file
    assert "[a.md#0]" in capsys.readouterr().out
    assert cli.main(["search", db, "rotate", "--embedder", "ollama:other"]) == 2
    assert "was built with hashing" in capsys.readouterr().err
