"""A retrieval index in one SQLite file: chunks, their embeddings, and a full-text index.

Three ways to search the same chunks:

    vector   cosine similarity on embeddings; finds paraphrases, misses exact identifiers
    keyword  SQLite FTS5 with BM25; finds exact terms, misses paraphrases
    hybrid   reciprocal rank fusion of the two; the default

An embedder is any callable ``(list[str]) -> list[list[float]]``. ``ollama_embedder`` calls a
local Ollama server. ``hashing_embedder`` needs no model and exists for tests and as a baseline.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import urllib.request
from array import array
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

Embedder = Callable[[list[str]], list[list[float]]]
MODES = ("vector", "keyword", "hybrid")
_WORD = re.compile(r"[A-Za-z0-9_]+")
# Common words carry no signal for keyword search and drag unrelated chunks into the results.
_STOP = frozenset(
    "a an and are as at be before by can do does for from get how i if in is it me my of on or "
    "our so than that the their them there this to we what when where which who why will with "
    "you your".split()
)


def ollama_embedder(
    model: str = "nomic-embed-text", url: str = "http://localhost:11434", timeout: int = 120
) -> Embedder:
    """Embed with a model served by a local Ollama (``ollama pull <model>`` first)."""

    def embed(texts: list[str]) -> list[list[float]]:
        body = json.dumps({"model": model, "input": texts}).encode("utf-8")
        request = urllib.request.Request(
            f"{url}/api/embed", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())["embeddings"]

    return embed


def hashing_embedder(dim: int = 256) -> Embedder:
    """Bag-of-words hashed into ``dim`` buckets. Deterministic, offline, and deliberately weak:
    it only matches shared words, so it shows what a real embedding model adds."""

    def embed(texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * dim
            for word in _WORD.findall(text.lower()):
                bucket = int.from_bytes(hashlib.md5(word.encode()).digest()[:4], "big") % dim
                vector[bucket] += 1.0
            vectors.append(vector)
        return vectors

    return embed


def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> list[str]:
    """Split text into chunks of at most ``max_chars``, breaking at paragraphs where possible.

    A paragraph longer than ``max_chars`` is cut into windows that overlap by ``overlap``
    characters, so a sentence on a boundary is still whole in one of them.
    """
    chunks, current = [], ""
    for paragraph in (p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()):
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}" if current else paragraph
            continue
        if current:
            chunks.append(current)
        current = paragraph
        while len(current) > max_chars:
            chunks.append(current[:max_chars])
            current = current[max_chars - overlap :]
    if current:
        chunks.append(current)
    return chunks


def _unit(vector: list[float]) -> bytes:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return array("f", (v / norm for v in vector)).tobytes()


class Index:
    """One corpus in one SQLite file. Use ``":memory:"`` for a throwaway index."""

    def __init__(self, path: str | Path, embedder: Embedder) -> None:
        self.embedder = embedder
        self.db = sqlite3.connect(str(path))
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks(
                id TEXT PRIMARY KEY, doc TEXT NOT NULL, text TEXT NOT NULL, vec BLOB NOT NULL);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(id UNINDEXED, text);
            """
        )

    def add(self, docs: Iterable[tuple[str, str]], *, max_chars: int = 800, batch: int = 32) -> int:
        """Chunk, embed and store ``(doc_id, text)`` pairs. Re-adding a doc replaces it."""
        rows = []
        for doc_id, text in docs:
            self.db.execute(
                "DELETE FROM chunks_fts WHERE id IN (SELECT id FROM chunks WHERE doc=?)", (doc_id,)
            )
            self.db.execute("DELETE FROM chunks WHERE doc=?", (doc_id,))
            rows += [
                (f"{doc_id}#{i}", doc_id, c) for i, c in enumerate(chunk_text(text, max_chars))
            ]
        for start in range(0, len(rows), batch):
            part = rows[start : start + batch]
            vectors = self.embedder([text for _, _, text in part])
            self.db.executemany(
                "INSERT INTO chunks VALUES (?,?,?,?)",
                [
                    (cid, doc, text, _unit(v))
                    for (cid, doc, text), v in zip(part, vectors, strict=True)
                ],
            )
            self.db.executemany("INSERT INTO chunks_fts VALUES (?,?)", [(c, t) for c, _, t in part])
        self.db.commit()
        return len(rows)

    def count(self) -> int:
        """Number of chunks stored."""
        return self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def _vector(self, query: str, limit: int) -> list[str]:
        q = array("f")
        q.frombytes(_unit(self.embedder([query])[0]))
        scored = []
        # ponytail: brute-force scan in pure Python, fine to roughly 50k chunks;
        # move to numpy or sqlite-vec when a corpus outgrows that.
        for chunk_id, blob in self.db.execute("SELECT id, vec FROM chunks"):
            v = array("f")
            v.frombytes(blob)
            scored.append((sum(a * b for a, b in zip(q, v, strict=True)), chunk_id))
        scored.sort(reverse=True)
        return [chunk_id for _, chunk_id in scored[:limit]]

    def _keyword(self, query: str, limit: int) -> list[str]:
        words = [w for w in _WORD.findall(query) if w.lower() not in _STOP]
        if not words:
            return []
        match = " OR ".join(f'"{w}"' for w in words)
        rows = self.db.execute(
            "SELECT id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
            (match, limit),
        )
        return [row[0] for row in rows]

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[dict[str, Any]]:
        """Return the top ``k`` chunks as dicts with ``id``, ``doc``, ``text`` and ``rank``."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        pool = max(k, 20)
        if mode == "vector":
            ids = self._vector(query, k)
        elif mode == "keyword":
            ids = self._keyword(query, k)
        else:  # reciprocal rank fusion: a chunk scores 1/(60+rank) in each list it appears in
            fused: dict[str, float] = {}
            for ranking in (self._vector(query, pool), self._keyword(query, pool)):
                for rank, chunk_id in enumerate(ranking, 1):
                    fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (60 + rank)
            ids = sorted(fused, key=lambda c: (-fused[c], c))[:k]
        hits = []
        for rank, chunk_id in enumerate(ids, 1):
            doc, text = self.db.execute(
                "SELECT doc, text FROM chunks WHERE id=?", (chunk_id,)
            ).fetchone()
            hits.append({"rank": rank, "id": chunk_id, "doc": doc, "text": text})
        return hits

    def context(self, query: str, k: int = 4, mode: str = "hybrid") -> str:
        """Build a grounded prompt: the retrieved passages, each tagged with its source id."""
        passages = "\n\n".join(f"[{h['id']}]\n{h['text']}" for h in self.search(query, k, mode))
        return (
            "Answer the question using only the passages below. Cite the passage ids you used "
            "in square brackets. If the passages do not contain the answer, say so.\n\n"
            f"{passages}\n\nQuestion: {query}\n"
        )


def evaluate(
    index: Index, queries: list[dict[str, Any]], k: int = 5
) -> dict[str, dict[str, float]]:
    """Score every search mode on labelled queries.

    Each query is ``{"query": ..., "relevant": [doc ids]}``. Reports, per mode, recall@k (share
    of queries with a relevant doc in the top k) and MRR (mean of 1/rank of the first relevant
    doc, 0 when none is found).
    """
    report = {}
    for mode in MODES:
        hits_at_k, reciprocal = 0, 0.0
        for q in queries:
            docs = [h["doc"] for h in index.search(q["query"], k, mode)]
            first = next((i for i, d in enumerate(docs, 1) if d in q["relevant"]), None)
            if first:
                hits_at_k += 1
                reciprocal += 1.0 / first
        n = len(queries) or 1
        report[mode] = {"recall": round(hits_at_k / n, 4), "mrr": round(reciprocal / n, 4)}
    return report
