"""Command line for pocket-rag.

pocket-rag index docs.db examples/orbit/docs.jsonl     index a JSONL file or a folder of text
pocket-rag search docs.db "how do I rotate a key"      top chunks (hybrid by default)
pocket-rag eval docs.db examples/orbit/queries.jsonl   recall and MRR for every search mode
pocket-rag ask docs.db "question" --llm "ollama run llama3.2"
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from pocket_rag import core


def make_embedder(spec: str) -> core.Embedder:
    """``hashing`` or ``ollama:<model>``."""
    if spec == "hashing":
        return core.hashing_embedder()
    kind, _, model = spec.partition(":")
    if kind == "ollama" and model:
        return core.ollama_embedder(model)
    raise ValueError(f"unknown embedder {spec!r}; use hashing or ollama:<model>")


def read_docs(path: Path) -> list[tuple[str, str]]:
    """Load ``(id, text)`` pairs from a JSONL file or from every .md/.txt file under a folder."""
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in (".md", ".txt"))
        return [(p.relative_to(path).as_posix(), p.read_text(encoding="utf-8")) for p in files]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [(row["id"], row["text"]) for row in rows]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def open_index(db: str, embedder: str | None) -> core.Index:
    """Open an index with the embedder it was built with; mixing models would corrupt search."""
    index = core.Index(db, lambda texts: [])
    index.db.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
    stored = index.db.execute("SELECT value FROM meta WHERE key='embedder'").fetchone()
    if stored and embedder and stored[0] != embedder:
        raise ValueError(f"{db} was built with {stored[0]}; rebuild it to use {embedder}")
    spec = stored[0] if stored else (embedder or "ollama:nomic-embed-text")
    index.db.execute("INSERT OR REPLACE INTO meta VALUES ('embedder', ?)", (spec,))
    index.db.commit()
    index.embedder = make_embedder(spec)
    return index


def format_eval(report: dict[str, dict[str, float]], k: int) -> str:
    """Render an evaluation report as a Markdown table."""
    lines = [f"| mode | recall@{k} | MRR |", "|---|---|---|"]
    lines += [f"| {mode} | {m['recall']:.3f} | {m['mrr']:.3f} |" for mode, m in report.items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(prog="pocket-rag", description="Local hybrid retrieval.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("index", "search", "eval", "ask"):
        p = sub.add_parser(name)
        p.add_argument("db", help="path to the index file")
        p.add_argument("--embedder", help="hashing or ollama:<model> (default: what the db uses)")
        if name == "index":
            p.add_argument("source", help="a .jsonl file of {id, text} or a folder of .md/.txt")
            p.add_argument("--max-chars", type=int, default=800, help="chunk size")
        elif name == "eval":
            p.add_argument("queries", help="a .jsonl file of {query, relevant}")
            p.add_argument("--by", help="also break results down by this field of each query")
        else:
            p.add_argument("query")
        if name != "index":
            p.add_argument("-k", type=int, default=5, help="results to return")
        if name in ("search", "ask"):
            p.add_argument("--mode", choices=core.MODES, default="hybrid")
        if name == "ask":
            p.add_argument("--llm", help="command that reads the prompt on stdin and answers")
    args = parser.parse_args(argv)

    try:
        index = open_index(args.db, args.embedder)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.command == "index":
        docs = read_docs(Path(args.source))
        chunks = index.add(docs, max_chars=args.max_chars)
        print(f"indexed {len(docs)} documents as {chunks} chunks ({index.count()} in index)")
    elif args.command == "search":
        for hit in index.search(args.query, args.k, args.mode):
            print(f"{hit['rank']}. [{hit['id']}] {' '.join(hit['text'].split())[:160]}")
    elif args.command == "eval":
        queries = read_jsonl(Path(args.queries))
        overall = format_eval(core.evaluate(index, queries, args.k), args.k)
        print(f"{len(queries)} queries")
        print()
        print(overall)
        for value in sorted({q[args.by] for q in queries}) if args.by else []:
            part = [q for q in queries if q[args.by] == value]
            report = core.evaluate(index, part, args.k)
            print(f"\n{args.by} = {value} ({len(part)} queries)\n\n{format_eval(report, args.k)}")
    else:
        prompt = index.context(args.query, args.k, args.mode)
        if not args.llm:
            print(prompt)
        else:
            done = subprocess.run(
                shlex.split(args.llm), input=prompt, text=True, encoding="utf-8", errors="replace"
            )
            return done.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
