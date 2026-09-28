#!/usr/bin/env python
"""Offline end-to-end demo -- no API key required.

Builds a tiny SQLite database, then runs the real agentic loop
(schema exploration -> SQL -> execute -> self-correct) against a scripted
stand-in for the LLM provider. The tool loop, dialect handling, tracing and
result parsing are all real; only the model's HTTP responses are scripted.

Usage:
    python demo_offline.py
"""

from __future__ import annotations

import os
import tempfile
import types
from unittest.mock import patch

from sqlalchemy import create_engine, text as sql_text

from text2sql import TextSQL
from text2sql.tracing import Tracer


# ---------------------------------------------------------------------------
# Scripted provider (replays responses like the test suite's fake client)
# ---------------------------------------------------------------------------


def _text_block(text):
    return types.SimpleNamespace(type="text", text=text)


def _tool_use_block(id, name, input):
    return types.SimpleNamespace(type="tool_use", id=id, name=name, input=input)


class _FakeResponse:
    def __init__(self, content):
        self.content = content
        self.usage = types.SimpleNamespace(input_tokens=100, output_tokens=20)


class _FakeMessages:
    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._scripted.pop(0)


class _FakeClient:
    def __init__(self, scripted):
        self.messages = _FakeMessages(scripted)


def _patch(scripted):
    holder = {}

    def factory(*args, **kwargs):
        client = _FakeClient(scripted)
        holder["client"] = client
        return client

    return patch("anthropic.Anthropic", factory), holder


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def build_db(path: str) -> None:
    engine = create_engine("sqlite:///{}".format(path))
    with engine.connect() as conn:
        conn.execute(sql_text(
            "CREATE TABLE customers (id INTEGER, name TEXT, region TEXT)"
        ))
        conn.execute(sql_text(
            "CREATE TABLE orders (id INTEGER, customer_id INTEGER, total REAL)"
        ))
        conn.execute(sql_text(
            "INSERT INTO customers VALUES "
            "(1,'Alice','Northeast'),(2,'Bob','West'),(3,'Cara','Northeast')"
        ))
        conn.execute(sql_text(
            "INSERT INTO orders VALUES "
            "(1,1,250.0),(2,1,300.0),(3,2,120.0),(4,3,900.0)"
        ))
        conn.commit()


def main() -> None:
    print("Agentic Text-to-SQL -- offline demo (scripted provider, real agent loop)\n")

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    build_db(db_path)

    # The "LLM" explores the schema first, then answers -- same loop it runs live.
    scripted = [
        _FakeResponse([
            _text_block("Let me look at the schema first."),
            _tool_use_block("t1", "execute_sql", {
                "sql": "SELECT name FROM sqlite_master WHERE type='table'"
            }),
        ]),
        _FakeResponse([
            _text_block("Orders reference customers; checking the join columns."),
            _tool_use_block("t2", "execute_sql", {"sql": "PRAGMA table_info(orders)"}),
        ]),
        _FakeResponse([
            _text_block(
                "Top customer by spend:\n"
                "```sql\n"
                "SELECT c.name, SUM(o.total) AS spend FROM customers c "
                "JOIN orders o ON o.customer_id = c.id "
                "GROUP BY c.name ORDER BY spend DESC LIMIT 1\n"
                "```"
            ),
        ]),
    ]

    patcher, _ = _patch(scripted)
    with patcher:
        engine = TextSQL(
            "sqlite:///{}".format(db_path),
            model="anthropic:claude-sonnet-4-6",
            trace_file=None,
        )
        engine.generator.tracer = Tracer()
        result = engine.ask("Who is the top customer by total spend?")

    trace = engine.generator.tracer.traces[-1]
    print("agent tool calls:")
    for call in trace.tool_calls:
        print("  -", call.name, "->", str(call.arguments.get("sql"))[:90])
    print("\nfinal SQL:\n ", result.sql)
    print("\nresult rows:", result.data)
    print("success:", result.success, "| tool calls:", trace.total_tool_calls)

    os.unlink(db_path)
    print("\nDone. Set ANTHROPIC_API_KEY or OPENAI_API_KEY to run this live.")


if __name__ == "__main__":
    main()
