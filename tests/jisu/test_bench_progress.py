"""Tests for the benchmark CLI's progress log."""

from __future__ import annotations
import io
import logging
from pathlib import Path
from dataclasses import replace

import pytest

from jisu.bench.memory.runner import BenchConfig, run_memory_bench
from jisu.bench.memory.metrics import QueryResult
from jisu.bench.memory.progress import ProgressPrinter, describe_query, remove_log_handler, install_log_handler


BASE = QueryResult(
    query_id="dog_q",
    text="강아지",
    tags=("short_word",),
    expected=("dog",),
    returned=("dog",),
    rank=1,
    hit_at_1=True,
    hit_at_k=True,
    recall_at_k=1.0,
    false_positive=False,
    latency_ms=12.0,
)
MISS = replace(BASE, rank=None, hit_at_1=False, hit_at_k=False, recall_at_k=0.0)
NEGATIVE = replace(MISS, expected=(), tags=("negative",), returned=())


def test_describe_query_cases() -> None:
    """Pass, miss, negative pass and false positive each read clearly."""
    assert describe_query(BASE, 1, 41) == "[O]  1/41 dog_q '강아지' (short_word) → 1위 · 12ms"

    miss = describe_query(replace(MISS, returned=("a", "b", "c", "d")), 2, 41)
    assert miss.startswith("[X]") and "못 찾음 (정답 dog)" in miss and "나온 것: a, b, c 외 1개" in miss
    empty = describe_query(replace(MISS, returned=(), note="degraded: x"), 3, 41)
    assert "결과 없음" in empty and empty.endswith("[degraded: x]")

    assert "결과 없음 (없는 기억, 정상)" in describe_query(NEGATIVE, 4, 41)
    fp = describe_query(replace(NEGATIVE, returned=("dog",), false_positive=True), 5, 41)
    assert fp.startswith("[X]") and "없는 기억인데 나옴: dog" in fp and "(negative)" not in fp


def test_printer_logs_a_json_run(tmp_path: Path) -> None:
    """A real json run prints the header, subject, save progress, every query and the summary."""
    out = io.StringIO()
    run_memory_bench(
        BenchConfig(subjects=("json",), repeat=1),
        on_event=ProgressPrinter(k=5, repeat=1, stream=out),
        workdir=tmp_path / "work",
    )
    text = out.getvalue()
    assert "기억 검색 벤치마크 · ko_desk_v1" in text
    assert "━━ [1/1] json" in text and "기준선" in text
    assert "저장 30/30" in text
    assert text.count("/41 ") == 41
    assert "결과: hit@5" in text and "틀린 질문" in text


def test_printer_brief_and_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--brief prints only failed queries; a skipped subject says why."""
    monkeypatch.delenv("GOOGLE_GENERATIVE_AI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    out = io.StringIO()
    run_memory_bench(
        BenchConfig(subjects=("json", "gbrain-embed"), repeat=1),
        on_event=ProgressPrinter(k=5, repeat=1, brief=True, stream=out),
        workdir=tmp_path / "work",
        dotenv_path=tmp_path / "missing.env",
    )
    lines = out.getvalue().splitlines()
    query_lines = [line for line in lines if "/41 " in line]
    assert query_lines and all("[X]" in line for line in query_lines)
    assert any("건너뜀: no embedding key" in line for line in lines)


def test_log_handler_prints_backend_steps() -> None:
    """Backend INFO logs appear as indented sub-steps while the handler is installed."""
    out = io.StringIO()
    handler = install_log_handler(out)
    try:
        logging.getLogger("jisu.dashboard.memory.gbrain_http").info("서버 시작")
    finally:
        remove_log_handler(handler)
    logging.getLogger("jisu.dashboard.memory.gbrain_http").info("not printed")
    assert out.getvalue() == "   · 서버 시작\n"
