"""Readable progress log for the benchmark CLI.

Turns :class:`~jisu.bench.memory.runner.BenchEvent`s into lines like::

    ━━ [3/3] gbrain-http
       GBrain 상주 HTTP 서버 · 임베딩 없음 · GBrain 순위 → 비면 키워드 대체 검색
       · 토큰 발급: gbrain auth create jisu-memory-1764...
       · 서버 시작: gbrain serve --http (http://127.0.0.1:52011, ...)
       준비 완료 3.0s
       저장 30/30 · 평균 16ms
       검색 41개 (질문마다 상위 5개 확인)
       [O]  1/41 dog_exact '모찌' → 1위 · 9.8ms
       [X]  8/41 job_para '직업' (paraphrase) → 못 찾음 (정답 job) · 결과 없음 · 10ms
       결과: hit@5 80.6% · 오탐 20.0% · 검색 p50 10ms · 4.9s

Marks are ASCII (``[O]``/``[X]``) so Windows consoles (cp949) print them.
Setup steps inside the backends arrive as ``logging`` records; :func:`install_log_handler`
prints them indented under the current subject.
"""

from __future__ import annotations
import sys
import time
import logging
from typing import TextIO

from jisu.bench.memory.table import ms, pct
from jisu.bench.memory.runner import BenchEvent
from jisu.bench.memory.metrics import QueryResult
from jisu.bench.memory.subjects import SUBJECT_DESCRIPTIONS


INDENT = "   "
# Print a save-progress line every this many facts (and on the last one).
FACT_PROGRESS_EVERY = 10
# Returned fact ids shown for a failed query.
MAX_RETURNED_SHOWN = 3
LOGGER_NAMES = ("jisu.dashboard.memory", "jisu.bench.memory")


def install_log_handler(stream: TextIO | None = None) -> logging.Handler:
    """Print INFO logs from the memory backends as indented sub-steps; returns the handler."""
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(logging.Formatter(f"{INDENT}· %(message)s"))
    for name in LOGGER_NAMES:
        logger = logging.getLogger(name)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return handler


def remove_log_handler(handler: logging.Handler) -> None:
    """Undo :func:`install_log_handler`."""
    for name in LOGGER_NAMES:
        logger = logging.getLogger(name)
        logger.removeHandler(handler)
        logger.propagate = True


def describe_query(result: QueryResult, done: int, total: int) -> str:
    """One line per query: pass mark, the question, what came back, and how long it took."""
    mark = "[O]" if result.passed else "[X]"
    width = len(str(total))
    tags = [tag for tag in result.tags if tag != "negative"]
    head = f"{mark} {done:>{width}}/{total} {result.query_id} '{result.text}'"
    if tags:
        head += f" ({', '.join(tags)})"

    shown = ", ".join(result.returned[:MAX_RETURNED_SHOWN])
    if len(result.returned) > MAX_RETURNED_SHOWN:
        shown += f" 외 {len(result.returned) - MAX_RETURNED_SHOWN}개"
    if result.negative:
        outcome = f"없는 기억인데 나옴: {shown}" if result.false_positive else "결과 없음 (없는 기억, 정상)"
    elif result.rank is not None:
        outcome = f"{result.rank}위"
    else:
        expected = ", ".join(result.expected)
        outcome = f"못 찾음 (정답 {expected}) · " + (f"나온 것: {shown}" if shown else "결과 없음")
    line = f"{head} → {outcome} · {ms(result.latency_ms)}ms"
    if result.note:
        line += f" [{result.note}]"
    return line


class ProgressPrinter:
    """Callable ``on_event`` handler for :func:`~jisu.bench.memory.runner.run_memory_bench`."""

    def __init__(self, *, k: int, repeat: int, brief: bool = False, stream: TextIO | None = None) -> None:
        """``brief`` prints only failed queries instead of every query."""
        self._k, self._repeat, self._brief = k, repeat, brief
        self._out = stream or sys.stdout
        self._subjects_total = 0
        self._subject_index = 0
        self._subject_started = 0.0
        self._fact_ms: list[float] = []

    def _print(self, text: str = "") -> None:
        print(text, file=self._out, flush=True)

    def __call__(self, event: BenchEvent) -> None:
        """Print the lines for one event."""
        handler = getattr(self, f"_on_{event.kind}", None)
        if handler is not None:
            handler(event)

    def _on_run_start(self, event: BenchEvent) -> None:
        self._subjects_total = event.total
        self._print(f"기억 검색 벤치마크 · {event.message}")
        self._print(f"질문마다 상위 {self._k}개 안에 정답이 있으면 맞힘 · 속도는 질문당 {self._repeat}번 측정")

    def _on_subject_start(self, event: BenchEvent) -> None:
        self._subject_index += 1
        self._subject_started = time.perf_counter()
        self._fact_ms = []
        self._print()
        self._print(f"━━ [{self._subject_index}/{self._subjects_total}] {event.subject}")
        description = SUBJECT_DESCRIPTIONS.get(event.subject)
        if description:
            self._print(f"{INDENT}{description}")
        self._print(f"{INDENT}준비 중 (새 저장소 생성)")

    def _on_subject_ready(self, event: BenchEvent) -> None:
        self._print(f"{INDENT}준비 완료 {event.elapsed_ms / 1000:.1f}s")

    def _on_subject_skipped(self, event: BenchEvent) -> None:
        self._print(f"{INDENT}건너뜀: {event.message}")

    def _on_fact_added(self, event: BenchEvent) -> None:
        self._fact_ms.append(event.elapsed_ms)
        if not event.ok:
            self._print(f"{INDENT}저장 실패: {event.message}")
        if event.done % FACT_PROGRESS_EVERY == 0 or event.done == event.total:
            average = sum(self._fact_ms) / len(self._fact_ms)
            self._print(f"{INDENT}저장 {event.done}/{event.total} · 평균 {ms(average)}ms")

    def _on_query_done(self, event: BenchEvent) -> None:
        if event.done == 1:
            note = " · 틀린 질문만 표시" if self._brief else ""
            self._print(f"{INDENT}검색 {event.total}개 (질문마다 상위 {self._k}개 확인{note})")
        if event.query is None or (self._brief and event.query.passed):
            return
        self._print(INDENT + describe_query(event.query, event.done, event.total))

    def _on_prompt_done(self, event: BenchEvent) -> None:
        self._print(f"{INDENT}세션 프롬프트 생성 {ms(event.elapsed_ms)}ms")

    def _on_subject_done(self, event: BenchEvent) -> None:
        result = event.result
        elapsed = time.perf_counter() - self._subject_started
        if result is None:
            self._print(f"{INDENT}완료 · {elapsed:.1f}s")
            return
        parts = [f"hit@{self._k} {pct(result.scores.hit_at_k)}"]
        paraphrase = result.tags.get("paraphrase")
        if paraphrase is not None:
            parts.append(f"유의어 {pct(paraphrase.hit_at_k)}")
        parts.append(f"오탐 {pct(result.scores.false_positive_rate)}")
        parts.append(f"검색 p50 {ms(result.latency['search'].p50_ms)}ms")
        parts.append(f"{elapsed:.1f}s")
        failed = len(result.failures())
        self._print(f"{INDENT}결과: {' · '.join(parts)} (틀린 질문 {failed}개)")
