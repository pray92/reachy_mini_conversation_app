"""Try relevance floors on results already measured, without calling GBrain again.

Every GBrain result in a report carries its ``relevance`` (cosine similarity +
term share). A floor ``t`` keeps only results with relevance >= ``t``; this
module re-scores each query at several floors so one benchmark run answers
"which floor keeps paraphrases but drops answers to questions nothing
answers?".

Exact for ``-native`` subjects. For an app-path subject it ignores the keyword
fallback that would run when the floor empties a result (results the fallback
already returned have no score and are always kept); run that subject with
``--min-relevance`` to measure the fallback too.
"""

from __future__ import annotations
import statistics
from dataclasses import dataclass
from collections.abc import Sequence

from jisu.bench.memory.table import pct, render_table
from jisu.bench.memory.report import BenchReport, SubjectResult
from jisu.bench.memory.dataset import BenchQuery
from jisu.bench.memory.metrics import Scores, QueryResult, aggregate, score_query, scores_by_tag
from jisu.bench.memory.constants import SWEEP_MAX_STEPS


@dataclass(frozen=True)
class SweepRow:
    """Scores of one subject at one relevance floor (None = no floor)."""

    threshold: float | None
    scores: Scores
    paraphrase: Scores | None


def parse_sweep(text: str) -> list[float]:
    """``"0.5:0.8:0.02"`` (start:stop:step, stop included) or ``"0.55,0.6"`` -> floors (raises ValueError)."""
    text = text.strip()
    if ":" in text:
        parts = text.split(":")
        if len(parts) != 3:
            raise ValueError(f"--sweep needs START:STOP:STEP, got {text!r}")
        start, stop, step = (float(part) for part in parts)
        if step <= 0 or stop < start:
            raise ValueError(f"--sweep needs STEP > 0 and STOP >= START, got {text!r}")
        count = int(round((stop - start) / step)) + 1
        if count > SWEEP_MAX_STEPS:
            raise ValueError(f"--sweep {text!r} gives {count} floors; at most {SWEEP_MAX_STEPS}")
        return [round(start + i * step, 6) for i in range(count)]
    floors = sorted({float(part) for part in text.split(",") if part.strip()})
    if not floors:
        raise ValueError("--sweep needs at least one floor")
    return floors


def has_scores(result: SubjectResult) -> bool:
    """Whether any query of this subject carries a GBrain relevance."""
    return any(score is not None for query in result.queries for score in query.scores)


def rescore(query: QueryResult, threshold: float, k: int) -> QueryResult:
    """``query`` as if results below ``threshold`` had been dropped (unscored results are kept)."""
    scores = query.scores or (None,) * len(query.returned)
    kept = [(fact_id, score) for fact_id, score in zip(query.returned, scores) if score is None or score >= threshold]
    bench_query = BenchQuery(query.query_id, query.text, query.expected, query.tags)
    return score_query(
        bench_query,
        [fact_id for fact_id, _ in kept],
        k,
        latency_ms=query.latency_ms,
        note=query.note,
        scores=[score for _, score in kept],
    )


def sweep_subject(result: SubjectResult, k: int, thresholds: Sequence[float]) -> list[SweepRow]:
    """No-floor row, then one row per floor."""
    rows = [SweepRow(None, result.scores, result.tags.get("paraphrase"))]
    for threshold in thresholds:
        rescored = [rescore(query, threshold, k) for query in result.queries]
        rows.append(SweepRow(threshold, aggregate(rescored), scores_by_tag(rescored).get("paraphrase")))
    return rows


def _first_hit_scores(result: SubjectResult) -> list[float]:
    """Relevance of the first correct result of each answered positive query."""
    found = []
    for query in result.queries:
        if query.negative or query.rank is None or len(query.scores) < query.rank:
            continue
        score = query.scores[query.rank - 1]
        if score is not None:
            found.append(score)
    return found


def _negative_top_scores(result: SubjectResult) -> list[float]:
    """Highest relevance returned for each negative query that returned something."""
    return [
        max(score for score in query.scores if score is not None)
        for query in result.queries
        if query.negative and any(score is not None for score in query.scores)
    ]


def _spread(values: list[float]) -> str:
    if not values:
        return "-"
    return f"{min(values):.3f} ~ {max(values):.3f} (중앙 {statistics.median(values):.3f}, {len(values)}개)"


def render_sweep(report: BenchReport, thresholds: Sequence[float]) -> str:
    """One table per GBrain subject with scores: hit@k, paraphrase and false positives at each floor."""
    k = int(report.config.get("k", 5))
    applied = report.config.get("min_relevance")
    blocks = []
    for result in report.subjects:
        if not has_scores(result):
            continue
        lines = [f"[{result.name}] relevance 하한별 결과 (GBrain 재호출 없이 다시 채점)"]
        lines.append(f"  정답의 relevance       : {_spread(_first_hit_scores(result))}")
        lines.append(f"  없는 기억에 나온 최고값 : {_spread(_negative_top_scores(result))}")
        if applied is not None and result.name.startswith("gbrain-embed"):
            lines.append(f"  (이 실행은 이미 하한 {applied}를 걸었다. 그보다 낮은 하한은 의미가 없다)")
        if not result.name.endswith("-native"):
            lines.append(
                "  (앱 경로: 하한으로 결과가 비면 실제로는 키워드 대체 검색이 돈다. 정확히 보려면 --min-relevance로 실행)"
            )
        rows = [
            [
                "없음" if row.threshold is None else f"{row.threshold:.3f}",
                pct(row.scores.hit_at_k),
                pct(row.paraphrase.hit_at_k) if row.paraphrase else "-",
                pct(row.scores.false_positive_rate),
            ]
            for row in sweep_subject(result, k, thresholds)
        ]
        lines.append(render_table(["하한", f"hit@{k}", "유의어", "오탐"], rows))
        blocks.append("\n".join(lines))
    if not blocks:
        return "sweep: relevance가 기록된 GBrain 대상이 없다 (이 기능 이전 결과이거나 json만 실행함)"
    return "\n\n".join(blocks)
