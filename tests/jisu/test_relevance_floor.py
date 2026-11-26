"""Tests for the GBrain relevance floor (backend) and the --sweep re-scoring (benchmark)."""

from __future__ import annotations
import json
from typing import Any
from pathlib import Path
from collections.abc import Mapping, Sequence

import pytest

from jisu.bench.memory.sweep import rescore, has_scores, parse_sweep, render_sweep, sweep_subject
from jisu.bench.memory.report import BenchReport, load_report, save_report
from jisu.bench.memory.runner import BenchConfig, run_memory_bench
from jisu.bench.memory.metrics import QueryResult
from jisu.bench.memory.progress import describe_query
from jisu.dashboard.memory.gbrain_backend import CommandResult, GBrainMemoryBackend, default_min_relevance


FACTS = [
    {"fact_id": "1", "fact": "사용자는 백엔드 개발자로 일한다", "kind": "fact", "entity_slug": "user"},
    {"fact_id": "2", "fact": "사용자의 강아지 이름은 모찌다", "kind": "fact", "entity_slug": "user"},
]


class ScoredGBrain:
    """Fake ``gbrain`` whose question recall returns facts with relevance scores."""

    def __init__(self, ranked: Mapping[str, list[tuple[str, float]]]) -> None:
        """``ranked``: question -> [(fact id, relevance)]; unknown questions return nothing."""
        self.ranked = ranked

    def __call__(self, args: Sequence[str], env: Mapping[str, str], timeout_s: float) -> CommandResult:
        """Answer init / recall like the real CLI."""
        argv = list(args)
        if argv[1] == "init":
            return CommandResult(0, '{"status":"ok"}', "")
        params: dict[str, Any] = json.loads(argv[3])
        if argv[2] != "recall":
            return CommandResult(0, json.dumps({"id": "9", "status": "inserted"}), "")
        if "question" not in params:
            return CommandResult(0, json.dumps({"facts": FACTS}), "")
        by_id = {fact["fact_id"]: fact for fact in FACTS}
        facts = [{**by_id[fid], "relevance": score} for fid, score in self.ranked.get(params["question"], [])]
        return CommandResult(0, json.dumps({"facts": facts}), "")


def _backend(tmp_path: Path, fake: ScoredGBrain, floor: float | None) -> GBrainMemoryBackend:
    binary = tmp_path / "fake-gbrain"
    binary.write_text("", encoding="utf-8")
    return GBrainMemoryBackend(
        runner=fake, binary=str(binary), home=tmp_path / "g", require_ready=False, min_relevance=floor
    )


def test_floor_drops_low_relevance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Facts under the floor are dropped and counted; scores come back by fact id."""
    monkeypatch.delenv("JISU_GBRAIN_MIN_RELEVANCE", raising=False)
    fake = ScoredGBrain({"직업": [("1", 0.62), ("2", 0.51)]})
    found = _backend(tmp_path, fake, 0.54).search_native("직업", 5)
    assert [f.fact.id for f in found.facts] == ["1"]
    assert found.scores == {"1": 0.62} and found.dropped == 1

    unfiltered = _backend(tmp_path, fake, None).search_native("직업", 5)
    assert [f.fact.id for f in unfiltered.facts] == ["1", "2"] and unfiltered.dropped == 0


def test_app_path_falls_back_to_keywords_after_floor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the floor empties GBrain's answer, the app path runs the keyword search (unscored)."""
    monkeypatch.delenv("JISU_GBRAIN_MIN_RELEVANCE", raising=False)
    fake = ScoredGBrain({"강아지 이름": [("2", 0.40)]})
    backend = _backend(tmp_path, fake, 0.54)
    found = backend.search_scored("강아지 이름", 5)
    assert [f.fact.id for f in found.facts] == ["2"]
    assert found.scores == {} and found.dropped == 1
    assert [f.text for f in backend.search("강아지 이름", 5)] == ["사용자의 강아지 이름은 모찌다"]


def test_floor_from_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """JISU_GBRAIN_MIN_RELEVANCE sets the default floor; junk means no floor."""
    monkeypatch.setenv("JISU_GBRAIN_MIN_RELEVANCE", "0.6")
    assert default_min_relevance() == 0.6
    assert _backend(tmp_path, ScoredGBrain({}), None).min_relevance == 0.6
    assert _backend(tmp_path, ScoredGBrain({}), 0.5).min_relevance == 0.5
    monkeypatch.setenv("JISU_GBRAIN_MIN_RELEVANCE", "high")
    assert default_min_relevance() is None
    monkeypatch.setenv("JISU_GBRAIN_MIN_RELEVANCE", "")
    assert default_min_relevance() is None


def test_parse_sweep() -> None:
    """Ranges include the stop value; lists are sorted; bad input raises."""
    assert parse_sweep("0.5:0.6:0.05") == [0.5, 0.55, 0.6]
    assert parse_sweep("0.6, 0.55") == [0.55, 0.6]
    for bad in ("0.6:0.5:0.1", "0.5:0.6:0", "1:2", "", "0:1000:0.001"):
        with pytest.raises(ValueError):
            parse_sweep(bad)


def _query(
    qid: str, expected: tuple[str, ...], returned: tuple[str, ...], scores: tuple[float | None, ...]
) -> QueryResult:
    rank = next((i + 1 for i, fid in enumerate(returned) if fid in expected), None)
    return QueryResult(
        query_id=qid,
        text=qid,
        tags=("paraphrase",) if expected else ("negative",),
        expected=expected,
        returned=returned,
        rank=rank,
        hit_at_1=rank == 1,
        hit_at_k=rank is not None,
        recall_at_k=1.0 if rank else 0.0,
        false_positive=not expected and bool(returned),
        latency_ms=1.0,
        scores=scores,
    )


def test_rescore_and_sweep_subject() -> None:
    """A floor between the hit and the false positive keeps the hit and clears the negative."""
    hit = _query("job", ("job",), ("job", "dog"), (0.62, 0.51))
    negative = _query("cat", (), ("dog",), (0.53,))
    assert rescore(hit, 0.55, 5).returned == ("job",)
    assert rescore(negative, 0.55, 5).false_positive is False
    unscored = _query("kw", ("job",), ("job",), (None,))
    assert rescore(unscored, 0.9, 5).hit_at_k  # keyword results have no score and are kept

    from jisu.bench.memory.report import SubjectResult
    from jisu.bench.memory.metrics import aggregate, scores_by_tag

    queries = [hit, negative]
    result = SubjectResult(
        name="gbrain-embed-native",
        categorized=True,
        setup_ms=0.0,
        facts_added=2,
        facts_failed=0,
        category_match=1.0,
        scores=aggregate(queries),
        tags=scores_by_tag(queries),
        latency={},
        queries=queries,
    )
    assert has_scores(result)
    rows = sweep_subject(result, 5, [0.5, 0.55, 0.7])
    assert [row.threshold for row in rows] == [None, 0.5, 0.55, 0.7]
    assert [(row.scores.hit_at_k, row.scores.false_positive_rate) for row in rows] == [
        (1.0, 1.0),
        (1.0, 1.0),
        (1.0, 0.0),
        (0.0, 0.0),
    ]
    report = BenchReport("2026-11-27T00:00:00+09:00", 1.0, {"name": "x"}, {"k": 5}, {}, subjects=[result])
    text = render_sweep(report, [0.55])
    assert "[gbrain-embed-native]" in text and "0.550" in text and "정답의 relevance" in text


def test_scores_survive_save_and_load(tmp_path: Path) -> None:
    """Relevance scores are kept in the report JSON; older reports without them still load."""
    queries = [_query("job", ("job",), ("job",), (0.62,))]
    from jisu.bench.memory.report import SubjectResult
    from jisu.bench.memory.metrics import aggregate, scores_by_tag

    result = SubjectResult(
        "gbrain-embed", True, 0.0, 1, 0, 1.0, aggregate(queries), scores_by_tag(queries), {}, queries
    )
    report = BenchReport("2026-11-27T00:00:00+09:00", 1.0, {"name": "x"}, {"k": 5}, {}, subjects=[result])
    path = save_report(report, tmp_path)
    assert load_report(path).subjects[0].queries[0].scores == (0.62,)

    raw = json.loads(path.read_text(encoding="utf-8"))
    del raw["subjects"][0]["queries"][0]["scores"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert load_report(path).subjects[0].queries[0].scores == ()


def test_progress_shows_relevance() -> None:
    """Query lines carry the relevance of the hit and of what came back."""
    line = describe_query(_query("job", ("job",), ("job",), (0.62,)), 1, 2)
    assert "1위 (relevance 0.62)" in line
    fp = describe_query(_query("cat", (), ("dog",), (0.53,)), 2, 2)
    assert "dog(0.53)" in fp


def test_min_relevance_reaches_embed_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--min-relevance is recorded in the report config and handed to the embedding brains."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    created: list[GBrainMemoryBackend] = []

    def fake_create_init(self: GBrainMemoryBackend) -> str:
        created.append(self)
        raise RuntimeError("stop after construction")

    monkeypatch.setattr(GBrainMemoryBackend, "binary_available", lambda self: True)
    monkeypatch.setattr(GBrainMemoryBackend, "initialize", fake_create_init)
    with pytest.raises(RuntimeError):
        run_memory_bench(BenchConfig(subjects=("gbrain-embed",), repeat=1, min_relevance=0.54), workdir=tmp_path / "w")
    assert created and created[0].min_relevance == 0.54
