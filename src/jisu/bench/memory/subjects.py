"""What the benchmark measures: one memory store + one search path per subject.

| subject               | store                                  | search                                               |
|-----------------------|----------------------------------------|------------------------------------------------------|
| ``json``              | upstream ``memory.v1.json``            | local keyword search (``JsonMemoryBackend.search``)  |
| ``gbrain``            | GBrain CLI, keyless, categorized       | the app's path: GBrain, then local keywords if empty |
| ``gbrain-native``     | GBrain CLI, keyless, categorized       | GBrain's own ranking only (no fallback)              |
| ``gbrain-embed``      | GBrain CLI with an embedding model     | the app's path                                       |
| ``gbrain-embed-native`` | GBrain CLI with an embedding model   | GBrain's own ranking only (keyword + vector)         |
| ``gbrain-http``       | GBrain keyless, resident HTTP server   | the app's path                                       |

``gbrain`` vs ``gbrain-embed*`` isolates the embedding model; ``gbrain`` vs
``gbrain-http`` isolates the transport (one Bun process per call vs one server).

Every subject gets its own store under the run's work directory, so a run never
touches the user's real memory. To benchmark a new backend, add a :class:`Subject`
implementation and register it in :data:`SUBJECT_BUILDERS`.
"""

from __future__ import annotations
import time
import logging
from typing import Protocol
from pathlib import Path
from dataclasses import field, dataclass
from collections.abc import Callable

from jisu.dashboard.constants import GBRAIN_EMBEDDING_KEY_ENV, GBRAIN_EMBEDDING_KEY_FALLBACK_ENV
from jisu.bench.memory.dataset import BenchFact, match_key
from jisu.dashboard.memory.base import BackendUnavailableError
from jisu.bench.memory.constants import BENCH_PROVENANCE
from jisu.dashboard.memory.gbrain_http import GBrainHttpBackend
from jisu.dashboard.memory.json_backend import JsonMemoryBackend
from jisu.dashboard.memory.gbrain_backend import (
    EmbeddingConfig,
    GBrainCallError,
    GBrainMemoryBackend,
    parse_json_object,
    find_embedding_key,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchOutcome:
    """Texts a search returned (rank order), their GBrain relevance, and an optional note for the report."""

    texts: list[str]
    note: str = ""
    # Aligned with ``texts``; None for results GBrain did not score (keyword fallback, JSON).
    scores: list[float | None] = field(default_factory=list)


class Subject(Protocol):
    """One benchmarked store + search path."""

    name: str
    categorized: bool
    # Time spent creating the store (e.g. ``gbrain init``), in ms.
    setup_ms: float

    def add(self, fact: BenchFact) -> bool:
        """Store one fact; False if nothing was saved."""
        ...

    def search(self, query: str, k: int) -> SearchOutcome:
        """Return up to ``k`` results for ``query``."""
        ...

    def format_prompt(self) -> str:
        """Build the session prompt fragment (measured for latency)."""
        ...

    def stored_categories(self) -> dict[str, str]:
        """Map stored text (see :func:`match_key`) to category key; {} if the store has no categories."""
        ...

    def close(self) -> None:
        """Release what the store holds (e.g. stop a server); called once after measuring."""
        ...


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


class JsonSubject:
    """The upstream JSON store with its keyword search."""

    categorized = False

    def __init__(self, workdir: Path, name: str = "json") -> None:
        """Create an empty store under ``workdir``."""
        started = time.perf_counter()
        self.name = name
        self._backend = JsonMemoryBackend(workdir)
        self.setup_ms = _elapsed_ms(started)

    def add(self, fact: BenchFact) -> bool:
        """Store the fact text (JSON has no categories)."""
        return self._backend.add_fact(fact.text) is not None

    def search(self, query: str, k: int) -> SearchOutcome:
        """Local keyword search."""
        return SearchOutcome([fact.text for fact in self._backend.search(query, k)])

    def format_prompt(self) -> str:
        """Upstream prompt fragment."""
        return self._backend.format_for_prompt()

    def stored_categories(self) -> dict[str, str]:
        """JSON stores no categories."""
        return {}

    def close(self) -> None:
        """Nothing to release."""


GBrainFactory = Callable[[Path], GBrainMemoryBackend]


def _initialized(brain: GBrainMemoryBackend) -> GBrainMemoryBackend:
    """Initialize ``brain`` (raises BackendUnavailableError)."""
    if not brain.binary_available():
        raise BackendUnavailableError(brain.readiness_problem() or "gbrain not found")
    try:
        brain.initialize()
    except GBrainCallError as exc:
        raise BackendUnavailableError(_short_gbrain_error(str(exc))) from exc
    return brain


def create_gbrain(home: Path) -> GBrainMemoryBackend:
    """Create and initialize a fresh, isolated, keyless brain at ``home`` (raises BackendUnavailableError)."""
    return _initialized(GBrainMemoryBackend(home=home, require_ready=False))


@dataclass(frozen=True)
class _EmbeddingOptions:
    embedding: EmbeddingConfig
    key: str
    min_relevance: float | None


def _embedding_options(
    embedding: EmbeddingConfig, dotenv_path: Path | None, min_relevance: float | None
) -> _EmbeddingOptions:
    """Collect the settings for an embedding brain (raises BackendUnavailableError without a key)."""
    found = find_embedding_key(dotenv_path)
    if found is None:
        raise BackendUnavailableError(
            f"no embedding key: set {GBRAIN_EMBEDDING_KEY_FALLBACK_ENV} or {GBRAIN_EMBEDDING_KEY_ENV}"
        )
    key, source = found
    logger.info("임베딩 키: %s 사용, 모델 %s · %d차원", source, embedding.model, embedding.dimensions)
    if min_relevance is not None:
        logger.info("relevance 하한 %.3f 적용 (이보다 낮은 결과는 버림)", min_relevance)
    return _EmbeddingOptions(embedding, key, min_relevance)


def embedded_gbrain_factory(
    embedding: EmbeddingConfig, dotenv_path: Path | None = None, min_relevance: float | None = None
) -> GBrainFactory:
    """Build a factory for CLI brains initialized with ``embedding`` (skips via BackendUnavailableError without a key)."""

    def create(home: Path) -> GBrainMemoryBackend:
        options = _embedding_options(embedding, dotenv_path, min_relevance)
        return _initialized(
            GBrainMemoryBackend(
                home=home,
                require_ready=False,
                embedding=options.embedding,
                extra_env={GBRAIN_EMBEDDING_KEY_ENV: options.key},
                min_relevance=options.min_relevance,
            )
        )

    return create


def _started(brain: GBrainHttpBackend) -> GBrainHttpBackend:
    """Initialize ``brain`` and start its HTTP server (raises BackendUnavailableError)."""
    _initialized(brain)
    try:
        brain.start_server()
    except GBrainCallError as exc:
        raise BackendUnavailableError(str(exc)[:300]) from exc
    return brain


def create_http_gbrain(home: Path) -> GBrainMemoryBackend:
    """Create a keyless brain and start its resident HTTP server (raises BackendUnavailableError)."""
    return _started(GBrainHttpBackend(home=home, require_ready=False))


def embedded_http_gbrain_factory(
    embedding: EmbeddingConfig, dotenv_path: Path | None = None, min_relevance: float | None = None
) -> GBrainFactory:
    """Build a factory for embedding brains served over HTTP (the server gets the key; no key -> skipped)."""

    def create(home: Path) -> GBrainMemoryBackend:
        options = _embedding_options(embedding, dotenv_path, min_relevance)
        return _started(
            GBrainHttpBackend(
                home=home,
                require_ready=False,
                embedding=options.embedding,
                extra_env={GBRAIN_EMBEDDING_KEY_ENV: options.key},
                min_relevance=options.min_relevance,
            )
        )

    return create


@dataclass(frozen=True)
class GBrainFactories:
    """How each GBrain subject creates its brain (swapped for fakes in tests)."""

    keyless: GBrainFactory = create_gbrain
    embedded: GBrainFactory | None = None
    http: GBrainFactory = create_http_gbrain
    embedded_http: GBrainFactory | None = None


def _short_gbrain_error(text: str) -> str:
    """One line from a GBrain failure: its JSON envelope's message, else the first line."""
    payload = parse_json_object(text)
    if payload and payload.get("message"):
        code = payload.get("code") or payload.get("error") or "error"
        return f"gbrain init failed ({code}): {payload['message']}"
    return text.strip().splitlines()[0][:300] if text.strip() else "gbrain init failed"


class GBrainSubject:
    """A dedicated GBrain brain; ``native=True`` disables the app's local keyword fallback."""

    categorized = True

    def __init__(
        self,
        workdir: Path,
        *,
        native: bool,
        factory: GBrainFactory = create_gbrain,
        name: str | None = None,
    ) -> None:
        """Create the brain under ``workdir`` (raises BackendUnavailableError)."""
        self.name = name or ("gbrain-native" if native else "gbrain")
        self._native = native
        started = time.perf_counter()
        self.backend = factory(workdir / "gbrain")
        self.setup_ms = _elapsed_ms(started)

    def add(self, fact: BenchFact) -> bool:
        """Store the fact under its category (GBrain entity + kind)."""
        return self.backend.add_categorized(fact.text, fact.category, provenance=BENCH_PROVENANCE) is not None

    def search(self, query: str, k: int) -> SearchOutcome:
        """GBrain ranking with relevance; the app path falls back to local keywords when GBrain finds nothing."""
        try:
            found = self.backend.search_native(query, k) if self._native else self.backend.search_scored(query, k)
        except GBrainCallError as exc:
            return SearchOutcome([], note=f"error: {exc}"[:200])
        notes = []
        if found.degraded_reason:
            notes.append(f"degraded: {found.degraded_reason}")
        if found.dropped:
            notes.append(f"하한 미만 {found.dropped}개 버림")
        if not self._native and found.facts and not found.scores:
            notes.append("키워드 대체 검색")
        return SearchOutcome(
            [item.fact.text for item in found.facts],
            note=" · ".join(notes),
            scores=[found.scores.get(item.fact.id) for item in found.facts],
        )

    def format_prompt(self) -> str:
        """Sectioned persona/user prompt."""
        return self.backend.format_for_prompt()

    def stored_categories(self) -> dict[str, str]:
        """Categories as GBrain reports them back."""
        return {match_key(item.fact.text): item.category.key for item in self.backend.list_categorized()}

    def close(self) -> None:
        """Stop the brain's server, if it has one."""
        self.backend.close()


def _embedded(factories: GBrainFactories) -> GBrainFactory:
    if factories.embedded is None:
        raise BackendUnavailableError("no embedding configuration given")
    return factories.embedded


def _embedded_http(factories: GBrainFactories) -> GBrainFactory:
    if factories.embedded_http is None:
        raise BackendUnavailableError("no embedding configuration given")
    return factories.embedded_http


SubjectBuilder = Callable[[Path, GBrainFactories], Subject]

# One line per subject for progress output: what is stored where, and how it is searched.
SUBJECT_DESCRIPTIONS: dict[str, str] = {
    "json": "JSON 파일 저장 · 로컬 키워드 검색 (기준선)",
    "gbrain": "GBrain CLI (호출마다 프로세스 기동) · 임베딩 없음 · GBrain 순위 → 비면 키워드 대체 검색",
    "gbrain-native": "GBrain CLI · 임베딩 없음 · GBrain 순위만 (대체 검색 없음)",
    "gbrain-embed": "GBrain CLI · 임베딩 켬 · GBrain 순위 → 비면 키워드 대체 검색",
    "gbrain-embed-native": "GBrain CLI · 임베딩 켬 · GBrain 순위만 (키워드 + 벡터)",
    "gbrain-http": "GBrain 상주 HTTP 서버 · 임베딩 없음 · GBrain 순위 → 비면 키워드 대체 검색",
    "gbrain-embed-http": "GBrain 상주 HTTP 서버 · 임베딩 켬 · GBrain 순위 → 비면 키워드 대체 검색 (실제 앱 후보 조합)",
}

SUBJECT_BUILDERS: dict[str, SubjectBuilder] = {
    "json": lambda workdir, _f: JsonSubject(workdir),
    "gbrain": lambda workdir, f: GBrainSubject(workdir, native=False, factory=f.keyless),
    "gbrain-native": lambda workdir, f: GBrainSubject(workdir, native=True, factory=f.keyless),
    "gbrain-embed": lambda workdir, f: GBrainSubject(workdir, native=False, factory=_embedded(f), name="gbrain-embed"),
    "gbrain-embed-native": lambda workdir, f: GBrainSubject(
        workdir, native=True, factory=_embedded(f), name="gbrain-embed-native"
    ),
    "gbrain-http": lambda workdir, f: GBrainSubject(workdir, native=False, factory=f.http, name="gbrain-http"),
    "gbrain-embed-http": lambda workdir, f: GBrainSubject(
        workdir, native=False, factory=_embedded_http(f), name="gbrain-embed-http"
    ),
}


def build_subject(name: str, workdir: Path, factories: GBrainFactories | None = None) -> Subject:
    """Create subject ``name`` in its own ``workdir`` (KeyError if unknown, BackendUnavailableError if unusable)."""
    try:
        builder = SUBJECT_BUILDERS[name]
    except KeyError:
        known = ", ".join(SUBJECT_BUILDERS)
        raise KeyError(f"unknown subject {name!r} (known: {known})") from None
    workdir.mkdir(parents=True, exist_ok=True)
    return builder(workdir, factories or GBrainFactories())
