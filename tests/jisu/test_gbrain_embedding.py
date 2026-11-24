"""Tests for GBrain embedding setup, key lookup and working-directory isolation."""

from __future__ import annotations
import sys
from pathlib import Path

import pytest
from gbrain_fakes import FakeGBrain

from jisu.dashboard.memory.gbrain_backend import (
    EmbeddingConfig,
    GBrainMemoryBackend,
    run_command,
    resolve_embedding_key,
)


def test_initialize_with_embedding(tmp_path: Path) -> None:
    """An embedding config replaces --no-embedding; extra env reaches every call."""
    fake = FakeGBrain()
    binary = tmp_path / "fake-gbrain"
    binary.write_text("", encoding="utf-8")
    brain = GBrainMemoryBackend(
        runner=fake,
        binary=str(binary),
        home=tmp_path / "gbrain",
        require_ready=False,
        embedding=EmbeddingConfig("google:gemini-embedding-2", 1024),
        extra_env={"GOOGLE_GENERATIVE_AI_API_KEY": "k"},
    )
    brain.initialize()
    argv, env = fake.calls[0]
    assert "--no-embedding" not in argv
    assert argv[argv.index("--embedding-model") + 1] == "google:gemini-embedding-2"
    assert argv[argv.index("--embedding-dimensions") + 1] == "1024"
    assert env["GOOGLE_GENERATIVE_AI_API_KEY"] == "k"
    assert env["GBRAIN_HOME"] == str(tmp_path / "gbrain")


def test_keyless_initialize_unchanged(tmp_path: Path) -> None:
    """Without an embedding config the brain stays keyless."""
    fake = FakeGBrain()
    binary = tmp_path / "fake-gbrain"
    binary.write_text("", encoding="utf-8")
    GBrainMemoryBackend(runner=fake, binary=str(binary), home=tmp_path / "g", require_ready=False).initialize()
    assert "--no-embedding" in fake.calls[0][0]


def test_resolve_embedding_key_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """GOOGLE_GENERATIVE_AI_API_KEY wins, then GEMINI_API_KEY, then the .env file."""
    monkeypatch.delenv("GOOGLE_GENERATIVE_AI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text("OPENAI_API_KEY=other\nGEMINI_API_KEY=from-file\n", encoding="utf-8")

    assert resolve_embedding_key(None) is None
    assert resolve_embedding_key(tmp_path / "missing.env") is None
    assert resolve_embedding_key(dotenv) == "from-file"
    monkeypatch.setenv("GEMINI_API_KEY", "gemini")
    assert resolve_embedding_key(dotenv) == "gemini"
    monkeypatch.setenv("GOOGLE_GENERATIVE_AI_API_KEY", "google")
    assert resolve_embedding_key(dotenv) == "google"


def test_run_command_runs_in_brain_home(tmp_path: Path) -> None:
    """CLI calls run inside GBRAIN_HOME so GBrain never reads the caller's .env."""
    home = tmp_path / "home"
    home.mkdir()
    script = "import os; print(os.getcwd())"
    result = run_command([sys.executable, "-c", script], {"GBRAIN_HOME": str(home)}, 30.0)
    assert Path(result.stdout.strip()).resolve() == home.resolve()
