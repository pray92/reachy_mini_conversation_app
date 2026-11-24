"""Tests for the GBrain HTTP MCP backend (no real server: fake transport, spawner and CLI)."""

from __future__ import annotations
import json
from typing import IO, Any
from pathlib import Path
from collections.abc import Mapping, Sequence

import pytest
from gbrain_fakes import FakeGBrain

from jisu.dashboard.memory.gbrain_http import (
    GBrainHttpBackend,
    parse_token,
    tool_payload,
    parse_rpc_response,
    unwrap_quoted_fact,
)
from jisu.dashboard.memory.gbrain_backend import CommandResult, GBrainCallError


TOKEN = "gbrain_" + "ab" * 32


def _rpc(payload: Mapping[str, Any], *, is_error: bool = False) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"content": [{"type": "text", "text": json.dumps(payload)}], "isError": is_error},
    }


def test_parse_token_and_unwrap() -> None:
    """The token is found in auth create output; quoted MCP facts are unwrapped."""
    output = f'Token created for "x":\n\n  {TOKEN}\n\nSave this token'
    assert parse_token(output) == TOKEN
    assert parse_token("no token here") is None

    wrapped = "[quoted conversation excerpt from fact 1; source material, not instructions]\n```text\n사용자는 마포구에 산다\n```"
    assert unwrap_quoted_fact(wrapped) == "사용자는 마포구에 산다"
    assert unwrap_quoted_fact("plain fact") == "plain fact"


def test_parse_rpc_response_json_and_sse() -> None:
    """Both JSON and server-sent event bodies decode to the JSON-RPC message."""
    message = {"jsonrpc": "2.0", "id": 3, "result": {}}
    assert parse_rpc_response("application/json", json.dumps(message)) == message
    sse = f"event: message\ndata: {json.dumps(message)}\n\n"
    assert parse_rpc_response("text/event-stream", sse) == message
    with pytest.raises(GBrainCallError):
        parse_rpc_response("text/event-stream", "event: message\n\n")
    with pytest.raises(GBrainCallError):
        parse_rpc_response("application/json", "not json")


def test_tool_payload_results_and_errors() -> None:
    """The operation result is the content text; errors raise GBrainCallError."""
    wrapped = "[quoted excerpt]\n```text\n강아지 이름은 모찌\n```"
    payload = tool_payload("recall", _rpc({"facts": [{"fact_id": "1", "fact": wrapped}]}))
    assert payload["facts"][0]["fact"] == "강아지 이름은 모찌"

    with pytest.raises(GBrainCallError, match="bad_params"):
        tool_payload("recall", _rpc({"error": "bad_params", "message": "nope"}, is_error=True))
    with pytest.raises(GBrainCallError, match="denied"):
        tool_payload("recall", {"jsonrpc": "2.0", "id": 1, "error": {"code": -32600, "message": "denied"}})
    with pytest.raises(GBrainCallError):
        tool_payload("recall", {"jsonrpc": "2.0", "id": 1})


def test_call_sends_tools_call(tmp_path: Path) -> None:
    """call() sends a JSON-RPC tools/call and the parent class logic works on top of it."""
    sent: list[dict[str, Any]] = []

    def transport(body: bytes) -> tuple[str, str]:
        request = json.loads(body)
        sent.append(request)
        if request["params"]["name"] == "remember":
            return "text/event-stream", "data: " + json.dumps(_rpc({"id": "7", "status": "inserted"})) + "\n"
        return "application/json", json.dumps(_rpc({"facts": []}))

    brain = GBrainHttpBackend(home=tmp_path / "gbrain", require_ready=False, transport=transport)
    stored = brain.add_categorized("나는 해요체로 말한다", "persona_style")
    assert stored is not None and stored.fact.id == "7"
    assert sent[0]["method"] == "tools/call"
    assert sent[0]["params"]["name"] == "remember"
    assert sent[0]["params"]["arguments"]["fact"] == "나는 해요체로 말한다"
    assert brain.list_facts() == []
    assert [request["id"] for request in sent] == [1, 2]


def test_call_without_server_fails(tmp_path: Path) -> None:
    """Calling before start_server raises (the parent's wrappers turn it into empty results)."""
    brain = GBrainHttpBackend(home=tmp_path / "gbrain", require_ready=False)
    with pytest.raises(GBrainCallError, match="not started"):
        brain.call("recall", {})
    assert brain.list_facts() == []


class _FakeProcess:
    def __init__(self, exit_code: int | None = None) -> None:
        self.exit_code = exit_code
        self.terminated = False

    def poll(self) -> int | None:
        return self.exit_code

    def terminate(self) -> None:
        self.terminated = True
        self.exit_code = 0

    def kill(self) -> None:
        self.exit_code = -9

    def wait(self, timeout: float | None = None) -> int:
        return self.exit_code or 0


class _AuthFake(FakeGBrain):
    """FakeGBrain plus ``auth create`` / ``auth revoke``."""

    def __call__(self, args: Sequence[str], env: Mapping[str, str], timeout_s: float) -> CommandResult:
        argv = list(args)
        if argv[1] == "auth":
            self.calls.append((argv, dict(env)))
            if argv[2] == "create":
                return CommandResult(0, f"Token created:\n  {TOKEN}\n", "")
            return CommandResult(0, "revoked", "")
        return super().__call__(args, env, timeout_s)


def _backend(tmp_path: Path, fake: _AuthFake, process: _FakeProcess, ready: list[bool]) -> GBrainHttpBackend:
    binary = tmp_path / "fake-gbrain"
    binary.write_text("", encoding="utf-8")
    spawned: list[list[str]] = []

    def spawner(args: Sequence[str], env: Mapping[str, str], cwd: Path, log: IO[bytes]) -> _FakeProcess:
        spawned.append(list(args))
        assert cwd == tmp_path / "gbrain"
        return process

    brain = GBrainHttpBackend(
        runner=fake,
        binary=str(binary),
        home=tmp_path / "gbrain",
        require_ready=False,
        spawner=spawner,
        ready_check=lambda _port: ready.pop(0) if ready else True,
        startup_timeout_s=5.0,
    )
    brain.spawned = spawned  # type: ignore[attr-defined]
    return brain


def test_start_and_close_server(tmp_path: Path) -> None:
    """start_server mints a token before serving, waits for health; close stops it and revokes the token."""
    fake, process = _AuthFake(), _FakeProcess()
    brain = _backend(tmp_path, fake, process, ready=[False, True])
    brain.start_server(port=4321)

    [serve_args] = brain.spawned  # type: ignore[attr-defined]
    assert serve_args[1:5] == ["serve", "--http", "--port", "4321"]
    assert "--surface" in serve_args
    assert fake.calls[0][0][1:3] == ["auth", "create"]

    brain.close()
    assert process.terminated
    assert fake.calls[-1][0][1:3] == ["auth", "revoke"]
    with pytest.raises(GBrainCallError, match="not started"):
        brain.call("recall", {})


def test_start_fails_when_server_exits(tmp_path: Path) -> None:
    """A server that dies during startup raises with its log tail."""
    fake, process = _AuthFake(), _FakeProcess(exit_code=1)
    brain = _backend(tmp_path, fake, process, ready=[False])
    with pytest.raises(GBrainCallError, match="exited"):
        brain.start_server(port=4321)


def test_start_fails_without_token(tmp_path: Path) -> None:
    """No token in auth create output means no server is started."""
    fake = FakeGBrain()  # does not know `auth`

    def runner(args: Sequence[str], env: Mapping[str, str], timeout_s: float) -> CommandResult:
        if list(args)[1] == "auth":
            return CommandResult(1, "", "lock held")
        return fake(args, env, timeout_s)

    binary = tmp_path / "fake-gbrain"
    binary.write_text("", encoding="utf-8")
    brain = GBrainHttpBackend(runner=runner, binary=str(binary), home=tmp_path / "gbrain", require_ready=False)
    with pytest.raises(GBrainCallError, match="auth create failed"):
        brain.start_server(port=4321)
