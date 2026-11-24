"""GBrain through its resident HTTP MCP server instead of one CLI process per call.

Every :class:`~jisu.dashboard.memory.gbrain_backend.GBrainMemoryBackend` call
starts a new Bun process (``gbrain call ...``), which costs 0.4-0.9 s. This
backend keeps one ``gbrain serve --http`` process per brain and sends the same
operations as MCP ``tools/call`` requests over a kept-alive HTTP connection.
Only :meth:`GBrainHttpBackend.call` differs, so categories, the keyword fallback
and forget behave exactly like the CLI backend.

Lifecycle (:meth:`GBrainHttpBackend.start_server`):

1. ``gbrain auth create <name>`` mints a bearer token. It runs before the server
   starts because PGLite allows only one process to hold the database.
2. ``gbrain serve --http --port <free port> --surface verbs`` starts on
   ``127.0.0.1``; its output goes to ``<home>/serve-http.log``.
3. ``GET /health`` is polled until it answers 200.

:meth:`GBrainHttpBackend.close` stops the server and revokes the token.

Wire format: the server is stateless, so each request is a single JSON-RPC
``tools/call`` (no ``initialize``). It answers either JSON or a server-sent
event stream whose ``data:`` line holds the JSON-RPC response. The operation's
result is the JSON text in ``result.content[0].text``. Facts written over MCP
come back wrapped as a quoted excerpt; :func:`unwrap_quoted_fact` restores the
stored text.
"""

from __future__ import annotations
import re
import json
import time
import socket
import logging
import subprocess
import http.client
import urllib.error
import urllib.request
from typing import IO, Any, Protocol
from pathlib import Path
from collections.abc import Mapping, Callable, Sequence

from jisu.dashboard.constants import (
    GBRAIN_HTTP_HOST,
    GBRAIN_HTTP_SURFACE,
    GBRAIN_HTTP_LOG_NAME,
    GBRAIN_HTTP_TOKEN_NAME,
    GBRAIN_HTTP_STOP_TIMEOUT_S,
    GBRAIN_HTTP_POLL_INTERVAL_S,
    GBRAIN_HTTP_STARTUP_TIMEOUT_S,
)
from jisu.dashboard.memory.gbrain_backend import (
    GBrainCallError,
    GBrainMemoryBackend,
    _error_detail,
    parse_json_object,
)


logger = logging.getLogger(__name__)

# `gbrain auth create` prints the token on its own line: gbrain_<hex>.
_TOKEN_RE = re.compile(r"\bgbrain_[0-9a-fA-F]{32,}\b")
# A fact returned over MCP: "[quoted ... ]\n```text\n<fact>\n```".
_QUOTED_FACT_RE = re.compile(r"\A\[quoted [^\]\n]*\]\n```[A-Za-z0-9_-]*\n(?P<body>.*)\n```\Z", re.DOTALL)
_LOG_TAIL_CHARS = 1500


class ServerProcess(Protocol):
    """The parts of :class:`subprocess.Popen` the backend uses."""

    def poll(self) -> int | None:
        """Exit code, or None while running."""
        ...

    def terminate(self) -> None:
        """Ask the process to stop."""
        ...

    def kill(self) -> None:
        """Stop the process now."""
        ...

    def wait(self, timeout: float | None = None) -> int:
        """Wait for the process to exit."""
        ...


# (argv, env, cwd, log file) -> running server
Spawner = Callable[[Sequence[str], Mapping[str, str], Path, IO[bytes]], ServerProcess]
# JSON-RPC request -> (content type, response body)
Transport = Callable[[bytes], tuple[str, str]]


def spawn_process(args: Sequence[str], env: Mapping[str, str], cwd: Path, log: IO[bytes]) -> ServerProcess:
    """Start the server process (default :data:`Spawner`)."""
    return subprocess.Popen(list(args), env=dict(env), cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT)


def free_port(host: str = GBRAIN_HTTP_HOST) -> int:
    """Ask the OS for a port nothing listens on."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        port: int = sock.getsockname()[1]
        return port


def health_ok(port: int, host: str = GBRAIN_HTTP_HOST) -> bool:
    """Return whether ``GET /health`` answers 200."""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=1.0) as response:
            return bool(response.status == 200)
    except (urllib.error.URLError, OSError):
        return False


def parse_token(output: str) -> str | None:
    """Pull the bearer token out of ``gbrain auth create`` output."""
    match = _TOKEN_RE.search(output)
    return match.group(0) if match else None


def unwrap_quoted_fact(text: str) -> str:
    """Return the stored fact text from MCP's quoted-excerpt wrapper (unchanged if not wrapped)."""
    match = _QUOTED_FACT_RE.match(text)
    return match.group("body") if match else text


def parse_rpc_response(content_type: str, body: str) -> dict[str, Any]:
    """Decode a JSON-RPC response sent as JSON or as a server-sent event stream."""
    if "text/event-stream" in content_type or body.lstrip().startswith(("event:", "data:")):
        data_lines = [line[5:].strip() for line in body.splitlines() if line.startswith("data:")]
        for data in reversed(data_lines):
            message = parse_json_object(data)
            if message is not None:
                return message
        raise GBrainCallError(f"gbrain http: no JSON-RPC message in event stream: {body[:300]}")
    message = parse_json_object(body)
    if message is None:
        raise GBrainCallError(f"gbrain http: response is not JSON: {body[:300]}")
    return message


def tool_payload(op: str, message: Mapping[str, Any]) -> dict[str, Any]:
    """Turn a ``tools/call`` response into the operation's JSON result (same shape as ``gbrain call``)."""
    error = message.get("error")
    if error:
        detail = error.get("message") if isinstance(error, Mapping) else error
        raise GBrainCallError(f"gbrain {op} failed over http: {str(detail)[:500]}")
    result = message.get("result")
    if not isinstance(result, Mapping):
        raise GBrainCallError(f"gbrain {op} returned no result over http")
    content = result.get("content")
    text = ""
    if isinstance(content, list) and content and isinstance(content[0], Mapping):
        text = str(content[0].get("text", ""))
    payload = parse_json_object(text)
    if result.get("isError"):
        detail = _error_detail(payload) if payload else text
        raise GBrainCallError(f"gbrain {op} error: {detail[:500]}")
    if payload is None:
        raise GBrainCallError(f"gbrain {op} returned no JSON over http: {text[:300]}")
    if payload.get("error"):
        raise GBrainCallError(f"gbrain {op} error: {_error_detail(payload)[:500]}")
    facts = payload.get("facts")
    if isinstance(facts, list):
        for fact in facts:
            if isinstance(fact, dict) and isinstance(fact.get("fact"), str):
                fact["fact"] = unwrap_quoted_fact(fact["fact"])
    return payload


class KeepAliveTransport:
    """POST /mcp over one reused HTTP connection (reconnects once if it dropped)."""

    def __init__(self, host: str, port: int, token: str, timeout_s: float) -> None:
        """Bind to a running server."""
        self._host, self._port, self._timeout_s = host, port, timeout_s
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        self._conn: http.client.HTTPConnection | None = None

    def __call__(self, body: bytes) -> tuple[str, str]:
        """Send one request and return (content type, body)."""
        for attempt in (1, 2):
            if self._conn is None:
                self._conn = http.client.HTTPConnection(self._host, self._port, timeout=self._timeout_s)
            try:
                self._conn.request("POST", "/mcp", body=body, headers=self._headers)
                response = self._conn.getresponse()
                text = response.read().decode("utf-8", errors="replace")
            except (http.client.HTTPException, OSError) as exc:
                self.close()
                if attempt == 2:
                    raise GBrainCallError(f"gbrain http request failed: {exc}") from exc
                continue
            if response.status >= 400:
                raise GBrainCallError(f"gbrain http status {response.status}: {text[:300]}")
            return response.getheader("Content-Type", ""), text
        raise GBrainCallError("gbrain http request failed")  # pragma: no cover - loop always returns or raises

    def close(self) -> None:
        """Drop the connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class GBrainHttpBackend(GBrainMemoryBackend):
    """A GBrain brain served by one resident ``gbrain serve --http`` process."""

    name = "gbrain-http"

    def __init__(
        self,
        *args: Any,
        transport: Transport | None = None,
        spawner: Spawner = spawn_process,
        ready_check: Callable[[int], bool] = health_ok,
        startup_timeout_s: float = GBRAIN_HTTP_STARTUP_TIMEOUT_S,
        **kwargs: Any,
    ) -> None:
        """Take the CLI backend's arguments; ``transport`` replaces the server (tests)."""
        super().__init__(*args, **kwargs)
        self._transport: Transport | None = transport
        self._spawner = spawner
        self._ready_check = ready_check
        self._startup_timeout_s = startup_timeout_s
        self._process: ServerProcess | None = None
        self._log: IO[bytes] | None = None
        self._token_name = ""
        self._request_id = 0
        self.port: int | None = None

    @property
    def log_path(self) -> Path:
        """Where the server's output goes."""
        return self.home / GBRAIN_HTTP_LOG_NAME

    def start_server(self, port: int | None = None) -> None:
        """Mint a token, start the server and wait for /health (raises :class:`GBrainCallError`)."""
        if self._transport is not None:
            return
        self._token_name = f"{GBRAIN_HTTP_TOKEN_NAME}-{int(time.time() * 1000)}"
        logger.info("토큰 발급: gbrain auth create %s", self._token_name)
        result = self._runner([self._binary, "auth", "create", self._token_name], self._env(), self._timeout_s)
        token = parse_token(result.stdout + "\n" + result.stderr)
        if result.returncode != 0 or token is None:
            output = (result.stdout + result.stderr).strip()
            raise GBrainCallError(f"gbrain auth create failed (exit {result.returncode}): {output[-500:]}")

        self.port = port or free_port()
        args = [
            self._binary,
            "serve",
            "--http",
            "--port",
            str(self.port),
            "--bind",
            GBRAIN_HTTP_HOST,
            "--surface",
            GBRAIN_HTTP_SURFACE,
            "--suppress-bootstrap-token",
        ]
        self.home.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("ab")
        logger.info(
            "서버 시작: gbrain serve --http (http://%s:%d, 로그 %s)", GBRAIN_HTTP_HOST, self.port, self.log_path
        )
        self._process = self._spawner(args, self._env(), self.home, self._log)

        started = time.monotonic()
        deadline = started + self._startup_timeout_s
        while not self._ready_check(self.port):
            if self._process.poll() is not None or time.monotonic() > deadline:
                reason = "exited" if self._process.poll() is not None else "did not answer /health in time"
                self.close()
                raise GBrainCallError(f"gbrain serve --http {reason}: {self._log_tail()}")
            time.sleep(GBRAIN_HTTP_POLL_INTERVAL_S)
        logger.info("서버 준비 완료: /health 응답까지 %.1fs", time.monotonic() - started)
        self._transport = KeepAliveTransport(GBRAIN_HTTP_HOST, self.port, token, self._timeout_s)

    def _log_tail(self) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")[-_LOG_TAIL_CHARS:].strip()
        except OSError:
            return ""

    def call(self, op: str, params: Mapping[str, Any]) -> dict[str, Any]:
        """Invoke one GBrain operation as an MCP ``tools/call``."""
        if self._transport is None:
            raise GBrainCallError(f"gbrain {op}: http server not started")
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": "tools/call",
            "params": {"name": op, "arguments": dict(params)},
        }
        content_type, body = self._transport(json.dumps(request, ensure_ascii=False).encode("utf-8"))
        return tool_payload(op, parse_rpc_response(content_type, body))

    def close(self) -> None:
        """Stop the server, then revoke its token (best effort)."""
        transport, self._transport = self._transport, None
        if isinstance(transport, KeepAliveTransport):
            transport.close()
        process, self._process = self._process, None
        if process is not None:
            logger.info("서버 종료 (port %s), 토큰 폐기", self.port)
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=GBRAIN_HTTP_STOP_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=GBRAIN_HTTP_STOP_TIMEOUT_S)
            if self._token_name:
                try:
                    self._runner([self._binary, "auth", "revoke", self._token_name], self._env(), self._timeout_s)
                except (OSError, subprocess.SubprocessError) as exc:
                    logger.warning("GBrain token revoke failed: %s", exc)
        if self._log is not None:
            self._log.close()
            self._log = None
