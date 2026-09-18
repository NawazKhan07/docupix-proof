"""The application never calls the network (NFR-4), and can prove it."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from app import offline_guard

APP = Path(__file__).resolve().parent.parent / "app"

# Anything here in the app would be a way to reach a network.
_FORBIDDEN_MODULES = {
    "requests", "urllib3", "httpx", "aiohttp", "httplib2", "websocket", "websockets",
    "http.client", "urllib.request", "ftplib", "smtplib", "poplib", "imaplib", "telnetlib",
    "xmlrpc.client", "paramiko", "boto3", "socketserver", "http.server", "asyncio.streams",
}
_ALLOWED_TO_IMPORT_SOCKET = {"offline_guard.py"}  # it exists to block sockets


def _imports(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def test_no_module_in_the_application_imports_a_network_library():
    offenders = {}
    for path in APP.rglob("*.py"):
        imported = _imports(path)
        bad = {name for name in imported if name in _FORBIDDEN_MODULES or name.split(".")[0] in {"requests", "urllib3", "httpx", "aiohttp"}}
        if "socket" in imported and path.name not in _ALLOWED_TO_IMPORT_SOCKET:
            bad.add("socket")
        if bad:
            offenders[str(path.relative_to(APP))] = sorted(bad)
    assert offenders == {}


def test_no_url_is_fetched_anywhere_in_the_source():
    """A belt-and-braces text check for the calls that would fetch one."""
    calls = ("urlopen(", "urlretrieve(", "requests.get(", "requests.post(", "create_connection(")
    hits = [
        f"{path.name}: {call}"
        for path in APP.rglob("*.py")
        if path.name != "offline_guard.py"
        for call in calls
        if call in path.read_text(encoding="utf-8")
    ]
    assert hits == []


def _run_guarded(code: str) -> subprocess.CompletedProcess:
    """Run `code` in a fresh interpreter with the guard on (it patches the socket module globally)."""
    script = "from app import offline_guard\noffline_guard.install()\n" + textwrap.dedent(code)
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60, cwd=APP.parent)


@pytest.mark.parametrize(
    "attempt",
    [
        'import socket; socket.create_connection(("example.com", 80), timeout=1)',
        'import socket; socket.getaddrinfo("example.com", 80)',
        'import socket; socket.gethostbyname("example.com")',
        'import socket; s = socket.socket(); s.connect(("93.184.216.34", 80))',
        'import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.sendto(b"x", ("8.8.8.8", 53))',
        'import socket; s = socket.socket(); print(s.connect_ex(("93.184.216.34", 80)))',
    ],
    ids=["create_connection", "getaddrinfo", "gethostbyname", "connect", "sendto", "connect_ex"],
)
def test_the_guard_makes_any_attempt_to_reach_the_network_fail(attempt):
    result = _run_guarded(attempt)
    assert result.returncode != 0
    assert "NetworkAccessAttempted" in result.stderr


def test_the_guard_leaves_the_machine_talking_to_itself_alone():
    result = _run_guarded(
        """
        import socket
        a, b = socket.socketpair()   # loopback on Windows
        a.sendall(b"hi"); assert b.recv(2) == b"hi"
        listener = socket.socket(); listener.bind(("127.0.0.1", 0)); listener.listen()
        client = socket.socket(); client.connect(listener.getsockname())   # to itself
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_the_guard_is_off_unless_asked_for(monkeypatch):
    monkeypatch.delenv(offline_guard.GUARD_ENV, raising=False)
    monkeypatch.setattr(offline_guard, "_installed", False)
    assert offline_guard.install_if_requested() is False


def test_the_whole_application_runs_offline():
    """The self-test that ships in the executable: a real worker process, PDF rendering, a comparison
    and symbol decoding, with every network attempt in either process turned into a failure."""
    result = subprocess.run(
        [sys.executable, "-m", "app.main", "--self-test"], capture_output=True, text=True, timeout=240, cwd=APP.parent
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASSED" in result.stdout
    assert "comparison in the worker process" in result.stdout
