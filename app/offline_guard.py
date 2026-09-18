"""Proof, on demand, that the application never reaches for the network (NFR-4).

The application has no HTTP client and needs none: OCR, barcode decoding and PDF
rendering are all local. "Never calls the network" is a claim worth checking
rather than asserting, so this can make any attempt to open an internet
connection fail loudly. It is switched on by the environment variable below (so
it reaches the worker process too) and used by the packaged executable's
`--self-test` and by the test suite; normal runs are untouched.

Only internet-family connections that leave the machine are refused. Local
inter-process plumbing (a Unix-domain socket, a named pipe, a loopback
socket pair) is not "the network".
"""

from __future__ import annotations

import os
import socket

GUARD_ENV = "ARTWORK_COMPARE_OFFLINE_GUARD"
_INTERNET = (socket.AF_INET, socket.AF_INET6)
_LOOPBACK = ("localhost", "::1", "")  # this computer talking to itself is not "the network"

_installed = False


class NetworkAccessAttempted(RuntimeError):
    """Something tried to reach the network while the offline guard was on."""


def _refuse(what: str):
    raise NetworkAccessAttempted(f"the application tried to use the network ({what}), which it must never do")


def install() -> None:
    """Make internet connections and name lookups raise. Idempotent."""
    global _installed
    if _installed:
        return
    _installed = True

    original_connect, original_connect_ex, original_sendto = socket.socket.connect, socket.socket.connect_ex, socket.socket.sendto

    def leaves_the_machine(sock, address) -> bool:
        """An internet-family connection to somewhere other than this computer."""
        if sock.family not in _INTERNET:
            return False
        host = address[0] if isinstance(address, tuple) and address else address
        return not (isinstance(host, str) and (host in _LOOPBACK or host.startswith("127.")))

    def connect(self, address):
        if leaves_the_machine(self, address):
            _refuse(f"connect to {address}")
        return original_connect(self, address)

    def connect_ex(self, address):
        if leaves_the_machine(self, address):
            _refuse(f"connect to {address}")
        return original_connect_ex(self, address)

    def sendto(self, *args):
        if self.family in _INTERNET:
            _refuse("send a datagram")
        return original_sendto(self, *args)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.socket.sendto = sendto
    socket.create_connection = lambda *a, **k: _refuse("create_connection")
    socket.getaddrinfo = lambda *a, **k: _refuse("look up a host name")
    socket.gethostbyname = lambda *a, **k: _refuse("look up a host name")
    socket.gethostbyname_ex = lambda *a, **k: _refuse("look up a host name")


def install_if_requested() -> bool:
    """Install the guard if the environment asks for it. Returns whether it is on."""
    if os.environ.get(GUARD_ENV):
        install()
    return _installed
