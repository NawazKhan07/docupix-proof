"""Stand-in worker processes for testing how the job controller treats a worker.

The real worker takes seconds to import OpenCV and friends; these start in a
fraction of a second and misbehave on demand. They live in a module (not in the
test file) because a spawned process imports its target by name.
"""

from __future__ import annotations

import os
import time


def _serve(connection, handle):
    connection.send(("ready",))
    while True:
        try:
            message = connection.recv()
        except (EOFError, OSError):
            return
        if message[0] == "stop":
            return
        if message[0] == "job":
            handle(connection, message[1], message[2])


def echo_main(connection):
    """Reports progress, then answers with its own process id and the request it got."""

    def handle(conn, job_id, request):
        conn.send(("progress", job_id, "working"))
        conn.send(("result", job_id, {"pid": os.getpid(), "request": request}))

    _serve(connection, handle)


def slow_main(connection):
    """Takes far longer than any test will wait."""

    def handle(conn, job_id, request):
        conn.send(("progress", job_id, "thinking"))
        time.sleep(60)
        conn.send(("result", job_id, "too late"))

    _serve(connection, handle)


def crash_main(connection):
    """Dies without a word, as a native-library fault does."""

    def handle(conn, job_id, request):
        os._exit(3)

    _serve(connection, handle)


def failing_main(connection):
    """Reports a user-facing failure."""

    def handle(conn, job_id, request):
        conn.send(("error", job_id, "input", "the file is too large"))

    _serve(connection, handle)


def never_ready_main(connection):
    """Starts, but takes a while before it says it is ready."""
    time.sleep(1.0)
    _serve(connection, lambda conn, job_id, request: conn.send(("result", job_id, "ok")))


def prepare_recording_main(connection):
    """Remembers every comparison it is asked to prepare, and answers a job with what it prepared before it."""
    prepared = []
    connection.send(("ready",))
    while True:
        try:
            message = connection.recv()
        except (EOFError, OSError):
            return
        if message[0] == "stop":
            return
        if message[0] == "prepare":
            prepared.append(message[1])
        elif message[0] == "job":
            connection.send(("result", message[1], {"prepared": list(prepared), "request": message[2]}))


def deaf_while_busy_main(connection):
    """Takes the first preparation, then reads nothing for a few seconds (a long stage), then answers a job with every
    preparation that reached it."""
    prepared = []
    connection.send(("ready",))
    while True:
        try:
            message = connection.recv()
        except (EOFError, OSError):
            return
        if message[0] == "stop":
            return
        if message[0] == "prepare":
            prepared.append(message[1])
            if len(prepared) == 1:
                time.sleep(4.0)
        elif message[0] == "job":
            connection.send(("result", message[1], {"prepared": len(prepared)}))


class _Sent:
    """A result as the worker sends it, with only what the controller looks at."""

    def __init__(self, ocr_pending: bool, label: str):
        self.ocr_pending = ocr_pending
        self.label = label


def ahead_main(connection):
    """Answers a job with a result sent ahead of its rest, then (a moment later) the rest as an update."""
    connection.send(("ready",))
    while True:
        try:
            message = connection.recv()
        except (EOFError, OSError):
            return
        if message[0] == "stop":
            return
        if message[0] == "job":
            connection.send(("result", message[1], _Sent(True, "pixels")))
            time.sleep(0.5)
            connection.send(("update", message[1], _Sent(False, "whole")))
