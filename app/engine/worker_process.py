"""The comparison, run inside a worker process (Architecture §5).

A malformed file can crash the native code underneath OpenCV and PyMuPDF. Run
in the UI process, that would take the whole application down mid-review, so the
comparison runs in a separate process instead: if it dies, only it does, the UI
reports a clean error, and a fresh worker replaces it.

This module is what the worker process runs, so it must not import Qt (a worker
that pulled in the GUI toolkit would start slower and use more memory for
nothing). The Qt side -- starting, watching, recycling and cancelling the worker
-- is `app.ui.job_controller`.

Protocol over the pipe (each message is a tuple):

  parent -> worker   ("job", job_id, ComparisonRequest)   run one comparison
                     ("prepare", ComparisonRequest)        run it ahead of Compare, quietly, and keep the work
                                                           (see `app.engine.prepared`); nothing is sent back
                     ("stop",)                             finish up and exit
  worker -> parent   ("ready",)                            imports done, waiting for a job
                     ("progress", job_id, text)            what it is doing now
                     ("result", job_id, ComparisonOutcome)
                     ("error", job_id, kind, message)      kind: see ErrorKind
"""

from __future__ import annotations

import gc
import logging
import traceback
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Callable

import numpy as np

from app import offline_guard
from app.config import AppConfig
from app.engine.alignment.aligner import AlignmentError
from app.engine.file_loader import ArtworkLoadError, load_artwork
from app.engine.isolation.auto_area import find_artwork, located_by_print
from app.engine.detectors.text_diff import shown_words, text_layer_words
from app.engine.isolation.block_grid import stroked_line_map
from app.engine.isolation.die_lines import vector_strokes
from app.engine import prepared
from app.engine.leaflet import NoPagesComparedError, compare_leaflet
from app.engine.page_choice import choose_pages
from app.engine import language_detect
from app.engine.language_detect import detect_languages
from app.models.languages import AUTO
from app.engine.ocr_locator import configure_ocr, usable_language
from app.engine.series import NoRepeatsFoundError, choose_series_pages, compare_series
from app.engine.aggregation.aggregator import build_clusters
from app.engine.word_strategy import spelling_changes, word_outcome
from app.engine.worker import ComparisonOutcome, run_comparison
from app.logging_config import job_log_file
from app.models.diff import renumbered, visible_by_source
from app.models.isolation import IsolationGeometry
from app.models.job import ComparisonMode

logger = logging.getLogger(__name__)


class ErrorKind:
    """Why a job failed, so the UI can word it for a person (never a traceback)."""

    INPUT = "input"  # a file could not be loaded, or is too large: the user can fix it
    ALIGNMENT = "alignment"  # the two files could not be registered
    MEMORY = "memory"  # the machine ran out of memory
    CRASH = "crash"  # the worker process died (a native-library fault, usually a corrupt file)
    TIMEOUT = "timeout"  # the job ran past its time limit and was stopped
    INTERNAL = "internal"  # a bug; details are in the job log


@dataclass(frozen=True)
class ComparisonRequest:
    """Everything a worker needs -- small, so it is cheap to send.

    Files are passed as paths and rendered inside the worker rather than sent as
    pixels: a 310 x 420 mm carton is 55 MB per file, and rendering there keeps
    that off the pipe.
    """

    original_path: Path
    suspect_path: Path
    mode: ComparisonMode
    config: AppConfig
    original_geometry: IsolationGeometry | None = None
    suspect_geometry: IsolationGeometry | None = None
    # Which page of each file to compare (0-based); None: the page most like the other file's (`page_choice`). A file
    # of one page is always compared by it. (Leaflet mode compares every page and pairs them itself.)
    original_page: int | None = None
    suspect_page: int | None = None
    # Leaflet mode: what the reviewer set on each page of each file (its own area, areas erased), as (page, geometry)
    # pairs in page order (a tuple, so a request says the same thing the same way every time).
    original_page_areas: tuple = ()
    suspect_page_areas: tuple = ()
    align_only: bool = False


def request_key(request: ComparisonRequest) -> bytes:
    """Everything a comparison's outcome depends on: the request, and what each file on disk is right now."""
    return prepared.settings_key(request, prepared.file_key(request.original_path), prepared.file_key(request.suspect_path))


def execute_request(
    request: ComparisonRequest,
    progress: Callable[[str], None] | None = None,
    reuse: bool = False,
    later: list | None = None,
) -> ComparisonOutcome:
    """Load, render and compare the two files. Runs in the worker; also usable directly.

    `reuse` (the worker's jobs and preparations): the outcome of exactly this request, if it was prepared already, is
    returned as it is, and a new one is kept for the next (see `app.engine.prepared`).

    `later` (a list): what can wait is left for after -- Automatic's word comparison, whose findings start hidden. The
    outcome returned then has `ocr_pending` set, and `later` holds one function that makes the whole outcome (and keeps
    it, with `reuse`). Without `later`, the whole outcome is made here."""
    try:
        key = request_key(request) if (reuse and not request.align_only) else None
    except OSError:
        key = None  # a file that is not there (any more): loading it says so, as the reviewer should hear it
    if key is not None:
        ready = prepared.outcome_for(key)
        if ready is not None:
            return ready
    deferred: list = []
    outcome = _execute(request, progress, (deferred if (later is not None and not request.align_only) else None))
    if deferred:

        def finished() -> ComparisonOutcome:
            whole = deferred[0]()
            if key is not None and not request.align_only:
                prepared.keep_outcome(key, whole)
            return whole

        later.append(finished)
        return outcome
    if key is not None and not request.align_only:
        prepared.keep_outcome(key, outcome)
    return outcome


def _rendered(artwork) -> np.ndarray:
    """The artwork's chosen page as a picture (kept, see `app.engine.prepared`)."""
    return prepared.remembered(
        "render",
        (prepared.file_key(artwork.path), artwork.page, artwork.working_dpi(), artwork.held_megapixels()),
        artwork.render,
    )


def _pdf_extras(artwork, image: np.ndarray) -> tuple:
    """A PDF's own stroked lines, words and stroked paths (with their colours), in its picture's pixels ((None, None,
    None) for an image file)."""
    if artwork.kind != "pdf":
        return None, None, None
    return prepared.remembered(
        "pdf_extras",
        (prepared.file_key(artwork.path), artwork.page, artwork.working_dpi(), image.shape[:2]),
        lambda: (
            stroked_line_map(artwork.pdf_page(), artwork.working_dpi(), image.shape[:2]),
            shown_words(text_layer_words(artwork.pdf_page(), artwork.working_dpi(), image.shape[:2]), image),
            vector_strokes(artwork.pdf_page(), artwork.working_dpi()),
        ),
    )


def _languages_begun(request: ComparisonRequest, original, suspect, original_image: np.ndarray, suspect_image: np.ndarray):
    """Start finding the languages the two files are printed in (`language_detect`), unless they were chosen: from a
    sample of each picture's lines and each PDF's own words -- once for these files and pages (kept, like the rest of a
    preparation). The future of it, or None."""
    if request.config.detection.ocr_language != AUTO:
        return None
    # (a PDF's own words are taken here, before the finding starts: a PDF is read by one thread at a time)
    texts = [
        [word.text for word in (text_layer_words(artwork.pdf_page(), artwork.working_dpi(), image.shape[:2]) or [])]
        if artwork.kind == "pdf" else []
        for artwork, image in ((original, original_image), (suspect, suspect_image))
    ]
    key = (
        prepared.file_key(original.path), original.page, original.working_dpi(),
        prepared.file_key(suspect.path), suspect.page, suspect.working_dpi(),
    )
    return language_detect.begin(
        lambda: prepared.remembered("languages", key, lambda: detect_languages([original_image, suspect_image], texts))
    )


def _execute(
    request: ComparisonRequest, progress: Callable[[str], None] | None = None, deferred: list | None = None
) -> ComparisonOutcome:
    say = progress or (lambda _text: None)
    language_detect.end()  # (whatever an earlier comparison found is not this one's)
    say("Loading the files…")
    limits = request.config.runtime
    with load_artwork(request.original_path, limits) as original, load_artwork(
        request.suspect_path, limits
    ) as suspect:
        if request.mode is ComparisonMode.LEAFLET:
            # Every page matters, not one: rendered and paired inside, not as a single "the" page here.
            outcome = compare_leaflet(
                original,
                suspect,
                request.config,
                progress=say,
                original_areas=dict(request.original_page_areas),
                suspect_areas=dict(request.suspect_page_areas),
                align_only=request.align_only,
            )
            for note in (original.reduced_note(), suspect.reduced_note()):
                if note:
                    outcome.notes.insert(0, note)
            return outcome

        page_notes = _select_pages(original, suspect, request)
        say("Rendering the artwork…")
        original_image, suspect_image = _rendered(original), _rendered(suspect)
        # The languages printed are found beside everything below (finding the artwork, aligning it, comparing it pixel
        # by pixel, none of which reads a word): the first reading waits only for what is left of it.
        languages = _languages_begun(request, original, suspect, original_image, suspect_image)
        original_geometry, suspect_geometry = request.original_geometry, request.suspect_geometry
        notes: list[str] = []
        def found_areas():
            return prepared.remembered(
                "find_artwork",
                (
                    prepared.file_key(original.path), original.page, original.working_dpi(),
                    prepared.file_key(suspect.path), suspect.page, suspect.working_dpi(),
                    prepared.settings_key(request.config.subject),
                ),
                lambda: find_artwork(
                    original_image,
                    suspect_image,
                    request.config.subject,
                    original.pdf_page() if original.kind == "pdf" else None,
                    suspect.pdf_page() if suspect.kind == "pdf" else None,
                    original.working_dpi() if original.kind == "pdf" else None,
                    suspect.working_dpi() if suspect.kind == "pdf" else None,
                ),
            )

        auto_find = request.config.detection.auto_artwork and original_geometry is None and suspect_geometry is None
        if auto_find and request.mode is not ComparisonMode.SERIES:
            # Nothing was cropped or lifted: the artwork is found on each sheet, trimmed to, aligned and compared.
            # Every mode benefits from this, not just Automatic -- Leaflet is the one exception, handled above,
            # since a leaflet's pages are paired and rendered inside `compare_leaflet` rather than here. (Series asks for
            # it only if the repeats cannot be found otherwise: pairing one artwork's border with a sheet of them
            # seldom works, and costs seconds.)
            say("Finding the artwork\u2026")
            found = found_areas()
            original_geometry, suspect_geometry = found.original, found.suspect
            if found.note:
                notes.append(found.note)
        if request.mode is ComparisonMode.SERIES:
            # The suspect is a whole sheet of repeats, never isolated by the reviewer: only the original's own
            # geometry (the one artwork, cropped or lifted as usual) means anything here.
            say("Finding the repeats…")
            expected_scale = None
            if original.kind == "pdf" and suspect.kind == "pdf" and original.working_dpi() > 0:
                expected_scale = suspect.working_dpi() / original.working_dpi()
            outcome = compare_series(
                original_image, suspect_image, request.mode, request.config, original_geometry, expected_scale,
                find_area=(lambda: found_areas().original) if auto_find else None,
                align_only=request.align_only,
            )
        else:
            # A PDF's own fold and cut lines, which the artwork is divided into blocks by (block-wise alignment), and its
            # own words, which say what a finding's text is without reading it from the picture.
            (original_lines, original_words, original_strokes), (suspect_lines, suspect_words, suspect_strokes) = (
                _pdf_extras(original, original_image),
                _pdf_extras(suspect, suspect_image),
            )
            lines, words = [original_lines, suspect_lines], [original_words, suspect_words]

            def compared(original_area, suspect_area) -> ComparisonOutcome:
                return run_comparison(
                    original_image,
                    suspect_image,
                    request.mode,
                    request.config,
                    original_geometry=original_area,
                    suspect_geometry=suspect_area,
                    progress=say,
                    original_lines=lines[0],
                    suspect_lines=lines[1],
                    original_words=words[0],
                    suspect_words=words[1],
                    original_strokes=original_strokes,
                    suspect_strokes=suspect_strokes,
                    align_only=request.align_only,
                )

            try:
                outcome = compared(original_geometry, suspect_geometry)
            except AlignmentError:
                if not auto_find:
                    raise  # (an area the reviewer chose is not second-guessed)
                # Neither a border nor the whole pages lined the files up (a proof sheet of several artworks and a job
                # sheet, against a scan of one of them): the artwork is found by what is printed, and compared alone.
                scale = suspect.working_dpi() / original.working_dpi() if original.kind == "pdf" and suspect.kind == "pdf" else None
                located = located_by_print(original_image, suspect_image, request.config.series, scale)
                if located is None:
                    raise
                notes.append(located.note)
                outcome = compared(located.original, located.suspect)
            if request.mode is ComparisonMode.AUTOMATIC and request.config.detection.automatic_ocr and not request.align_only:
                if deferred is not None:
                    # The pixel findings go first; the words are read after (`execute_request`'s `later`). (`pixels` is
                    # finished below -- its notes, its pages -- and the whole outcome takes them from it.)
                    pixels = replace(outcome, ocr_pending=True)

                    def with_words(compared_outcome=outcome, pixels=pixels) -> ComparisonOutcome:
                        language_detect.adopt(languages)  # (read after the job returned: in the languages it found)
                        whole = _with_word_findings(compared_outcome, words[0], words[1], request.config)
                        return replace(
                            whole, compare_mask=None, ocr_pending=False, notes=pixels.notes,
                            original_page=pixels.original_page, suspect_page=pixels.suspect_page,
                            original_dpi=pixels.original_dpi, suspect_dpi=pixels.suspect_dpi,
                        )

                    deferred.append(with_words)
                    outcome = pixels
                else:
                    say("Reading the words\u2026")
                    outcome = _with_word_findings(outcome, words[0], words[1], request.config)
            outcome.compare_mask = None  # (only the word comparison needed it: never sent back)
        said = language_detect.note(languages)
        if request.align_only:
            outcome.notes = ["Files updated and aligned. Click Recompare to run comparison."]
        else:
            outcome.notes = page_notes + notes + outcome.notes + ([said] if said else [])
        outcome.original_page, outcome.suspect_page = original.page, suspect.page
        outcome.original_dpi = original.working_dpi() if original.kind == "pdf" else None
        outcome.suspect_dpi = suspect.working_dpi() if suspect.kind == "pdf" else None
        for note in (original.reduced_note(), suspect.reduced_note()):
            if note:
                outcome.notes.insert(0, note)
        return outcome


def _with_word_findings(outcome: ComparisonOutcome, original_words, suspect_words, config) -> ComparisonOutcome:
    """Automatic mode's pixel comparison with its word comparison added (`app.engine.word_strategy`), on the same
    alignment and area, kept to the words spelt differently: those findings are kept with the result but hidden, for the
    results page's OCR toggle to show at once. A word comparison that fails leaves the pixel comparison as it was."""
    language, _installed = usable_language(config.detection.ocr_language)
    try:
        words = word_outcome(outcome, original_words, suspect_words, config, language)
    except Exception:  # noqa: BLE001 -- the OCR findings are extra: their failing must not lose the pixel comparison
        logger.exception("automatic mode's word comparison failed")
        return outcome
    # Automatic's OCR findings are spellings only: a word spelt differently, nothing else (see `spelling_changes`)
    spelt = spelling_changes(words.candidates)
    spelling_clusters = build_clusters(spelt, outcome.alignment.transform, config.clustering)
    pixels = list(outcome.all_clusters or outcome.clusters)
    everything = pixels + spelling_clusters
    shown = renumbered(visible_by_source(everything, ocr_wanted=False, pixels_wanted=True), everything)
    return replace(
        outcome,
        clusters=shown,
        all_clusters=everything,
        candidates=outcome.candidates + spelt,
        warnings=outcome.warnings + [f"OCR findings: {warning}" for warning in words.warnings],
        text_compared=False,  # the OCR findings start hidden...
        graphics_compared=True,  # ...the pixel ones shown
    )


def _select_pages(original, suspect, request: ComparisonRequest) -> list[str]:
    """Set the page each file is compared by (see `ComparisonRequest.original_page`), and say which, for a file of
    several pages -- so a reviewer is never left thinking a page was compared that was not."""
    series_pages = None
    if request.mode is ComparisonMode.SERIES and max(original.page_count, suspect.page_count) > 1:
        # a sheet of repeats shares every feature with the original ten times over, which is no way to tell pages apart:
        # the page is the one the sheet's own repeated copy is on
        series_pages = choose_series_pages(original, suspect, request.original_page, request.suspect_page, request.config.series)
    if series_pages is not None:
        first, second = series_pages
        first_guessed, second_guessed = request.original_page is None, request.suspect_page is None
    else:
        first, second, first_guessed, second_guessed = choose_pages(original, suspect, request.original_page, request.suspect_page)
    original.select_page(first)
    suspect.select_page(second)
    notes = []
    for artwork, guessed in ((original, first_guessed), (suspect, second_guessed)):
        if artwork.page_count > 1:
            how = "the page most like the other file's" if guessed else "as chosen"
            notes.append(
                f"'{artwork.path.name}' has {artwork.page_count} pages: page {artwork.page + 1} was compared ({how}; "
                "another can be picked with its page selector)."
            )
    return notes


def _classify(exc: BaseException) -> tuple[str, str]:
    """(ErrorKind, message a person can act on) for an exception raised by a job."""
    if isinstance(exc, ArtworkLoadError):
        return ErrorKind.INPUT, str(exc)
    if isinstance(exc, (AlignmentError, NoRepeatsFoundError, NoPagesComparedError)):
        return ErrorKind.ALIGNMENT, str(exc)
    if isinstance(exc, MemoryError):
        return ErrorKind.MEMORY, (
            "The computer ran out of memory comparing these files. Close other applications, "
            "or export the artwork at a smaller page size or resolution."
        )
    return ErrorKind.INTERNAL, (
        "The comparison failed unexpectedly. The details are in the job log; "
        f"the error was: {type(exc).__name__}: {exc}"
    )


def worker_main(connection: Connection) -> None:
    """The worker process's loop: take a job, run it, send back the result."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    offline_guard.install_if_requested()  # a spawned process starts clean; it must opt in like its parent
    configure_ocr()
    connection.send(("ready",))
    waiting: list = []  # a message a preparation gave way to, taken off the pipe: handled next
    while True:
        try:
            message = waiting.pop(0) if waiting else connection.recv()
        except (EOFError, OSError):
            return  # the app went away; nothing left to serve
        if message[0] == "stop":
            return
        if message[0] == "prepare":
            _prepare(connection, message[1], waiting)
            continue
        if message[0] != "job":
            continue

        _kind, job_id, request = message
        try:
            later: list = []
            with job_log_file(job_id):
                outcome = execute_request(
                    request, progress=lambda text, job=job_id: connection.send(("progress", job, text)), reuse=True, later=later
                )
            connection.send(("result", job_id, outcome))
            _send_the_rest(connection, job_id, later)
        except Exception as exc:  # a job's failure is reported, and the worker carries on
            error_kind, text = _classify(exc)
            if error_kind == ErrorKind.INTERNAL:
                logger.error("job %s failed\n%s", job_id, traceback.format_exc())
            connection.send(("error", job_id, error_kind, text))
        finally:
            # Large arrays from the finished job would otherwise wait for the next
            # collection while this process sits idle holding hundreds of MB.
            outcome = request = None
            gc.collect()


def _send_the_rest(connection: Connection, job_id: str, later: list) -> None:
    """What a job's result was sent without (`execute_request`'s `later`: Automatic's word comparison), made now and
    sent as an update to that result. Its failing loses nothing: the result stands as it was sent."""
    if not later:
        return
    try:
        with job_log_file(job_id):
            whole = later[0]()
        connection.send(("update", job_id, whole))
    except Exception:  # noqa: BLE001 -- the words are extra: the result already sent stands
        logger.error("job %s: the rest of its result could not be made\n%s", job_id, traceback.format_exc())
        connection.send(("update_failed", job_id))


def _prepare(connection: Connection, request: ComparisonRequest, waiting: list | None = None) -> None:
    """Run `request` ahead of Compare, quietly, keeping the work (see `app.engine.prepared`) -- unless something newer
    is already waiting, and giving way at the next stage the moment something newer arrives (a job is never held up
    by more than a stage, and never runs beside a preparation).

    Compare pressed for exactly this comparison while it is being prepared does not start it again: the preparation
    carries on AS that job -- its progress reported, its outcome the job's result -- so the reviewer waits only for what
    was left. Anything else that arrives is put in `waiting` (it has been taken off the pipe) and the preparation gives
    way to it."""
    waiting = waiting if waiting is not None else []
    try:
        key = _files_now(request)
    except OSError:
        key = None
    adopted: list = []  # the job this preparation became, if Compare asked for it

    def check(text: str) -> None:
        if adopted:
            connection.send(("progress", adopted[0], text))
        if not connection.poll():
            return
        message = connection.recv()
        if not adopted and message[0] == "job" and key is not None and _same_request(message[2], request, key):
            adopted.append(message[1])
            connection.send(("progress", message[1], text or "Comparing\u2026"))
            return
        waiting.append(message)
        raise prepared.Superseded()

    try:
        check("")
        later: list = []
        if adopted:
            with job_log_file(adopted[0]):
                outcome = execute_request(request, progress=check, reuse=True, later=later)
        else:
            outcome = execute_request(request, progress=check, reuse=True, later=later)
        if adopted:
            connection.send(("result", adopted[0], outcome))
            logger.info("prepared comparison handed to Compare")
            _send_the_rest(connection, adopted[0], later)
        else:
            if later:
                check("")  # (something newer already waiting: it is not held up by the words)
                whole = later[0]()  # the whole outcome kept, for Compare to find ready
                check("")
                if adopted:  # (Compare asked for exactly this while the words were read: it is all ready)
                    connection.send(("result", adopted[0], whole))
                    logger.info("prepared comparison handed to Compare")
                    return
            logger.info("comparison prepared ahead of Compare")
    except prepared.Superseded:
        logger.info("preparation overtaken by a newer request")
        if adopted:  # (only another message after Compare's own can do this: the job is run for real, next)
            waiting.insert(0, ("job", adopted[0], request))
    except Exception as exc:  # noqa: BLE001 -- a preparation that fails says nothing: Compare runs it for real and reports
        if adopted:
            error_kind, text = _classify(exc)
            connection.send(("error", adopted[0], error_kind, text))
        else:
            logger.info("preparation did not complete\n%s", traceback.format_exc())
    finally:
        outcome = None
        gc.collect()


def _files_now(request: ComparisonRequest) -> tuple:
    """What the two files on disk are right now (path, size, last written)."""
    return prepared.file_key(request.original_path), prepared.file_key(request.suspect_path)


def _same_request(request: ComparisonRequest, preparing: ComparisonRequest, files: tuple) -> bool:
    """Whether `request` asks for exactly the comparison being prepared: the same settings, by value (a request that
    has crossed the pipe does not pickle byte for byte like one that has not), of the same files, unchanged since."""
    try:
        return request == preparing and _files_now(request) == files
    except Exception:  # noqa: BLE001 -- a file gone, or settings that cannot be compared: not the same, run it afresh
        return False
