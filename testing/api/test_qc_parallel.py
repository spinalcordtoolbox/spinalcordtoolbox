# pytest unit tests for spinalcordtoolbox.reports.qc in parallel
#
# This test spins up many parallel worker processes that all try to write to
# the same QC report, to reproduce and diagnose a reported 60s mutex timeout
# (16 parallel jobs across 267 subjects).
#
# It's built to distinguish between two competing hypotheses for the timeout:
#
#   (A) "Slow regeneration": each call to generate_qc() re-renders the *whole*
#       HTML report, which grows as more subjects are added. If true, the time
#       spent *holding* the lock (not just waiting for it) should trend upward
#       as more entries accumulate in the report.
#
#   (B) "Unfair queueing": portalocker does not grant the lock in FIFO order --
#       a process that just arrived can win the race for the lock ahead of a
#       process that has been waiting much longer. If true, we should see
#       acquisition order disagree with arrival order ("queue jumps"), and
#       individual processes can accumulate long waits even though no single
#       lock hold was very long.
#
# To measure this we monkeypatch portalocker.Lock.acquire/.release in every
# worker process so each lock attempt records
# (pid, request_time, acquired_time, released_time) into a
# multiprocessing.Manager().list() shared across all workers. After the pool
# finishes, we reconstruct the full lock timeline and report stats for both
# hypotheses, plus a raw JSON dump for further offline analysis.
#
# NOTE ON TEMP PATH: this deliberately does NOT use the shared `tmp_path_qc`
# fixture. That fixture backs a QC report directory reused across hundreds of
# other tests, so writing a large, slow, artificially-bloated report into it
# would risk interfering with (and slowing down) unrelated tests. Instead we
# build our own throwaway directory off pytest's built-in, per-test `tmp_path`
# fixture, so this test's QC report is fully isolated.
#
# USAGE:
#   Fast smoke test (small numbers, safe for CI), just run normally:
#       sct_testing testing/api/test_qc_parallel.py::test_many_qc_parallel
#
#   Full-scale stress test mirroring the reported bug (16 workers, 267
#   subjects) -- slow, meant to be run manually:
#       SCT_QC_STRESS_WORKERS=16 SCT_QC_STRESS_SUBJECTS=267 \
#       sct_testing -o log_cli=true testing/api/test_qc_parallel.py::test_many_qc_parallel
#
#   To also see portalocker's own debug logs (retries/backoff), bump the
#   log level: -o log_cli_level=DEBUG

import json
import logging
import multiprocessing
import os
import statistics
import sys
import time

import pytest
import portalocker

from spinalcordtoolbox.utils.sys import sct_test_path
import spinalcordtoolbox.reports.qc as qc


# ----------------------------------------------------------------------------
# Scale knobs. Defaults are small enough to run quickly in CI. Set the env
# vars below to reproduce (or exceed) the reported real-world failure.
# ----------------------------------------------------------------------------
N_WORKERS = int(os.environ.get("SCT_QC_STRESS_WORKERS", "0")) or 16
N_SUBJECTS = int(os.environ.get("SCT_QC_STRESS_SUBJECTS", "0")) or 1000

# Comfortably under the real 60s mutex timeout; waits at/above this are
# flagged as "near-timeout" even when nothing actually errors out.
NEAR_TIMEOUT_THRESHOLD_S = 45


def _instrument_portalocker(events):
    """
    Monkeypatch portalocker.Lock.acquire/.release so every lock attempt in
    this process records precise request/acquired/released timestamps into
    `events` (a Manager-backed list shared across all worker processes).

    NOTE: if your installed version of qc.py wraps portalocker in a custom
    mutex helper instead of calling portalocker.Lock directly, patch that
    wrapper's acquire/release instead -- the timing logic below doesn't care
    where the timestamps come from.
    """
    original_acquire = portalocker.Lock.acquire
    original_release = portalocker.Lock.release

    def instrumented_acquire(self, *args, **kwargs):
        request_time = time.time()
        result = original_acquire(self, *args, **kwargs)
        acquired_time = time.time()
        self._instrumented_record = {
            "pid": os.getpid(),
            "request_time": request_time,
            "acquired_time": acquired_time,
        }
        events.append(dict(self._instrumented_record))
        return result

    def instrumented_release(self, *args, **kwargs):
        result = original_release(self, *args, **kwargs)
        record = getattr(self, "_instrumented_record", None)
        if record is not None:
            events.append({
                "_release_for": (record["pid"], record["request_time"]),
                "released_time": time.time(),
            })
        return result

    portalocker.Lock.acquire = instrumented_acquire
    portalocker.Lock.release = instrumented_release


def _pool_initializer(events, log_level):
    """Runs once in every worker process before it picks up any tasks."""
    _instrument_portalocker(events)

    # Keep the original debug-logging setup too, for manual/local inspection.
    screen_handler = logging.StreamHandler(stream=sys.stderr)
    logger = logging.getLogger("portalocker.utils")
    logger.setLevel(log_level)
    logger.addHandler(screen_handler)


def gen_qc(args):
    subject_index, path_qc = args
    t2_image = sct_test_path('t2', 't2.nii.gz')
    t2_seg = sct_test_path('t2', 't2_seg-manual.nii.gz')

    call_start = time.time()
    error = None
    try:
        qc.generate_qc(
            fname_in1=t2_image,
            fname_seg=t2_seg,
            path_qc=str(path_qc),
            process="sct_deepseg_gm",
            # If your generate_qc() supports a `subject`/`dataset` kwarg and
            # entries with the same subject name get merged/overwritten
            # rather than appended, uncomment this to force N_SUBJECTS
            # genuinely distinct entries in the growing report:
            # subject=f"subject_{subject_index:04d}",
        )
    except Exception as e:  # noqa: BLE001 -- capture *any* failure, e.g. the lock timeout itself
        error = repr(e)
    call_end = time.time()

    return {
        "subject_index": subject_index,
        "pid": os.getpid(),
        "call_start": call_start,
        "call_end": call_end,
        "call_duration": call_end - call_start,
        "error": error,
    }


def _analyze_lock_events(raw_events):
    """
    Merge acquire/release records and compute the stats needed to tell
    hypothesis A (slow regeneration) apart from hypothesis B (unfair queue).
    """
    acquisitions = {}
    for ev in raw_events:
        if "_release_for" in ev:
            key = ev["_release_for"]
            if key in acquisitions:
                acquisitions[key]["released_time"] = ev["released_time"]
        else:
            key = (ev["pid"], ev["request_time"])
            ev = dict(ev)
            ev["released_time"] = None
            acquisitions[key] = ev

    records = sorted(acquisitions.values(), key=lambda r: r["request_time"])
    for r in records:
        r["wait_time"] = r["acquired_time"] - r["request_time"]
        r["hold_time"] = (r["released_time"] - r["acquired_time"]) if r["released_time"] else None

    # Hypothesis B: count "queue jumps" -- an acquisition J that completes
    # (acquired_time) before some earlier-arriving K that requested the lock
    # before J did, but hadn't yet acquired it when J showed up.
    overtakes = 0
    for i, k in enumerate(records):
        for j in records[i + 1:]:
            if j["request_time"] >= k["acquired_time"]:
                break  # j (and everything after it) arrived after k already had the lock
            if j["acquired_time"] < k["acquired_time"]:
                overtakes += 1

    return records, overtakes


def test_many_qc_parallel(tmp_path):
    """
    Stress-test spinalcordtoolbox.reports.qc under heavy parallel writes to a
    single QC report, and profile the file lock to help determine why
    real-world runs are timing out on the 60s mutex.
    """
    # Deliberately isolated from `tmp_path_qc` -- see module docstring.
    path_qc = tmp_path / "qc_stress"
    path_qc.mkdir()

    if multiprocessing.cpu_count() < 2:
        pytest.skip("Can't test parallel behaviour")

    manager = multiprocessing.Manager()
    events = manager.list()

    print(f"\n[test_many_qc_parallel] workers={N_WORKERS} subjects={N_SUBJECTS} path_qc={path_qc}")

    p = multiprocessing.Pool(
        N_WORKERS,
        initializer=_pool_initializer,
        initargs=(events, logging.DEBUG),
    )

    tasks = [(i, path_qc) for i in range(N_SUBJECTS)]

    # This `try, finally` pattern mitigates hanging with pytest-cov
    # See: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3661#issuecomment-1029057900
    try:
        results = p.map(gen_qc, tasks)
    finally:
        p.close()
        p.join()

    # --- Hypothesis A: does lock *hold* time (regeneration cost) grow over time? ---
    records, overtakes = _analyze_lock_events(list(events))
    holds = [r["hold_time"] for r in records if r["hold_time"] is not None]
    first_half_holds = holds[: len(holds) // 2] or [0]
    second_half_holds = holds[len(holds) // 2:] or [0]
    mean_hold_first = statistics.mean(first_half_holds)
    mean_hold_second = statistics.mean(second_half_holds)

    print("\n--- Hypothesis A: report regeneration slowing down as it grows ---")
    print(f"mean lock hold time, first half of acquisitions:  {mean_hold_first:.3f}s")
    print(f"mean lock hold time, second half of acquisitions: {mean_hold_second:.3f}s")
    print(f"hold-time growth ratio (second/first): {mean_hold_second / max(mean_hold_first, 1e-6):.2f}x")

    # --- Hypothesis B: lock queue fairness ---
    waits = [r["wait_time"] for r in records]
    near_timeouts = [r for r in records if r["wait_time"] >= NEAR_TIMEOUT_THRESHOLD_S]

    print("\n--- Hypothesis B: unfair lock queueing ---")
    print(f"total lock acquisitions: {len(records)}")
    print(f"max wait time:  {max(waits, default=0):.2f}s")
    print(f"mean wait time: {statistics.mean(waits) if waits else 0:.2f}s")
    print(f"max hold time:  {max(holds, default=0):.3f}s")
    print(f"'queue jump' events (later arrival acquired before earlier arrival): {overtakes}")
    print(f"waits >= {NEAR_TIMEOUT_THRESHOLD_S}s (near-timeout): {len(near_timeouts)}")

    errors = [r for r in results if r["error"]]
    if errors:
        print("\n--- Errors / timeouts encountered ---")
        for r in errors:
            print(f"subject {r['subject_index']} (pid {r['pid']}): {r['error']}")

    # Dump the full timeline for offline analysis (e.g. plotting wait/hold
    # time vs. acquisition order) when running this as a manual stress test.
    debug_dump = path_qc.parent / "qc_stress_timeline.json"
    with open(debug_dump, "w") as f:
        json.dump({"call_results": results, "lock_records": records}, f, indent=2, default=str)
    print(f"\nFull timeline written to: {debug_dump}")

    # The main point of this test is to *diagnose*, not just pass/fail -- but
    # CI should still flag it clearly if the mutex actually timed out.
    assert not errors, f"{len(errors)}/{len(results)} subjects hit an error acquiring the QC lock"