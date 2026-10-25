"""Persistent LLM budgets, bounded Claude subprocess, usage validation, call-provenance
hashes and the CLI budget/built-in-executor options.

Offline only: providers are mocks, injected transports/popen doubles, or tiny local
scripts standing in for the ``claude`` executable. No model is ever called.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
import types
from pathlib import Path

import pytest

from pod_platform.agents.budgets import (
    DEFAULT_BUDGET_ID,
    BudgetConfigError,
    BudgetError,
    BudgetIntegrityError,
    PersistentCallBudget,
    Reservation,
)
from pod_platform.agents.durable_cache import (
    CacheCorruptionError,
    DurableExtractionCache,
    LegacyProvenanceError,
    _canonical,
)
from pod_platform.agents.provider import (
    BudgetExhausted,
    CallBudget,
    ClaudeCodeProvider,
    DeterministicMockProvider,
    IncompleteOutputError,
    MalformedOutputError,
    OutputLimitExceeded,
    ProviderError,
    call_structured,
    parse_usage,
    redact_secrets,
    run_bounded_process,
    sha256_text,
)
from pod_platform.pods.demo import demo_provider

from .test_llm_provider import (
    Answer,
    FakeTransport,
    clock,
    extractor,
    http_provider,
    mock,
    request,
    run,
)
from .test_pod_research import DATASET, PAPERS, load_cli

UTC = dt.timezone.utc
ROOT = Path(__file__).resolve().parent.parent
posix_only = pytest.mark.skipif(
    os.name != "posix", reason="process-group semantics are POSIX-only"
)


def budget(tmp_path, **kw) -> PersistentCallBudget:
    args = dict(budget_id="b1", max_calls=3, max_tokens=10_000, clock=clock)
    args.update(kw)
    return PersistentCallBudget(tmp_path / "budget.sqlite", **args)


# ---------------------------------------------------------------- persistence / caps


def test_charges_survive_restart_and_the_cap_holds_across_instances(tmp_path):
    first = budget(tmp_path, max_calls=2)
    first.reserve(100)
    second = budget(tmp_path, max_calls=2)  # a restarted worker
    assert second.calls_used == 1 and second.tokens_used == 100
    second.reserve(100)
    third = budget(tmp_path, max_calls=2)
    with pytest.raises(BudgetExhausted) as info:
        third.reserve(1)
    assert "call budget exhausted (2/2)" in str(info.value)
    snap = third.snapshot()
    assert (
        snap["calls_used"] == 2
        and snap["denied_reservations"] == 1
        and snap["persistent"] is True
    )
    assert third.verify() == len(third.events())


def test_token_cap_is_enforced_before_the_call(tmp_path):
    b = budget(tmp_path, max_tokens=500)
    reserved = b.reserve(400)
    assert (
        isinstance(reserved, Reservation)
        and reserved == 400
        and len(reserved.reservation_id) == 32
    )
    with pytest.raises(BudgetExhausted) as info:
        b.reserve(101)
    assert "token budget exhausted" in str(info.value)
    assert b.snapshot()["remaining_tokens"] == 100


def test_limits_can_only_tighten_without_an_explicit_audited_raise(tmp_path):
    b = budget(tmp_path, max_calls=5, max_tokens=1000)
    b.reserve(300)
    b.reserve(300)
    lowered = budget(
        tmp_path, max_calls=3, max_tokens=700
    )  # tightening is allowed ...
    assert (lowered.max_calls, lowered.max_tokens) == (3, 700)
    assert (
        lowered.calls_used == 2 and lowered.tokens_used == 600
    )  # ... and never resets charges
    with pytest.raises(BudgetConfigError) as info:
        budget(tmp_path, max_calls=50, max_tokens=1000)
    assert "loosen" in str(info.value) and "new budget id or cycle" in str(
        info.value
    )
    # a mixed request cannot raise one dimension while lowering the other
    with pytest.raises(BudgetConfigError):
        budget(tmp_path, max_calls=1, max_tokens=5000)
    assert budget(tmp_path, max_calls=3, max_tokens=700).calls_used == 2
    raised = budget(tmp_path, max_calls=9, max_tokens=5000, raise_limits=True)
    assert (
        raised.max_calls == 9
        and raised.calls_used == 2
        and raised.tokens_used == 600
    )
    kinds = [e["kind"] for e in raised.events()]
    assert (
        kinds.count("limit_lowered") == 1 and kinds.count("limit_raised") == 1
    )


def test_accounts_are_scoped_by_budget_id_pod_and_cycle(tmp_path):
    a = budget(tmp_path, max_calls=1)
    a.reserve(1)
    for other in (
        budget(tmp_path, budget_id="b2", max_calls=1),
        budget(tmp_path, pod="valuation", max_calls=1),
        budget(tmp_path, cycle="2026q4", max_calls=1),
    ):
        other.reserve(1)  # a different named account has its own allowance
        assert other.calls_used == 1 and other.account_key != a.account_key
    with pytest.raises(BudgetExhausted):
        budget(tmp_path, max_calls=1).reserve(1)
    assert a.calls_used == 1


@pytest.mark.parametrize(
    "kw",
    [
        dict(max_calls=-1),
        dict(max_tokens=-5),
        dict(max_calls=True),
        dict(max_tokens=1.5),
        dict(max_calls=10**9),
        dict(budget_id=""),
        dict(budget_id="has space"),
        dict(cycle="a/b"),
    ],
)
def test_config_bounds_are_validated(tmp_path, kw):
    with pytest.raises(BudgetError):
        budget(tmp_path, **kw)


def test_reserve_rejects_nonsense_estimates(tmp_path):
    b = budget(tmp_path)
    for bad in (-1, True, 1.5, 10**13):
        with pytest.raises(BudgetError):
            b.reserve(bad)
    assert b.calls_used == 0


# -------------------------------------------------------------------- settlement


def test_settlement_is_idempotent_and_never_refunds_twice(tmp_path):
    b = budget(tmp_path)
    reserved = b.reserve(1000)
    b.settle(reserved, 400)  # refund of 600
    assert b.tokens_used == 400
    b.settle(reserved, 400)  # replay of the same report: no effect
    assert b.tokens_used == 400
    b.settle(reserved, 10)  # a conflicting report is audited, not applied
    assert b.tokens_used == 400
    conflicts = [e for e in b.events() if e["kind"] == "settle_conflict"]
    assert (
        len(conflicts) == 1
        and conflicts[0]["payload"]["first_actual"] == 400
        and conflicts[0]["payload"]["applied"] is False
    )
    assert budget(tmp_path).tokens_used == 400  # durable
    with pytest.raises(BudgetError):
        b.settle(1000, 5, reservation_id="not-a-reservation")
    assert b.verify() == len(b.events())


def test_unknown_usage_keeps_the_reservation_without_inventing_precision(
    tmp_path,
):
    b = budget(tmp_path)
    reserved = b.reserve(700)
    b.settle(reserved, None)
    assert (
        b.tokens_used == 700
        and b.snapshot()["unknown_usage_calls"] == 1
        and b.snapshot()["unsettled_reservations"] == 0
    )
    b.settle(
        reserved, 5
    )  # a later number cannot retroactively shrink an unknown-usage call
    assert b.tokens_used == 700
    b.settle(
        700, None
    )  # a bare int cannot identify a reservation: nothing to do
    assert b.tokens_used == 700


def test_bare_int_settlement_never_refunds_but_records_overspend(tmp_path):
    b = budget(tmp_path)
    b.reserve(500)
    b.settle(500, 100)  # not a Reservation: cannot be matched, so no refund
    assert b.tokens_used == 500
    b.settle(500, 900)  # but usage beyond the reservation is still charged
    assert b.tokens_used == 900 and any(
        e["kind"] == "unattributed_settlement" for e in b.events()
    )


def test_nonsensical_usage_reports_are_rejected(tmp_path):
    b = budget(tmp_path)
    reserved = b.reserve(10)
    for bad in (-1, True, 1.5, "3", 10**10):
        with pytest.raises(BudgetError):
            b.settle(reserved, bad)
    assert b.tokens_used == 10 and b.snapshot()["unsettled_reservations"] == 1


def test_overspend_is_charged_and_stops_later_calls(tmp_path):
    b = budget(tmp_path, max_tokens=1000)
    reserved = b.reserve(100)
    b.settle(reserved, 1500)  # the provider billed far more than reserved
    snap = b.snapshot()
    assert (
        snap["tokens_used"] == 1500
        and snap["overspent_tokens"] == 500
        and snap["remaining_tokens"] == 0
    )
    with pytest.raises(BudgetExhausted) as info:
        b.reserve(1)
    assert "overspent" in str(info.value)
    with pytest.raises(BudgetExhausted):
        budget(tmp_path, max_tokens=1000).reserve(
            0
        )  # still true after a restart, even for a zero estimate


# ----------------------------------------------------------- call_structured wiring


def request_for(**kw):
    return request(user="in", **kw)


def test_call_structured_settles_with_provider_reported_usage(tmp_path):
    b = budget(tmp_path)
    call_structured(mock(), request_for(), Answer, budget=b, clock=clock)
    assert (
        b.calls_used == 1 and b.tokens_used == 7
    )  # mock usage: 1 prompt + 6 completion tokens
    assert b.snapshot()["unsettled_reservations"] == 0
    # malformed output still arrived and billed: it is settled, not left reserved
    bad = DeterministicMockProvider(lambda r: "not json", clock=clock)
    with pytest.raises(MalformedOutputError):
        call_structured(bad, request_for(), Answer, budget=b, clock=clock)
    assert (
        b.calls_used == 2
        and b.snapshot()["unsettled_reservations"] == 0
        and b.tokens_used < 100
    )


def test_failed_and_interrupted_provider_calls_keep_their_reservation_across_restart(
    tmp_path,
):
    def down(req):
        raise ProviderError("down", retryable=True)

    b = budget(tmp_path, max_calls=2)
    with pytest.raises(ProviderError):
        call_structured(
            DeterministicMockProvider(down, clock=clock),
            request_for(),
            Answer,
            budget=b,
            clock=clock,
        )

    def interrupted(req):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        call_structured(
            DeterministicMockProvider(interrupted, clock=clock),
            request_for(),
            Answer,
            budget=b,
            clock=clock,
        )
    restarted = budget(tmp_path, max_calls=2)
    snap = restarted.snapshot()
    assert snap["calls_used"] == 2 and snap["unsettled_reservations"] == 2
    assert (
        snap["tokens_used"] >= 2 * 2048
    )  # the full conservative estimate is still charged
    provider = mock()
    with pytest.raises(BudgetExhausted):
        call_structured(
            provider, request_for(), Answer, budget=restarted, clock=clock
        )
    assert (
        provider.calls == 0
    )  # nothing reaches the provider once the account is spent


def test_call_budget_in_memory_rejects_nonsense_actual():
    b = CallBudget(3, 1000)
    reserved = b.reserve(100)
    for bad in (-1, True, 2.5, 10**10):
        with pytest.raises(ValueError):
            b.settle(reserved, bad)
    b.settle(reserved, None)
    assert b.tokens_used == 100


# ------------------------------------------------------------------ concurrency


def test_two_connections_cannot_both_take_the_last_slot(tmp_path):
    path = tmp_path / "budget.sqlite"
    kw = dict(budget_id="race", max_calls=5, max_tokens=10**6)
    PersistentCallBudget(path, **kw)
    outcomes: list[str] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(12)

    def worker():
        try:
            mine = PersistentCallBudget(
                path, **kw
            )  # its own connection, like another worker process
            barrier.wait(timeout=30)
            try:
                mine.reserve(10)
                outcomes.append("ok")
            except BudgetExhausted:
                outcomes.append("denied")
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors
    assert outcomes.count("ok") == 5 and outcomes.count("denied") == 7
    final = PersistentCallBudget(path, **kw)
    assert (
        final.calls_used == 5
        and final.tokens_used == 50
        and final.verify() >= 12
    )


def test_separate_processes_share_one_cap(tmp_path):
    path = tmp_path / "budget.sqlite"
    PersistentCallBudget(path, budget_id="mp", max_calls=4, max_tokens=10**6)
    code = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(ROOT)!r})
        from pod_platform.agents.budgets import PersistentCallBudget
        from pod_platform.agents.provider import BudgetExhausted
        b = PersistentCallBudget({str(path)!r}, budget_id="mp", max_calls=4, max_tokens=10**6)
        ok = 0
        for _ in range(3):
            try:
                b.reserve(5); ok += 1
            except BudgetExhausted:
                pass
        print(ok)
        """
    )
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(4)
    ]
    results = []
    for proc in procs:
        out, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, err
        results.append(int(out.strip()))
    assert sum(results) == 4  # 12 attempts, 4 allowed in total
    assert (
        PersistentCallBudget(
            path, budget_id="mp", max_calls=4, max_tokens=10**6
        ).calls_used
        == 4
    )


# ----------------------------------------------------------------- tamper evidence


def test_budget_events_are_immutable_and_tampering_is_detected(tmp_path):
    b = budget(tmp_path)
    b.reserve(100)
    conn = sqlite3.connect(b.path)
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("UPDATE budget_events SET tokens_delta = 0")
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("DELETE FROM budget_events")
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("UPDATE budget_accounts SET budget_id = 'x'")
    conn.execute(
        "DROP TRIGGER budget_events_no_update"
    )  # an attacker with file access
    conn.execute(
        "UPDATE budget_events SET tokens_delta = 0 WHERE kind = 'reserve'"
    )
    conn.commit()
    conn.close()
    with pytest.raises(BudgetIntegrityError):
        b.verify()
    with pytest.raises(BudgetIntegrityError):
        budget(tmp_path)  # verification on open
    with pytest.raises(BudgetIntegrityError):
        b.reserve(1)  # and the tail is re-checked before every new charge


# ------------------------------------------------------- OpenAI-compatible provider


def body_with_usage(usage):
    return json.dumps(
        {
            "model": "served",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"label":"a","score":1}',
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }
    ).encode()


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": -1, "completion_tokens": 5},
        {"prompt_tokens": True, "completion_tokens": 5},
        {"prompt_tokens": "10", "completion_tokens": 5},
        {"prompt_tokens": 1.5, "completion_tokens": 5},
        {"prompt_tokens": 10**12, "completion_tokens": 5},
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 3},
        "lots",
    ],
)
def test_openai_usage_that_is_negative_or_nonsensical_is_rejected(usage):
    provider = http_provider(FakeTransport(200, body_with_usage(usage)))
    with pytest.raises(ProviderError) as info:
        provider.complete(request())
    assert "usage" in str(info.value) and len(info.value.response_hash) == 64


def test_openai_usage_absent_partial_and_total_larger_than_sum(tmp_path):
    none = http_provider(FakeTransport(200, body_with_usage(None))).complete(
        request()
    )
    assert (
        none.prompt_tokens is None
        and none.completion_tokens is None
        and none.total_tokens is None
    )
    partial = http_provider(
        FakeTransport(200, body_with_usage({"prompt_tokens": 9}))
    ).complete(request())
    assert (
        partial.prompt_tokens is None
    )  # half a pair is not a usable settlement
    assert parse_usage(
        {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 9}
    ) == (1, 2, 9)
    b = budget(tmp_path)
    provider = http_provider(
        FakeTransport(
            200,
            body_with_usage(
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 40,
                }
            ),
        )
    )
    call_structured(provider, request(), Answer, budget=b, clock=clock)
    assert (
        b.tokens_used == 40
    )  # never settle below what the provider says it billed
    unknown = http_provider(FakeTransport(200, body_with_usage(None)))
    before = b.tokens_used
    call_structured(unknown, request(), Answer, budget=b, clock=clock)
    assert (
        b.tokens_used > before + 2000
        and b.snapshot()["unknown_usage_calls"] == 1
    )  # whole reservation kept


def test_openai_usage_above_the_reservation_is_charged_and_stops_the_account(
    tmp_path,
):
    b = budget(tmp_path, max_tokens=5000)
    huge = http_provider(
        FakeTransport(
            200,
            body_with_usage({"prompt_tokens": 9000, "completion_tokens": 100}),
        )
    )
    call_structured(
        huge, request(max_tokens=100), Answer, budget=b, clock=clock
    )
    assert b.tokens_used == 9100 and b.snapshot()["overspent_tokens"] == 4100
    with pytest.raises(BudgetExhausted):
        call_structured(
            huge, request(max_tokens=1), Answer, budget=b, clock=clock
        )


def test_errors_never_carry_credentials():
    leaky = FakeTransport(
        401,
        b'{"error": "Authorization: Bearer abcdef123456 key sk-secret and sk-other-1234567890"}',
    )
    with pytest.raises(ProviderError) as info:
        http_provider(leaky).complete(request())
    text = str(info.value)
    for needle in ("abcdef123456", "sk-secret", "sk-other-1234567890"):
        assert needle not in text
    assert "HTTP 401" in text

    def exploding(url, headers, body, timeout_s, max_bytes):
        raise RuntimeError(
            f"cannot reach {url} with {headers['Authorization']}"
        )

    with pytest.raises(ProviderError) as info:
        http_provider(exploding).complete(request())
    assert (
        "sk-secret" not in str(info.value)
        and "llm.example" not in str(info.value)
        and "RuntimeError" in str(info.value)
    )
    assert redact_secrets("a\x00b\x1b[31m", None) == "a?b?[31m"
    assert redact_secrets("token sk-abcdefghij", "zzz") == "token [redacted]"


# ----------------------------------------------------------- bounded Claude subprocess


def make_script(tmp_path, body: str, name="fake_claude") -> str:
    path = tmp_path / name
    path.write_text(f"#!{sys.executable}\n" + textwrap.dedent(body))
    path.chmod(0o755)
    if os.name == "posix" and os.statvfs(tmp_path).f_flag & os.ST_NOEXEC:
        pytest.skip("temp directory is mounted noexec")
    return str(path)


def alive(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        try:
            return stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
        except (OSError, IndexError):
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_dead(pid: int, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not alive(pid):
            return True
        time.sleep(0.05)
    return not alive(pid)


@posix_only
def test_bounded_claude_runs_toolless_in_an_empty_temp_dir_without_shell(
    tmp_path,
):
    script = make_script(
        tmp_path,
        """
        import json, os, sys
        data = sys.stdin.buffer.read()
        print(json.dumps({"label": os.getcwd(), "score": len(data), "argv": sys.argv[1:], "listing": os.listdir(".")}))
        """,
    )
    provider = ClaudeCodeProvider("claude-x", executable=script, clock=clock)
    response = provider.complete(request(user='{"q": "data"}'))
    doc = json.loads(response.text)
    assert (
        "claude_research_" in doc["label"]
        and Path(doc["label"]) != Path.cwd()
        and doc["listing"] == []
    )
    assert doc["score"] > 10  # the prompt arrived on stdin
    argv = doc["argv"]
    assert argv[:6] == [
        "-p",
        "--safe-mode",
        "--tools",
        "",
        "--strict-mcp-config",
        "--no-session-persistence",
    ]
    assert not any(
        "dangerously" in a or "permission" in a for a in argv
    ) and "data" not in " ".join(argv)
    assert response.ignored_params == ("max_tokens", "temperature", "seed")
    assert (
        response.prompt_tokens is None
        and response.completion_tokens is None
        and response.total_tokens is None
    )


@posix_only
def test_bounded_claude_kills_a_process_that_floods_stdout(tmp_path):
    script = make_script(
        tmp_path,
        """
        import sys
        chunk = b"x" * 65536
        try:
            for _ in range(5000):  # ~320 MB if nobody stopped it
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
        except BrokenPipeError:
            pass
        """,
    )
    started = time.monotonic()
    with pytest.raises(ProviderError) as info:
        ClaudeCodeProvider(executable=script, max_output_bytes=1000).complete(
            request(max_tokens=32768)
        )
    assert "configured limit" in str(info.value) and not isinstance(
        info.value, IncompleteOutputError
    )
    assert time.monotonic() - started < 30


@posix_only
def test_bounded_claude_stops_early_at_the_max_tokens_plausibility_bound(
    tmp_path,
):
    script = make_script(tmp_path, "import sys\nsys.stdout.write('x' * 5000)\n")
    with pytest.raises(IncompleteOutputError) as info:
        ClaudeCodeProvider(executable=script).complete(
            request(max_tokens=10)
        )  # bound: 80 bytes
    assert "does not enforce max_tokens" in str(info.value)


@posix_only
def test_bounded_claude_caps_stderr(tmp_path):
    script = make_script(
        tmp_path,
        """
        import sys
        sys.stderr.write("e" * 300000)
        sys.stderr.flush()
        print('{"label":"a","score":1}')
        """,
    )
    with pytest.raises(OutputLimitExceeded) as info:
        ClaudeCodeProvider(executable=script, max_stderr_bytes=1000).complete(
            request()
        )
    assert info.value.stream == "stderr" and len(info.value.captured) <= 1000
    ok = make_script(
        tmp_path,
        'import sys\nsys.stderr.write(\'warn\')\nprint(\'{"label":"a","score":1}\')\n',
        "quiet",
    )
    assert (
        json.loads(ClaudeCodeProvider(executable=ok).complete(request()).text)[
            "label"
        ]
        == "a"
    )


@posix_only
def test_bounded_claude_timeout_kills_the_whole_process_group(tmp_path):
    script = make_script(
        tmp_path,
        """
        import os, subprocess, sys, time
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        with open(sys.argv[0] + ".pids", "w") as fh:
            fh.write(f"{os.getpid()} {child.pid}")
        time.sleep(120)
        """,
    )
    started = time.monotonic()
    with pytest.raises(ProviderError) as info:
        ClaudeCodeProvider(executable=script, timeout_s=2.0).complete(
            request(timeout_s=2.0)
        )
    assert info.value.retryable and "timed out" in str(info.value)
    assert time.monotonic() - started < 30
    pids_file = Path(script + ".pids")
    if (
        pids_file.exists()
    ):  # the script got far enough to record itself within the timeout
        parent, grandchild = (int(x) for x in pids_file.read_text().split())
        assert wait_dead(parent) and wait_dead(grandchild)


@posix_only
def test_bounded_claude_timeout_applies_even_if_the_child_never_reads_stdin(
    tmp_path,
):
    script = make_script(tmp_path, "import time\ntime.sleep(120)\n")
    big = (
        "x" * 1_500_000
    )  # larger than any pipe buffer: the stdin writer would block forever
    started = time.monotonic()
    with pytest.raises(ProviderError) as info:
        ClaudeCodeProvider(executable=script, timeout_s=1.0).complete(
            request(user=big, timeout_s=1.0)
        )
    assert info.value.retryable and time.monotonic() - started < 30


@posix_only
def test_bounded_claude_error_paths(tmp_path):
    with pytest.raises(ProviderError, match="status 3"):
        ClaudeCodeProvider(
            executable=make_script(tmp_path, "import sys\nsys.exit(3)\n", "bad")
        ).complete(request())
    with pytest.raises(ProviderError, match="empty output"):
        ClaudeCodeProvider(
            executable=make_script(tmp_path, "pass\n", "empty")
        ).complete(request())
    with pytest.raises(ProviderError, match="unavailable"):
        ClaudeCodeProvider(
            executable=str(tmp_path / "does-not-exist")
        ).complete(request())
    calls = []
    guarded = ClaudeCodeProvider(
        executable=make_script(tmp_path, "pass\n", "never"),
        max_input_bytes=10,
        popen=lambda *a, **k: calls.append(a),
    )
    with pytest.raises(ProviderError, match="input limit"):
        guarded.complete(request())
    assert calls == []  # an oversized prompt never launches anything
    for bad in (
        dict(max_output_bytes=0),
        dict(max_input_bytes=True),
        dict(max_stderr_bytes=-1),
    ):
        with pytest.raises(ValueError):
            ClaudeCodeProvider(**bad)


class FakeProc:
    pid = (
        2**22 + 12345
    )  # not a real process: must never be signalled via killpg

    def __init__(self, out=b"", err=b"", returncode=None):
        self.stdin, self.stdout, self.stderr = (
            io.BytesIO(),
            io.BytesIO(out),
            io.BytesIO(err),
        )
        self.returncode = returncode
        self.killed = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


def test_bounded_runner_with_popen_double_enforces_caps_without_signalling_real_pids():
    seen = {}

    def factory(cmd, **kwargs):
        seen["cmd"], seen["kwargs"] = cmd, kwargs
        seen["proc"] = FakeProc(out=b"x" * 100)
        return seen["proc"]

    with pytest.raises(OutputLimitExceeded) as info:
        run_bounded_process(
            ["claude", "-p"],
            input_bytes=b"hi",
            cwd="/",
            timeout_s=5,
            max_stdout_bytes=10,
            popen=factory,
        )
    assert (
        seen["proc"].killed
        and info.value.stream == "stdout"
        and len(info.value.captured) == 10
    )
    kwargs = seen["kwargs"]
    assert (
        kwargs["shell"] is False
        and kwargs["stdin"] == subprocess.PIPE
        and kwargs["cwd"] == "/"
    )
    done = run_bounded_process(
        ["claude"],
        input_bytes=b"hi",
        cwd="/",
        timeout_s=5,
        max_stdout_bytes=100,
        popen=lambda cmd, **kw: FakeProc(
            out=b'{"a":1}', err=b"w", returncode=0
        ),
    )
    assert (
        done.stdout == b'{"a":1}'
        and done.stderr == b"w"
        and done.returncode == 0
    )
    with pytest.raises(ValueError):
        run_bounded_process(
            ["x"],
            input_bytes=b"",
            cwd="/",
            timeout_s=float("inf"),
            max_stdout_bytes=1,
            popen=factory,
        )


def test_provider_with_popen_double_and_injected_runner_paths_are_both_supported():
    provider = ClaudeCodeProvider(
        popen=lambda cmd, **kw: FakeProc(
            out=b'{"label":"a","score":1}', returncode=0
        ),
        clock=clock,
    )
    assert json.loads(provider.complete(request()).text) == {
        "label": "a",
        "score": 1,
    }
    seen = []

    def runner(cmd, **kwargs):
        seen.append(kwargs)
        return types.SimpleNamespace(
            stdout=b'{"label":"a","score":1}', stderr=b"", returncode=0
        )

    ClaudeCodeProvider(runner=runner, clock=clock).complete(request())
    assert (
        seen[0]["shell"] is False and seen[0]["capture_output"] is True
    )  # legacy double sees the old call shape
    bounded = ClaudeCodeProvider(
        runner=subprocess.run,
        popen=lambda cmd, **kw: FakeProc(
            out=b'{"label":"b","score":2}', returncode=0
        ),
    )
    assert (
        "b" in bounded.complete(request()).text
    )  # naming subprocess.run selects the bounded (capped) path instead


# ------------------------------------------------------ durable cache provenance hashes


def swap_triggers(path):
    """Simulate an attacker with file access: prove the real append-only guard exists and bites, then remove it."""
    assert Path(path).exists(), (
        "the cache file must already exist (never create an empty database here)"
    )
    conn = sqlite3.connect(path)
    triggers = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' AND tbl_name = 'extraction_cache'"
        )
    }
    assert triggers == {
        "extraction_cache_no_update",
        "extraction_cache_no_delete",
    }, triggers
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("UPDATE extraction_cache SET namespace = namespace")
    conn.rollback()
    conn.execute("DROP TRIGGER extraction_cache_no_update")
    return conn


def test_call_provenance_is_hash_verified_alongside_fields(tmp_path):
    ex, cache = extractor(mock(), tmp_path)
    first = run(ex)
    assert (
        first.call_provenance == "verified"
        and cache.provenance_status(first.cache_key) == "verified"
    )
    hit = run(ex)
    assert (
        hit.from_cache
        and hit.call_provenance == "verified"
        and hit.call.response_hash == first.call.response_hash
    )
    assert cache.verify_all() == {
        "entries": 1,
        "provenance": {"verified": 1},
        "errors": [],
    }

    tampered = first.call.model_copy(
        update={"latency_s": 99.0}
    ).model_dump_json()
    conn = swap_triggers(cache.path)
    conn.execute("UPDATE extraction_cache SET call_json = ?", (tampered,))
    conn.commit()
    conn.close()
    with pytest.raises(CacheCorruptionError, match="call-provenance"):
        cache.get_call_record(first.cache_key)
    with pytest.raises(CacheCorruptionError):
        run(
            ex
        )  # a replay never trusts a record whose provenance fails its hash
    report = cache.verify_all()
    assert report["errors"] and report["provenance"] == {}


def test_call_records_cannot_be_swapped_between_entries_and_metadata_is_bound(
    tmp_path,
):
    ex, cache = extractor(mock(), tmp_path)
    a = run(ex, {"q": 1})
    b = run(ex, {"q": 2})
    conn = swap_triggers(cache.path)
    rows = dict(conn.execute("SELECT key, call_json FROM extraction_cache"))
    conn.execute(
        "UPDATE extraction_cache SET call_json = ? WHERE key = ?",
        (rows[b.cache_key], a.cache_key),
    )
    conn.commit()
    conn.close()
    with pytest.raises(CacheCorruptionError):
        cache.get_call_record(a.cache_key)
    for column, value in (
        ("namespace", "other_pod"),
        ("created_at", "2020-01-01T00:00:00+00:00"),
    ):
        ex2, cache2 = extractor(mock(), tmp_path / column)
        c = run(ex2)
        conn = swap_triggers(cache2.path)
        conn.execute(f"UPDATE extraction_cache SET {column} = ?", (value,))
        conn.commit()
        conn.close()
        with pytest.raises(
            CacheCorruptionError, match="entry hash"
        ):  # both columns are inside entry_hash
            cache2.get(c.cache_key)


def legacy_cache(tmp_path, source):
    """A cache file as written before call/entry hashes existed (6 columns, no verification)."""
    path = tmp_path / "legacy" / "cache.sqlite"
    path.parent.mkdir(parents=True)
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE extraction_cache (key TEXT PRIMARY KEY, fields_json TEXT NOT NULL, fields_hash TEXT NOT NULL,
            created_at TEXT NOT NULL, call_json TEXT, namespace TEXT NOT NULL DEFAULT '');
        CREATE TRIGGER extraction_cache_no_update BEFORE UPDATE ON extraction_cache
        BEGIN SELECT RAISE(ABORT, 'extraction cache is append-only'); END;
        CREATE TRIGGER extraction_cache_no_delete BEFORE DELETE ON extraction_cache
        BEGIN SELECT RAISE(ABORT, 'extraction cache is append-only'); END;
        """
    )
    fields_json = _canonical(source.record.fields)
    conn.execute(
        "INSERT INTO extraction_cache VALUES (?,?,?,?,?,?)",
        (
            source.cache_key,
            fields_json,
            sha256_text(fields_json),
            source.record.created_at.isoformat(),
            source.call.model_dump_json(),
            "pod_a",
        ),
    )
    conn.commit()
    conn.close()
    return tmp_path / "legacy"


def test_legacy_cache_is_migrated_explicitly_without_fabricating_validation(
    tmp_path,
):
    src_ex, _ = extractor(mock(), tmp_path / "src")
    source = run(src_ex)
    legacy_dir = legacy_cache(tmp_path, source)
    provider = mock()
    ex, cache = extractor(provider, legacy_dir)
    assert {
        r[1]
        for r in sqlite3.connect(cache.path).execute(
            "PRAGMA table_info(extraction_cache)"
        )
    } >= {"call_hash", "entry_hash"}
    assert cache.provenance_status(source.cache_key) == "legacy_unverified"
    hit = run(ex)
    # the validated fields still verify (fields_hash), but the call record is NOT attributed
    assert (
        hit.from_cache
        and provider.calls == 0
        and hit.call is None
        and hit.call_provenance == "legacy_unverified"
    )
    with pytest.raises(LegacyProvenanceError):
        cache.get_call_record(source.cache_key)
    assert (
        cache.get_call_record(
            source.cache_key, allow_unverified=True
        ).response_hash
        == source.call.response_hash
    )
    assert cache.verify_all()["provenance"] == {"legacy_unverified": 1}

    with pytest.raises(ValueError):
        cache.migrate_legacy_provenance(operator=" ", note="x")
    report = cache.migrate_legacy_provenance(
        operator="ops", note="reviewed on 2026-10-25"
    )
    assert (
        report["migrated"] == 1
        and report["validated_retroactively"] is False
        and report["corrupt"] == []
    )
    migrated = run(ex)
    assert (
        migrated.call_provenance == "legacy_migrated"
        and migrated.call.response_hash == source.call.response_hash
    )
    again = cache.migrate_legacy_provenance(operator="ops", note="again")
    assert again["migrated"] == 0 and again["already_migrated"] == 1

    changed = source.call.model_copy(
        update={"latency_s": 123.0}
    ).model_dump_json()
    conn = swap_triggers(cache.path)
    conn.execute("UPDATE extraction_cache SET call_json = ?", (changed,))
    conn.commit()
    conn.close()
    with pytest.raises(
        CacheCorruptionError, match="after its provenance migration"
    ):
        cache.get_call_record(source.cache_key)


def test_corrupt_legacy_rows_are_reported_not_migrated(tmp_path):
    src_ex, _ = extractor(mock(), tmp_path / "src")
    source = run(src_ex)
    legacy_dir = legacy_cache(tmp_path, source)
    conn = swap_triggers(legacy_dir / "cache.sqlite")
    conn.execute(
        'UPDATE extraction_cache SET fields_json = \'{"label":"evil","score":9}\''
    )
    conn.commit()
    conn.close()
    cache = DurableExtractionCache(legacy_dir / "cache.sqlite")
    report = cache.migrate_legacy_provenance(operator="ops", note="n")
    assert report["migrated"] == 0 and report["corrupt"] == [source.cache_key]
    with pytest.raises(CacheCorruptionError):
        cache.get(source.cache_key)


def test_new_rows_without_a_call_record_and_reopening_are_stable(tmp_path):
    from pod_platform.agents.extractor import ExtractionRecord

    cache = DurableExtractionCache(tmp_path / "c.sqlite")
    now = dt.datetime(2026, 10, 25, tzinfo=UTC)
    assert (
        cache.put(
            "k",
            ExtractionRecord(key="k", fields={"v": 1}, created_at=now),
            namespace="n",
        )
        is True
    )
    assert (
        cache.provenance_status("k") == "none"
        and cache.get_call_record("k") is None
    )
    again = DurableExtractionCache(
        tmp_path / "c.sqlite"
    )  # reopening a current-format file changes nothing
    assert again.get("k").fields == {"v": 1} and again.verify_all()[
        "provenance"
    ] == {"none": 1}
    with pytest.raises(KeyError):
        again.provenance_status("missing")


# ------------------------------------------------------------------------- the CLI


def submit_job(cli, state, tmp_path, question, capsys):
    cards = tmp_path / "papers.json"
    cards.write_text(json.dumps(PAPERS))
    data = tmp_path / "data.json"
    data.write_text(json.dumps({"cards": [DATASET]}))
    asof = dt.datetime.now(UTC).replace(microsecond=0).isoformat()
    argv = {
        "command": "submit",
        "state_dir": state,
        "pod": "valuation",
        "question": question,
        "paper_cards": str(cards),
        "dataset_cards": str(data),
        "executor_id": ["exec_a"],
        "research_asof": asof,
    }
    assert cli.main(argv) == 0
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])[
        "job_id"
    ]


def online_args(state, settings):
    return {
        "command": "run",
        "state_dir": state,
        "online": True,
        "provider": "claude-code",
        "max_jobs": 1,
        **settings,
    }


def test_online_runs_share_one_durable_budget_across_restarts(
    tmp_path, capsys, monkeypatch
):
    state = str(tmp_path / "state")
    provider = demo_provider()
    cli = load_cli()
    monkeypatch.setattr(cli, "_build_provider", lambda args: provider)
    job1 = submit_job(cli, state, tmp_path, "First question?", capsys)
    job2 = submit_job(cli, state, tmp_path, "Second question?", capsys)
    assert cli.main(online_args(state, {"max_calls": 1})) == 0
    first = capsys.readouterr()
    assert provider.calls == 1 and '"state": "succeeded"' in first.out
    assert (
        "ONLINE budget account" in first.err
        and '"persistent": true' in first.err
    )
    budget_file = Path(state) / "budget.sqlite"
    assert budget_file.exists()

    cli_restarted = load_cli()  # a fresh process: nothing in memory
    monkeypatch.setattr(cli_restarted, "_build_provider", lambda args: provider)
    assert (
        cli_restarted.main(online_args(state, {"max_calls": 1})) == 1
    )  # the job fails closed on the spent cap
    second = capsys.readouterr()
    assert provider.calls == 1 and "PROVIDER_BUDGET_EXCEEDED" in second.out
    assert (
        json.loads(second.err.strip().splitlines()[-1])["budget"]["calls_used"]
        == 1
    )
    assert job1 != job2

    # raising the cap needs an explicit, audited flag; a new named account has its own allowance
    assert cli_restarted.main(online_args(state, {"max_calls": 5})) == 2
    assert "loosen" in capsys.readouterr().err
    assert (
        cli_restarted.main(
            online_args(state, {"max_calls": 5, "raise_budget_limits": True})
        )
        == 0
    )
    capsys.readouterr()
    reopened = PersistentCallBudget(
        budget_file, budget_id=DEFAULT_BUDGET_ID, max_calls=5, max_tokens=100000
    )
    assert (
        reopened.calls_used == 1 and reopened.max_calls == 5
    )  # the stable default account; charges kept
    submit_job(cli, state, tmp_path, "Third question?", capsys)
    assert (
        cli_restarted.main(
            online_args(state, {"max_calls": 1, "budget_id": "other-account"})
        )
        == 0
    )
    capsys.readouterr()
    assert provider.calls == 2
    assert (
        PersistentCallBudget(
            budget_file,
            budget_id="other-account",
            max_calls=1,
            max_tokens=100000,
        ).calls_used
        == 1
    )


def test_demo_budget_is_separate_in_memory_and_clearly_labelled(
    tmp_path, capsys
):
    cli = load_cli()
    state = str(tmp_path / "state")
    submit_job(cli, state, tmp_path, "Demo question?", capsys)
    assert cli.main({"command": "run", "state_dir": state, "demo": True}) == 0
    captured = capsys.readouterr()
    assert (
        "NOT durable" in captured.err
        and not (Path(state) / "budget.sqlite").exists()
    )
    summary = json.loads(captured.err.strip().splitlines()[-1])["budget"]
    assert (
        summary["mode"] == "demo"
        and summary["persistent"] is False
        and summary["calls_used"] == 1
    )


def test_per_pod_scope_gives_each_pod_its_own_account(
    tmp_path, capsys, monkeypatch
):
    cli = load_cli()
    state = str(tmp_path / "state")
    monkeypatch.setattr(cli, "_build_provider", lambda args: demo_provider())
    assert (
        cli.main(online_args(state, {"budget_scope": "per-pod"})) == 0
    )  # empty queue: accounts are still opened
    err = capsys.readouterr().err
    assert '"accounts"' in err
    conn = sqlite3.connect(Path(state) / "budget.sqlite")
    pods = {r[0] for r in conn.execute("SELECT pod FROM budget_accounts")}
    assert len(pods) == 4 and "*" not in pods


def fake_builtins_module(calls):
    module = types.ModuleType("pod_platform.research.builtins")

    def register_executors(registry, *, manifest_path=None, state_dir=None):
        calls.append((manifest_path, state_dir))
        registry.register("builtin_fake", lambda req: {})

    module.register_executors = register_executors
    return module


def test_dataset_manifest_registers_builtin_executors_lazily(
    tmp_path, capsys, monkeypatch
):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "pod_platform.research.builtins",
        fake_builtins_module(calls),
    )
    cli = load_cli()
    state = str(tmp_path / "state")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    assert (
        cli.main({"command": "run", "state_dir": state, "demo": True}) == 0
    )  # no manifest: nothing registered, nothing required
    assert (
        cli.main({"command": "run", "state_dir": state}) == 0
    )  # dry run never needs one
    assert (
        cli.main(
            {
                "command": "run",
                "state_dir": state,
                "dataset_manifest": str(tmp_path / "missing.json"),
            }
        )
        == 0
    )  # dry run
    assert calls == []
    assert (
        cli.main(
            {
                "command": "run",
                "state_dir": state,
                "demo": True,
                "dataset_manifest": str(manifest),
            }
        )
        == 0
    )
    assert (
        len(calls) == 1
        and calls[0][0] == manifest
        and calls[0][1] == Path(state) / "builtin_executors"
    )
    assert (Path(state) / "builtin_executors").is_dir()
    capsys.readouterr()


def test_dataset_manifest_errors_are_clean_and_executors_module_still_works(
    tmp_path, capsys, monkeypatch
):
    cli = load_cli()
    state = str(tmp_path / "state")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    assert (
        cli.main(
            {
                "command": "run",
                "state_dir": state,
                "demo": True,
                "dataset_manifest": str(tmp_path / "missing.json"),
            }
        )
        == 2
    )
    assert "dataset-manifest not found" in capsys.readouterr().err
    monkeypatch.setitem(
        sys.modules, "pod_platform.research.builtins", None
    )  # import fails
    assert (
        cli.main(
            {
                "command": "run",
                "state_dir": state,
                "demo": True,
                "dataset_manifest": str(manifest),
            }
        )
        == 2
    )
    assert "built-in executors are unavailable" in capsys.readouterr().err

    registered = []
    user_module = types.ModuleType("my_pod_executors")
    user_module.register_executors = lambda registry: registered.append(
        registry.register("user_exec", lambda r: {})
    )
    monkeypatch.setitem(sys.modules, "my_pod_executors", user_module)
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "pod_platform.research.builtins",
        fake_builtins_module(calls),
    )
    assert (
        cli.main(
            {
                "command": "run",
                "state_dir": state,
                "demo": True,
                "executors_module": "my_pod_executors",
            }
        )
        == 0
    )
    assert registered == [None] and calls == []
    assert (
        cli.main(
            {
                "command": "run",
                "state_dir": state,
                "demo": True,
                "dataset_manifest": str(manifest),
                "executors_module": "my_pod_executors",
            }
        )
        == 0
        and len(calls) == 1
        and len(registered) == 2
    )
