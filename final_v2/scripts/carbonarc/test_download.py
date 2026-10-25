"""Deterministic safety/recovery checks; no live analytical queries or real sleeps."""

import asyncio
import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import download as d


def result(data="", **metadata):
    return {"structuredContent": {"data": data, "metadata": metadata}}


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.q = d.Queue(self.tmp.name)
        self.addCleanup(self.q.db.close)
        self.metric = {
            "insight_id": 379,
            "insight_label": "Website Traffic",
            "topic_label": "Core Panel",
            "entities": [{"id": 568, "rep": "website"}],
        }

    def job(self):
        d.add_history(self.q, self.metric, "2020-01-01", "2020-12-31")
        return self.q.db.execute("SELECT * FROM jobs").fetchone()

    def test_minute_window_counts_rejected_attempts(self):
        job = self.job()
        for n in range(8):
            self.q.reserve(job, now=10000 + n)
        self.assertGreaterEqual(self.q.delay("text_to_insight", now=10008), 53)
        self.assertEqual(self.q.delay("text_to_insight", now=10070), 0)

    def test_hour_window_survives_restart_and_shares_analytics_bucket(self):
        with self.q.db:
            self.q.db.executemany(
                "INSERT INTO calls VALUES(?,?)",
                [("analytics", 10000 + i) for i in range(90)],
            )
        reopened = d.Queue(self.tmp.name)
        self.addCleanup(reopened.db.close)
        self.assertGreater(
            reopened.delay("framework_to_insight", now=10200), 3300
        )
        self.assertEqual(reopened.delay("framework_to_insight", now=14000), 0)

    def test_retry_after_and_cooldown_persist(self):
        job = self.job()
        response = result(
            rate_limited=True, admission_rejected=True, retry_after=120
        )
        self.assertEqual(d.classify(response), ("rate_limited", 120))
        self.q.defer(job, 121, now=1000)
        self.assertEqual(self.q.delay("text_to_insight", now=1001), 120)
        self.assertEqual(
            d.retry_seconds("Thu, 01 Jan 1970 00:20:00 GMT", now=1000), 200
        )

    def test_no_error_or_default_window_is_counted_as_download(self):
        self.assertEqual(d.classify(result(error="ERROR 101"))[0], "error")
        self.assertEqual(
            d.classify(result("KeyError: '__import__'"))[0], "error"
        )
        self.assertEqual(
            d.classify(
                result(
                    "table", frameworks=[{"default_date_range_applied": "True"}]
                )
            )[0],
            "default_range",
        )
        self.assertEqual(
            d.classify(result("result set is too large to display"))[0],
            "oversized",
        )
        self.assertEqual(d.classify(result(""))[0], "empty_unverified")

    def test_split_ranges_are_inclusive_nonoverlapping_and_oldest_first(self):
        job = self.job()
        self.assertEqual(
            d.build_children(
                self.q,
                job,
                result("result set is too large"),
                "1900-01-01",
                "2020-12-31",
            ),
            "split",
        )
        details = [
            json.loads(r["details"])
            for r in self.q.db.execute(
                "SELECT * FROM jobs WHERE id!=? ORDER BY priority", (job["id"],)
            )
        ]
        self.assertEqual(len(details), 2)
        self.assertEqual(details[0]["start"], "2020-01-01")
        self.assertEqual(details[1]["end"], "2020-12-31")
        self.assertEqual(
            dt.date.fromisoformat(details[0]["end"]) + dt.timedelta(days=1),
            dt.date.fromisoformat(details[1]["start"]),
        )

    def test_single_day_oversized_needs_bulk_export(self):
        d.add_history(self.q, self.metric, "2020-01-01", "2020-01-01")
        job = self.q.db.execute("SELECT * FROM jobs").fetchone()
        self.assertEqual(
            d.build_children(
                self.q,
                job,
                result("result set is too large"),
                "1900-01-01",
                "2020-12-31",
            ),
            "needs_bulk_export",
        )

    def test_discovery_uses_advertised_history_and_idempotent_queue(self):
        key = self.q.add(
            "data_library",
            {"insight_id": 379},
            {"kind": "provenance", "metric": self.metric},
            "0:",
        )
        job = self.q.db.execute(
            "SELECT * FROM jobs WHERE id=?", (key,)
        ).fetchone()
        response = {"structuredContent": {"data": {"history": "2020-Present"}}}
        for _ in range(2):
            d.build_children(self.q, job, response, "1900-01-01", "2026-10-24")
        jobs = self.q.db.execute(
            "SELECT * FROM jobs WHERE tool='text_to_insight'"
        ).fetchall()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(json.loads(jobs[0]["details"])["start"], "2020-01-01")

    def test_crash_after_claim_remains_running_until_explicit_recovery(self):
        job = self.job()
        self.q.reserve(job, now=10000)
        reopened = d.Queue(self.tmp.name)
        self.addCleanup(reopened.db.close)
        self.assertEqual(reopened.status(), {"running": 1})

    def test_raw_response_and_tables_saved_separately(self):
        job = self.job()
        self.q.reserve(job)
        response = result(
            "| Date | Visits |\n| --- | --- |\n| 2020-01-01 | 123 |"
        )
        path = d.save_response(self.q, job, response)
        self.assertEqual(json.loads(path.read_text())["response"], response)
        csvs = list(Path(self.tmp.name).glob("responses/*.csv"))
        self.assertEqual(len(csvs), 1)
        self.assertIn("2020-01-01,123", csvs[0].read_text())

    def test_wrong_insight_or_silently_shortened_range_is_flagged(self):
        job = self.job()
        wrong_metric = result(
            "table", frameworks=[{"insights": [{"insight_id": 381}]}]
        )
        self.assertEqual(
            d.validate_scope(job, wrong_metric, "downloaded_unverified"),
            "scope_mismatch",
        )
        shortened = result(
            "table",
            frameworks=[
                {
                    "framework_request": {
                        "filters": {"date_range": {"start_date": "2020-09-01"}}
                    }
                }
            ],
        )
        self.assertEqual(
            d.validate_scope(job, shortened, "downloaded_unverified"),
            "scope_mismatch",
        )

    def test_worker_retries_same_request_and_checkpoints_without_real_sleeps(
        self,
    ):
        self.job()
        attempts = []

        class Reply:
            def __init__(self, value):
                self.value = value

            def model_dump(self, **_):
                return self.value

        class Session:
            async def call_tool(self, tool, args):
                attempts.append((tool, args))
                return Reply(
                    result(rate_limited=True, retry_after=120)
                    if len(attempts) == 1
                    else result(
                        "| Date | Visits |\n| --- | --- |\n| 2020-01-01 | 10 |"
                    )
                )

        clock = [10000]

        async def fake_sleep(seconds):
            clock[0] += seconds

        async def fake_inspect(*_):
            return {"text_to_insight": {"inputSchema": {"type": "object"}}}

        from argparse import Namespace

        args = Namespace(
            max_requests=2,
            fallback_start="1900-01-01",
            end="2020-12-31",
            start=None,
            discovery_only=False,
        )
        with (
            patch.object(d.time, "time", side_effect=lambda: clock[0]),
            patch.object(d.asyncio, "sleep", fake_sleep),
            patch.object(d, "inspect_tools", fake_inspect),
        ):
            asyncio.run(d.work(Session(), self.q, args, {"sent": 0}))
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0], attempts[1])
        self.assertGreaterEqual(clock[0], 10121)
        self.assertEqual(self.q.status(), {"downloaded_unverified": 1})
        self.assertEqual(
            len(list(Path(self.tmp.name).glob("responses/*.json"))), 2
        )

    def test_discovery_only_never_submits_paid_history_jobs(self):
        self.job()

        async def fake_inspect(*_):
            return {"text_to_insight": {"inputSchema": {"type": "object"}}}

        class Session:
            async def call_tool(self, *_):
                raise AssertionError(
                    "Paid tool must never be called in discovery-only mode"
                )

        from argparse import Namespace

        args = Namespace(max_requests=0, discovery_only=True)
        with patch.object(d, "inspect_tools", fake_inspect):
            asyncio.run(d.work(Session(), self.q, args, {"sent": 0}))
        self.assertEqual(self.q.status(), {"pending": 1})

    def test_startup_retries_interrupted_free_jobs_but_never_paid_jobs(self):
        paid = self.job()
        key = self.q.add(
            "data_library",
            {"insight_id": 379},
            {"kind": "provenance", "metric": self.metric},
            "0:",
        )
        free = self.q.db.execute(
            "SELECT * FROM jobs WHERE id=?", (key,)
        ).fetchone()
        self.q.reserve(paid)
        self.q.reserve(free)
        self.q.recover_interrupted()
        states = dict(self.q.db.execute("SELECT id,status FROM jobs"))
        self.assertEqual(states[paid["id"]], "uncertain")
        self.assertEqual(states[free["id"]], "pending")

    def test_transport_429_requeues_same_free_task(self):
        key = self.q.add(
            "data_library",
            {"insight_id": 379},
            {"kind": "provenance", "metric": self.metric},
            "0:",
        )
        job = self.q.db.execute(
            "SELECT * FROM jobs WHERE id=?", (key,)
        ).fetchone()
        self.q.reserve(job)
        budget = {"active_id": key}
        self.assertEqual(
            d.recover_transport_limit(self.q, budget, {"retry_after": "120"}),
            121,
        )
        self.assertEqual(self.q.status(), {"pending": 1})
        self.assertIsNone(budget["active_id"])

    def test_paid_transport_retries_only_proven_rejected_post(self):
        job = self.job()
        self.q.reserve(job)
        budget = {"active_id": job["id"]}
        self.assertIsNone(
            d.recover_transport_limit(self.q, budget, {"retry_after": "10"})
        )
        self.assertEqual(self.q.status(), {"uncertain": 1})
        self.q.reserve(job)
        rejection = {
            "retry_after": "60",
            "tool": job["tool"],
            "arguments": json.loads(job["args"]),
        }
        self.assertEqual(
            d.recover_transport_limit(self.q, budget, rejection), 61
        )
        self.assertEqual(self.q.status(), {"pending": 1})

    def test_paid_cap_blocks_an_eleventh_attempt_after_restart(self):
        job = self.job()
        for _ in range(10):
            self.q.reserve(job, now=1)
        self.q.finish(job, "pending")
        reopened = d.Queue(self.tmp.name)
        self.addCleanup(reopened.db.close)

        async def fake_inspect(*_):
            return {"text_to_insight": {"inputSchema": {"type": "object"}}}

        class Session:
            async def call_tool(self, *_):
                raise AssertionError(
                    "An eleventh paid attempt must never be sent"
                )

        from argparse import Namespace

        args = Namespace(
            discovery_only=False, max_requests=0, max_paid_requests=10
        )
        with patch.object(d, "inspect_tools", fake_inspect):
            asyncio.run(d.work(Session(), reopened, args, {"sent": 0}))
        self.assertEqual(reopened.paid_attempts(), 10)
        self.assertEqual(reopened.status(), {"pending": 1})

    def test_catalog_discovery_has_no_company_filter_or_paid_jobs(self):
        seed = Path(self.tmp.name) / "catalog.json"
        seed.write_text(
            json.dumps(
                {
                    "data": [
                        {"tearsheet_name": "Clickstream"},
                        {"tearsheet_name": "Advertising"},
                    ]
                }
            )
        )
        for _ in range(2):
            self.assertEqual(d.prepare_catalog(self.q, seed), 2)
        jobs = self.q.db.execute("SELECT * FROM jobs").fetchall()
        self.assertEqual(len(jobs), 2)
        self.assertTrue(all(job["tool"] == "data_library" for job in jobs))
        self.assertTrue(
            all("entities" not in json.loads(job["args"]) for job in jobs)
        )
        d.build_children(
            self.q,
            jobs[0],
            {
                "structuredContent": {
                    "mode": "describe",
                    "data": {"history": "2018-Present"},
                }
            },
            "1900-01-01",
            "2026-10-24",
        )
        self.assertEqual(
            self.q.db.execute("SELECT count(*) FROM jobs").fetchone()[0], 2
        )
        saved = list((Path(self.tmp.name) / "datasets").glob("*.json"))
        self.assertEqual(len(saved), 1)
        self.assertEqual(
            json.loads(saved[0].read_text())["raw_download_status"],
            "not_downloaded",
        )

    def test_oauth_state_files_private_and_expiry_adjusts_across_restart(self):
        from mcp.shared.auth import OAuthToken

        store = d.FileTokens(Path(self.tmp.name) / "auth/token.json")
        with patch.object(d.time, "time", return_value=1000):
            asyncio.run(
                store.set_tokens(
                    OAuthToken(
                        access_token="fake-test-only",
                        token_type="Bearer",
                        expires_in=60,
                    )
                )
            )
        with patch.object(d.time, "time", return_value=1070):
            self.assertEqual(asyncio.run(store.get_tokens()).expires_in, 0)
        self.assertEqual(store.path.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
