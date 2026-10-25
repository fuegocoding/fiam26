import unittest
from types import SimpleNamespace

from check_access import authorized_skus, probe


class AccessTests(unittest.TestCase):
    def test_only_active_and_unblocked_skus_are_selected(self):
        response = {
            "datasets": [
                {
                    "dataset_id": "CA0030",
                    "platform_access_blocked": None,
                    "cuts": [
                        {
                            "cut": "core",
                            "request_statuses": {
                                "6m": "APPROVED",
                                "1y": "PENDING",
                                "0d": "NONE",
                            },
                        }
                    ],
                },
                {
                    "dataset_id": "CA0056",
                    "platform_access_blocked": True,
                    "cuts": [
                        {
                            "cut": "core",
                            "request_statuses": {"6m": "CONTRACTED"},
                        }
                    ],
                },
            ]
        }
        selected = authorized_skus(response)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["dataset_id"], "CA0030")
        self.assertEqual(selected[0]["lag"], "6m")

    def test_probe_uses_only_read_only_discovery_calls(self):
        calls = []

        def library():
            calls.append("get_datasets")
            return {"datasources": [], "total_pages": 1}

        def block():
            calls.append("list_datasets")
            return {"datasets": []}

        client = SimpleNamespace(
            data=SimpleNamespace(get_datasets=library),
            block=SimpleNamespace(list_datasets=block),
        )
        report, catalog = probe(client)
        self.assertEqual(calls, ["get_datasets", "list_datasets"])
        self.assertEqual(report["state"], "no_active_bulk_entitlements")
        self.assertEqual(report["purchases_or_access_requests_submitted"], 0)

    def test_server_errors_do_not_leak_exception_text(self):
        def library():
            raise RuntimeError("sensitive-test-string")

        report, _ = probe(
            SimpleNamespace(data=SimpleNamespace(get_datasets=library))
        )
        self.assertNotIn("sensitive-test-string", str(report))
        self.assertEqual(report["state"], "platform_api_error")

    def test_enterprise_restriction_is_reported_as_account_access_failure(self):
        class Denied(Exception):
            response = SimpleNamespace(
                status_code=403,
                json=lambda: {
                    "detail": "Block Data is available on Enterprise plans only."
                },
            )

        def block():
            raise Denied("not logged")

        client = SimpleNamespace(
            data=SimpleNamespace(get_datasets=lambda: {"datasources": []}),
            block=SimpleNamespace(list_datasets=block),
        )
        report, _ = probe(client)
        self.assertEqual(report["state"], "block_enterprise_plan_required")
        self.assertEqual(report["http_status"], 403)


if __name__ == "__main__":
    unittest.main()
