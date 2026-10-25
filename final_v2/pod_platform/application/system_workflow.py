"""Finite host application: autonomous research publications become a coordinated book.

The fixture mode plants data, never admission or forecasts. Research, challenge,
publication, CIO sizing, risk, execution and accounting use their production APIs.
All resulting economic evidence remains explicitly synthetic and test scoped.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import random
from pathlib import Path
from typing import Any

from ..contracts.forecasts import Forecast
from ..contracts.decisions import DecisionEvent, ActorKind
from ..research.promotion import AdmissionCertificate
from ..research.memory import PodMemory
from ..orchestration.replay import ControlledReplay, load_fixture, coordination_hook_from_records, COORDINATION_MODE
from .common import UTC, SYNTHETIC_LABEL, atomic_write_json, read_json, digest, file_sha256, now_utc
from .stages import StageRunner
from .namespaces import RoleContext
from .publication import PublicationService
from .decision_store import DecisionStore
from .diagnostics import ResearchDiagnostics


class CompletePlatformWorkflow:
    """One restartable application run with bounded offline research by default."""

    def __init__(self, state_dir, output_dir, *, run_id="complete_fixture_v1", policy_path=None):
        self.state = Path(state_dir)
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.runner = StageRunner(self.state, run_id)
        self.run_id = run_id
        self.policy_path = Path(policy_path) if policy_path else Path(__file__).resolve().parents[2] / "configs/runtime/complete_fixture_policy.json"
        self.policy = read_json(self.policy_path)
        self.source_hashes = {str(p.relative_to(Path(__file__).resolve().parents[1])): file_sha256(p)
                              for p in sorted(Path(__file__).resolve().parents[1].rglob("*.py"))}
        self._delivered = {}
        self.loop = None

    def _receipt(self, **payload):
        return {**payload, "artifacts": {str(p.relative_to(self.output)): file_sha256(p)
                for p in sorted(self.output.rglob("*")) if p.is_file()}}

    def _stage(self, name, fn, inputs, outputs):
        return self.runner.run(name, fn, inputs={"output": str(self.output.resolve()), **inputs},
            source_hashes=self.source_hashes, artifacts_root=self.output, owned_outputs=outputs)

    def _on_publication(self, publication):
        """Idempotent transport receipt, preserving the exact backend publication."""
        key = publication.get("delivery_key") or publication["batch_id"]
        path = self.output / "deliveries" / (digest(key) + ".json")
        payload = dict(publication)
        if path.exists():
            if read_json(path) != payload:
                raise ValueError("publication delivery key was reused with changed content")
        else:
            atomic_write_json(path, payload)
        self._delivered[publication["batch_id"]] = payload
        return {"status": "received", "delivery_key": key, "path": str(path), "sha256": file_sha256(path)}

    def _on_cycle(self, cycle):
        # Research delivery acknowledgement precedes the ordered replay stage.
        return {"status": "research_received", "cycle_digest": digest(cycle),
                "consumer": "CompletePlatformWorkflow", "next_stage": "coordinated_replay"}

    def run_demo(self, *, max_jobs=16, max_iterations=4):
        from .research_loop import build_fixture_loop
        self.loop = build_fixture_loop(self.state / "research", on_publication=self._on_publication, on_cycle=self._on_cycle)
        result = self.loop.tick(max_jobs=max_jobs, max_iterations=max_iterations)
        publications = result.get("publications", [])
        # Receipts can already be committed by an earlier process; consume the
        # backend's returned validated publication set on every restart too.
        for publication in publications:
            self._on_publication(publication)
        if not publications:
            atomic_write_json(self.output / "blocked.json", {"status": "NO_VALID_PUBLICATIONS", "research": result})
            raise ValueError("research produced no valid publication; no trading fixture is authorized")

        def ingest():
            frozen = self._freeze_publications(publications)
            atomic_write_json(self.output / "research_manifest.json", {"label": SYNTHETIC_LABEL, "research": result,
                "publications": frozen, "llm_skill_evidence": False})
            return self._receipt(publications=len(frozen), research_digest=digest(result))
        research = self._stage("research_publications", ingest,
            {"publication_artifacts": self._publication_hashes(publications)}, ("published", "research_manifest.json"))

        def book():
            records = read_json(self.output / "research_manifest.json")["publications"]
            fixture_dir, hook = self._build_fixture(records)
            fixture = load_fixture(fixture_dir)
            outcome = ControlledReplay(fixture, self.output / "book", coordination=hook).run()
            store = DecisionStore(self.state / "decision_store.sqlite")
            store.ingest_replay_run(self.output / "book", run_id=self.run_id,
                                    forecasts=fixture.forecasts, certificates=fixture.certificates)
            self._research_events(store, records, fixture)
            examples = []
            for day, _, _ in fixture.periods:
                held = store.positions_on(day, self.run_id)
                if held:
                    examples.append(store.explain(held[0]["security_id"], day, self.run_id))
            atomic_write_json(self.output / "dossiers.json", examples)
            return self._receipt(book_summary=outcome["summary"])
        replay = self._stage("coordinated_replay", book,
            {"research": research, "policy": self.policy}, ("fixture", "risk_inputs.json", "book", "dossiers.json"))

        def feedback():
            queued = self._feedback()
            atomic_write_json(self.output / "feedback.json", queued)
            return self._receipt(queued=len(queued))
        feedback_result = self._stage("matured_feedback", feedback, {"replay": replay}, ("feedback.json",))
        summary = {"run_id": self.run_id, "mode": "complete_fixture", "label": SYNTHETIC_LABEL,
            "publication_count": research["publications"], "book": replay["book_summary"],
            "feedback_items": feedback_result["queued"], "llm_called": False, "llm_skill_evidence": False,
            "investment_validation": "UNPROVEN", "prospective_status": "NOT_PROSPECTIVE",
            "certificates_authorize_competition_book": False,
            "research_manifest": str(self.output / "research_manifest.json"),
            "dossiers": str(self.output / "dossiers.json"), "output_dir": str(self.output.resolve())}
        def finish():
            atomic_write_json(self.output / "complete_summary.json", summary)
            # Actual write time, distinct from the historical replay clock.
            atomic_write_json(self.output / "content_freeze.json", {"frozen_at": now_utc().isoformat(),
                "scope": "fixture_content_only", "prospective": False,
                "sha256": {str(p.relative_to(self.output)): file_sha256(p) for p in sorted(self.output.rglob("*"))
                    if p.is_file() and p.name != "content_freeze.json"}})
            return self._receipt(summary=summary)
        return self._stage("content_freeze", finish, {"feedback": feedback_result},
                           ("complete_summary.json", "content_freeze.json"))["summary"]

    def run_research(self, manifest_path, dataset_id, research_asof):
        """Official-shaped inputs exercise research gates without trading authority.

        This uses the registered charter campaign and the same integrity source
        classification. It intentionally stops before challenge/publication when
        the manifest cannot establish production source eligibility.
        """
        from ..strategies.charter_campaign import run_charter_campaign
        from ..research.builtins.manifest import load_manifest
        from .integrity import IntegrityPolicy, normalize_campaign, classify_research_candidate
        from .common import parse_aware
        manifest_path = Path(manifest_path)
        asof = parse_aware(research_asof) if isinstance(research_asof, str) else research_asof
        dataset = load_manifest(manifest_path).dataset(dataset_id)
        def research():
            campaign = run_charter_campaign(manifest_path, self.state / "official_campaign",
                                            dataset_id=dataset_id, research_asof=asof)
            policy = IntegrityPolicy(policy_id="complete_official_research_v1", frozen_at=asof)
            candidates = [classify_research_candidate(candidate, policy,
                          {dataset.spec.source_id: dataset.source_card}).as_dict()
                          for candidate in normalize_campaign(campaign)]
            atomic_write_json(self.output / "research_campaign.json", campaign)
            atomic_write_json(self.output / "research_candidates.json", candidates)
            summary = {"run_id": self.run_id, "mode": "research_only", "candidate_count": len(candidates),
                "candidate_outcomes": [c["outcome"] for c in candidates], "admitted": 0, "publications": 0,
                "orders": 0, "llm_called": False, "investment_validation": "UNPROVEN",
                "prospective_status": "NOT_PROSPECTIVE", "certificates_authorize_competition_book": False,
                "limitation": "Registered research/source classification only; independent frozen admission remains required."}
            atomic_write_json(self.output / "complete_summary.json", summary)
            return self._receipt(summary=summary)
        return self._stage("official_research", research, {"manifest": file_sha256(manifest_path),
            "dataset": file_sha256(dataset.path), "source_card": dataset.source_card_hash,
            "dataset_id": dataset_id, "research_asof": asof.isoformat()},
            ("research_campaign.json", "research_candidates.json", "complete_summary.json"))["summary"]

    @staticmethod
    def _publication_hashes(publications):
        result = {}
        for item in publications:
            for key, value in item.items():
                if key.endswith("_path") and value and Path(value).is_file():
                    result[f"{item['batch_id']}:{key}"] = file_sha256(value)
        return result

    def _freeze_publications(self, publications):
        result = []
        for publication in publications:
            item = dict(publication)
            folder = self.output / "published" / digest(item["batch_id"])
            folder.mkdir(parents=True, exist_ok=True)
            for key, value in list(item.items()):
                if key.endswith("_path") and value and Path(value).is_file():
                    dest = folder / (key + Path(value).suffix)
                    dest.write_bytes(Path(value).read_bytes())
                    item[key] = str(dest)
            result.append(item)
        return result

    def _build_fixture(self, publications):
        forecasts, certificates, cards = [], {}, {}
        for item in publications:
            raw = read_json(item["forecast_path"])
            raw = raw.get("forecasts", raw) if isinstance(raw, dict) else raw
            forecasts.extend(Forecast.model_validate(f) for f in raw)
            certificate = AdmissionCertificate.model_validate(read_json(item["certificate_path"]))
            if not certificate.verify_seal():
                raise ValueError("published certificate failed its seal")
            certificates[certificate.certificate_id] = certificate
            raw_cards = read_json(item.get("source_cards_path") or item["source_card_path"])
            if isinstance(raw_cards, dict):
                raw_cards = [raw_cards]
            for raw_card in raw_cards:
                cards[raw_card["source_id"]] = raw_card
        if len({f.strategy_id for f in forecasts}) < 2:
            raise ValueError("complete fixture requires at least two independently admitted pods")
        if not all(f.verify_seal() for f in forecasts):
            raise ValueError("backend publication forecast failed its seal")
        root = self.output / "fixture"
        root.mkdir(parents=True, exist_ok=True)
        policy = json.loads(json.dumps(self.policy))
        pods = sorted({f.strategy_id for f in forecasts})
        policy["hedge_requests"] += [{"requester": pod, "security_id": "HEDGE", "dimension": "beta", "max_abs_weight": .25}
                                       for pod in pods[:2]]
        # Family identity comes from the published mechanism; a source label
        # alone never joins unrelated mechanisms into one economic vote.
        policy["declarations"] = {pod: {"family_label": next(f.mechanism_ids[0] for f in forecasts if f.strategy_id == pod and f.mechanism_ids)}
                                  for pod in pods}
        formations = sorted({f.earliest_formation_at.date() for f in forecasts})
        names = sorted({f.security_id for f in forecasts})
        risk_rows = self._risk_history(names + ["HEDGE"], formations[0], formations[-1])
        atomic_write_json(self.output / "risk_inputs.json", risk_rows)
        hook = coordination_hook_from_records(policy, risk_rows)
        config = {"run_id": self.run_id, "run_kind": "replay", "starting_nav": "1000000",
            "certificate_policy": "test", "universe_id": "fixture_universe",
            "official_limits": {"min_positions": 4, "max_positions": 50, "max_gross": 2., "min_net": -.5,
                                "max_net": .5, "max_abs_name_weight": .3, "min_short_market_cap_usd": 0},
            "internal_policy": {"target_positions": 8, "sector_neutral": False, "beta_neutral": False, "capital_usd": 1000000},
            "admitted_families": pods, "neutral_uncertainty": .1,
            "cost_model": {"model_id": "complete_fixture_cost_v1", "half_spread_by_size_bp": {"large": 2.}, "eta": .1, "explicit_fee_bp": .5},
            "execution_policy": {"policy_id": "complete_fixture_execution_v1", "max_daily_participation": .2,
                "min_resize_fraction": 0., "no_trade_band": .0005, "min_economic_edge_bp": 0., "order_ttl_days": 5},
            "fill_window_days": 5, "core_sources": sorted(cards),
            "risk": {"policy_version": "complete_fixture_risk_v1", "max_gross": 2., "max_abs_net": .5, "max_name_weight": .3, "min_names": 4},
            "coordination": {"mode": COORDINATION_MODE, "policy_digest": hook.digest()}, "frozen_at": now_utc().isoformat()}
        atomic_write_json(root / "config.json", config)
        atomic_write_json(root / "forecasts.json", [f.model_dump(mode="json") for f in forecasts])
        atomic_write_json(root / "certificates.json", [c.model_dump(mode="json") for c in certificates.values()])
        atomic_write_json(root / "source_cards.json", list(cards.values()))
        atomic_write_json(root / "lineage.json", {"security_master_version": "complete_fixture_master_v1",
            "economic_graph_version": "published_mechanisms_v1", "calendar_policy": "fixture_monthly",
            "timezone_policy": "utc_close_2100", "data_access_policy_version": "runtime_protected_partitions_v1",
            "evaluation_policy_version": "runtime_frozen_independent_check_v1", "trial_count": 4, "evaluation_access_count": 0})
        fields = ["formation_date", "security_id", "ticker", "name", "label_source", "label_reference_date", "market_cap_usd",
                  "beta", "sector", "tradable", "price", "adv_dollars", "daily_volatility", "size_tier", "outcome_excess_return"]
        with (root / "market.csv").open("w", newline="") as handle:
            writer = csv.writer(handle); writer.writerow(fields)
            for date in formations:
                active = [f for f in forecasts if f.earliest_formation_at.date() == date]
                expected = {s: sum(float(f.return_forecast.expected_excess_return) for f in active if f.security_id == s) /
                            sum(f.security_id == s for f in active) for s in {f.security_id for f in active}}
                losing = max(expected, key=expected.get)
                for n, sid in enumerate(sorted(expected)):
                    er = expected[sid]
                    outcome = -.12 if sid == losing else er + ((n % 3) - 1) * .003
                    writer.writerow([date, sid, sid, f"Planted fixture {sid}", SYNTHETIC_LABEL, date, "1000000000",
                        1.3 if er > 0 else .7, "fixture", "true", 100, 50000000, .02, "large", outcome])
                writer.writerow([date, "HEDGE", "HEDGE", "Fixture shared beta hedge", SYNTHETIC_LABEL, date,
                                 "500000000000", 1., "index", "true", 100, 100000, .02, "large", .005])
        with (root / "periods.csv").open("w", newline="") as handle:
            writer = csv.writer(handle); writer.writerow(["formation_date", "period_end", "tb3ms_annual_percent"])
            for i, date in enumerate(formations):
                end = formations[i + 1] if i + 1 < len(formations) else date + dt.timedelta(days=30)
                writer.writerow([date, end, 3.])
        return root, hook

    @staticmethod
    def _risk_history(names, first, last):
        rng = random.Random(260825)
        rows = []
        day = first - dt.timedelta(days=180)
        while day <= last:
            market = rng.gauss(0, .01)
            rows.append({"observed_at": dt.datetime.combine(day, dt.time(20), UTC).isoformat(),
                         "returns": {s: .8 * market + rng.gauss(0, .007) for s in names}})
            day += dt.timedelta(days=1)
        return rows

    def _research_events(self, store, publications, fixture):
        """Persist upstream links alongside the replay's exact security-tagged chain."""
        by_cert = {p["certificate_id"]: p for p in publications}
        for forecast in fixture.forecasts:
            item = by_cert[forecast.admission_certificate_id]
            inputs = {key: file_sha256(value) for key, value in item.items()
                      if key.endswith("_path") and value and Path(value).is_file()}
            at = forecast.earliest_formation_at
            event = DecisionEvent(owner="reporting", created_at=now_utc(),
                event_id="upstream_" + forecast.forecast_id, actor_kind=ActorKind.CODE, actor_id="complete_platform",
                effective_research_at=at, effective_market_at=at, input_artifact_ids=sorted(inputs), input_hashes=inputs,
                chosen_action="consume backend validated source/method/hypothesis/experiment/check/publication chain",
                rationale="Exact immutable backend artifacts authorize this forecast; fixture scope only.",
                authorizing_rule_or_certificate_id=forecast.admission_certificate_id, resulting_artifact_id=forecast.forecast_id)
            store.record(event, security_ids=[forecast.security_id], run_id=self.run_id)

    def _feedback(self):
        """Queue one bounded diagnostic per owning pod from actual matured fills/PnL."""
        publications = read_json(self.output / "research_manifest.json")["publications"]
        receipts = read_json(self.output / "book/coordination_receipts.json")
        monthly = read_json(self.output / "book/monthly.json")
        periods = read_json(self.output / "book/periods.json")
        end = dt.date.fromisoformat(monthly[-1]["period_end"])
        at = dt.datetime.combine(end, dt.time(23, 59), UTC)
        result = []
        # The fixture payoff deliberately includes a losing held long. Attribute
        # its measured signed contribution to the pods in the actual CIO receipt.
        with (self.output / "book/holdings_by_period.csv").open() as handle:
            held = list(csv.DictReader(handle))
        with (self.output / "fixture/market.csv").open() as handle:
            market = {(r["formation_date"], r["security_id"]): r for r in csv.DictReader(handle)}
        owners = {p["namespace"] for p in publications}
        for namespace in sorted(owners):
            contributions = []
            for row in held:
                key = (row["period"], row["security_id"])
                m = market.get(key)
                if m:
                    contribution = float(row["post_trade_weight"]) * float(m["outcome_excess_return"])
                    contributions.append((contribution, row))
            if not contributions:
                continue
            contribution, worst = min(contributions, key=lambda pair: pair[0])
            if contribution >= 0:
                continue
            # Net return is the accounted all-cost book return, never an invented
            # attribution of costs or an IC inferred from one losing investment.
            metrics = {"net_return": float(monthly[-1]["net_return"])}
            memory = self.loop.memory(namespace) if self.loop is not None else PodMemory(self.state / "research/memory", namespace, clock=lambda: at)
            diagnostics = ResearchDiagnostics(memory, clock=lambda: at)
            diagnoses = diagnostics.classify(metrics)
            subject = f"{self.run_id}:{worst['period']}:{worst['security_id']}"
            diagnostics.record(subject, diagnoses, attempted="actual coordinated orders/fills and matured fixture payoff",
                outcome=f"measured position contribution {contribution:.12f}; cause unresolved", retry_justified=False)
            item = diagnostics.queue_item_for(diagnoses[0], subject_id=subject,
                decision_deadline=at + dt.timedelta(days=7), existing_evidence=[str(self.output / "book/monthly.json"),
                    str(self.output / "book/periods.json"), str(self.output / "book/coordination_receipts.json")],
                expected_resource_cost={"max_variants": "1", "max_executor_jobs": "1", "auto_retrain": "false"})
            entry = diagnostics.enqueue(item)
            from ..research.scheduler import ResearchScheduler
            from ..research.jobs import ResearchQueue
            from .research_loop import SCORES
            scheduler = ResearchScheduler(self.state / 'research/scheduler.sqlite',
                queue=ResearchQueue(self.state / 'research/queue.sqlite', clock=lambda: at),
                clock=lambda: at, memory_root=self.state / 'research/memory')
            trigger = scheduler.add_trigger(namespace, 'calibration', 'fund_feedback:' + subject,
                'Review measured combined-fund loss; investigate causes before selecting another experiment.',
                SCORES, line_id='fund_' + digest(subject)[:16], cohort_month=worst['period'][:7])
            scheduled = scheduler.tick()
            result.append({"namespace": namespace, "item_id": item.item_id, "memory_entry_id": entry.entry_id,
                "scheduler_trigger_id": trigger['trigger_id'], "scheduler_receipt": scheduled['pods'][namespace],
                "measured_position_contribution": contribution, "metrics": metrics,
                "diagnoses": [d.model_dump(mode="json") for d in diagnoses], "automatic_retraining": False})
        return result
