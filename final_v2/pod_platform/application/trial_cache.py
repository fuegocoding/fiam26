"""Restore trusted evaluated agent artifacts only on the exact evaluated prefix."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import textwrap

import numpy as np
import pandas as pd

from ..contracts.base import canonical_digest
from ..research.strategy_code import CodeBatch, CodeStrategy, required_features
from ..research.training_tactical import monthly_net, unit_return_month_ends
from .common import atomic_write_json, file_sha256


def _read(path):
    return json.loads(Path(path).read_text())


def _function(source, name):
    tree = ast.parse(textwrap.dedent(source))
    return ast.dump(next(node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name), include_attributes=False)


def _evaluation_hashes(snapshot, fitted, alternative=False):
    root = Path(__file__).parents[1]
    modules = ("strategy_code strategy_scope forecast_construction pod_plans score_worker "
        "training_data training_tactical training_execution training_search edge_capacity edge_curve "
        "predictive_selection predictive_calibration campaign_metrics desk_metrics decision_filings "
        "filing_access filing_text research_hypotheses input_permissions").split()
    if fitted:
        modules += ["forecast_labels", "causal_forecast", "causal_tree_forecast"]
    hashes = {}
    for name in modules:
        relative = Path("research") / (name + ".py")
        old, current = snapshot / relative, root / relative
        if not old.is_file() or file_sha256(old) != file_sha256(current):
            raise ValueError(f"Cached evaluation module changed: {relative}")
        hashes[str(relative)] = file_sha256(current)
    if alternative:
        for name in ("campaign_sources", "feature_documentation"):
            relative = Path("application" if name == "campaign_sources" else "research") / (name + ".py")
            if file_sha256(snapshot / relative) != file_sha256(root / relative):
                raise ValueError(f"Cached alternative feature evaluator changed: {relative}")
            hashes[str(relative)] = file_sha256(root / relative)
    if not fitted:
        relative = Path("research/forecast_labels.py")
        old = _function((snapshot / relative).read_text(), "prediction_horizon")
        current = _function((root / relative).read_text(), "prediction_horizon")
        if old != current:
            raise ValueError("Cached forecast horizon policy changed")
        hashes[str(relative) + ":prediction_horizon"] = canonical_digest({"ast": current})
    # The evaluator retains only predictive_selection from this review. The
    # funding_supported feedback field may change without changing that evidence.
    def review_ast(path):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                keep = [i for i, key in enumerate(node.keys)
                    if not isinstance(key, ast.Constant) or key.value != "funding_supported"]
                node.keys, node.values = [node.keys[i] for i in keep], [node.values[i] for i in keep]
        return ast.dump(tree, include_attributes=False)
    relative = Path("research/candidate_review.py")
    old, current = review_ast(snapshot / relative), review_ast(root / relative)
    if old != current:
        raise ValueError("Cached candidate evidence evaluator changed")
    hashes[str(relative) + ":evidence_ast"] = canonical_digest({"ast": current})
    return hashes


def restore_evaluated_trials(campaign, source: Path):
    """Import exact-prefix results; mismatches require a normal fresh evaluation."""
    source = Path(source).resolve()
    experiment_bytes = (source / "experiment.json").read_bytes()
    experiment = json.loads(experiment_bytes)
    ledger_bytes = (source / "attempt_ledger.json").read_bytes()
    ledger = json.loads(ledger_bytes)
    if experiment["settings"] != campaign.settings:
        raise ValueError("Cached trials require identical settings and raw data prefix")
    defaults = {"align_forecast_horizons": True, "robustness_suite": "expanded",
        "issuer_validation_folds": 2, "funding_method": "predictive",
        "portfolio_evidence_scope": "prior_settled", "evaluation_period": None,
        "alternative_bindings": []}
    for key in ("minimum_edge_months", "edge_uncertainty_haircut", *defaults):
        if experiment["policy"].get(key, defaults.get(key)) != campaign.policy.get(key, defaults.get(key)):
            raise ValueError(f"Cached trial policy mismatch: {key}")
    input_hashes = {}
    for name in ("panel", "daily", "filings"):
        if name == "filings" and not campaign.settings.get("filings_research_enabled"):
            continue
        actual = file_sha256(campaign.settings[name + "_path"])
        if actual != experiment.get(name + "_sha256"):
            raise ValueError(f"Cached trial input changed: {name}")
        input_hashes[name] = actual
    snapshot = source / "source_snapshot/pod_platform"
    old = _function((snapshot / "application/research_campaign.py").read_text(), "_evaluate")
    current = _function(inspect.getsource(campaign._evaluate), "_evaluate")
    if old != current:
        raise ValueError("Cached _evaluate implementation changed")
    for name in ("_known_outcomes", "_first_session_reviews"):
        if _function((snapshot / "application/research_campaign.py").read_text(), name) != _function(inspect.getsource(getattr(campaign, name)), name):
            raise ValueError(f"Cached evaluation helper changed: {name}")
    code_hashes = _evaluation_hashes(snapshot, False, bool(campaign.policy.get("alternative_bindings")))
    expected_order = [int(issuer) for issuer in campaign.panel.securities]
    order_path = source / "security_order.json"
    if order_path.is_file():
        if experiment.get("security_order_sha256") and file_sha256(order_path) != experiment["security_order_sha256"]:
            raise ValueError("Cached source security order receipt changed")
        order = _read(order_path)
        order = order.get("securities") if isinstance(order, dict) else order
    else:
        candidates = []
        for path in source.parent.glob("*_campaign_daily_*.parquet.identity.json"):
            identity = _read(path).get("identity", {})
            if (identity.get("source_sha256") == input_hashes["daily"]
                and identity.get("source") == str(Path(campaign.settings["daily_path"]).resolve())
                and identity.get("start") == pd.Timestamp(campaign.settings.get("daily_warmup_start", "2014-01-01")).isoformat()
                and identity.get("end") == pd.Timestamp(campaign.settings["end"]).isoformat()
                and identity.get("compact") == campaign.settings.get("compact_storage", False)):
                candidates.append((path, identity["securities"]))
        if not candidates or any(order != candidates[0][1] for _, order in candidates):
            raise ValueError("Cached trials require a verified source security order")
        order_path, order = candidates[0]
    if order != expected_order:
        raise ValueError("Cached trial security order differs from the current cohort")
    order_output = Path(campaign.output) / "security_order.json"
    if order_output.is_file():
        current_order = _read(order_output)
        if (current_order.get("securities") if isinstance(current_order, dict) else current_order) != expected_order:
            raise ValueError("Current output security order changed")
    else:
        atomic_write_json(order_output, expected_order)
    receipt = {"source": str(source), "experiment_sha256": hashlib.sha256(experiment_bytes).hexdigest(),
        "ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(), "input_sha256": input_hashes,
        "evaluate_ast_sha256": canonical_digest({"ast": current}), "evaluation_modules": code_hashes,
        "securities_sha256": canonical_digest({"securities": order}),
        "source_security_order": {"path": str(order_path), "sha256": file_sha256(order_path)},
        "imported_strategy_ids": [], "excluded_strategy_ids": [],
        "rejected_trials": [], "artifacts": {}}
    responses = []
    paths = set(source.rglob("*_response.json")) | {Path(p) for p in experiment.get("recorded_code_proposal_hashes", {})}
    for path in sorted(paths):
        if path.is_file():
            payload = _read(path)
            responses.append((path, payload))
    output = Path(campaign.output)
    bundle = output / "cache_import_sources" / receipt["experiment_sha256"][:16]
    feature_digests = {}
    allowed = campaign.policy.get("recorded_strategy_ids")
    for original in ledger:
        if original.get("status") != "evaluated" or original.get("strategy_id") in campaign.trials:
            continue
        identity = original.get("strategy_id")
        if allowed is not None and identity not in allowed:
            receipt["excluded_strategy_ids"].append(identity)
            continue
        try:
            strategy = CodeStrategy.model_validate(original["strategy"])
            if original.get("origin") not in {"agent_written_code", "recorded_agent_code"}:
                raise ValueError("Cached trial lacks agent origin")
            expected = strategy.pod + ":" + strategy.name + ":" + canonical_digest(strategy.model_dump(mode="json"))[:10]
            if identity != expected:
                raise ValueError("Cached strategy identity changed")
            if strategy.forecast_model != "code":
                _evaluation_hashes(snapshot, True, bool(campaign.policy.get("alternative_bindings")))
            feature_paths = []
            features = ((set(required_features(strategy.code)) - {"model_forecast"})
                | set(strategy.model_features) | {condition.feature for condition in strategy.universe.filters})
            for name in sorted(features):
                path = source / "feature_arrays" / (name + ".npy")
                if name not in campaign.panel.features or not path.is_file():
                    raise ValueError(f"Cached consumed feature is unavailable: {name}")
                if name not in feature_digests:
                    values = np.load(path, mmap_mode="r", allow_pickle=False)
                    current_values = np.asarray(campaign.panel.features[name])
                    digest = hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()
                    if (values.shape != current_values.shape or values.dtype != current_values.dtype
                        or digest != hashlib.sha256(np.ascontiguousarray(current_values).view(np.uint8)).hexdigest()):
                        raise ValueError(f"Cached consumed feature values changed: {name}")
                    feature_digests[name] = digest
                feature_paths.append(path)
            provenance = original.get("model_provenance") or {}
            if not all(provenance.get(k) for k in ("provider", "model", "request_hash", "response_hash")):
                raise ValueError("Cached trial lacks model provenance")
            response = next(((p, r) for p, r in responses if all(r.get(k) == provenance[k]
                for k in ("provider", "model", "request_hash", "response_hash"))
                and (not provenance.get("source_response_sha256") or file_sha256(p) == provenance["source_response_sha256"])), None)
            if response is None:
                raise ValueError("Cached model response receipt is missing")
            response_path, payload = response
            batch, _ = CodeBatch.validate_partial(payload["text"])
            if batch is None or not any(s.model_dump(mode="json") == strategy.model_dump(mode="json") for s in batch.strategies):
                raise ValueError("Cached strategy is absent from its actual model response")
            request = response_path.with_name(response_path.name.replace("_response.json", "_request.json"))
            if not request.is_file() or _read(request).get("request_hash") != provenance["request_hash"]:
                raise ValueError("Cached model request receipt is missing")
            stem = identity.replace(":", "_")
            record = _read(source / (stem + ".json"))
            if record != original or (source / (stem + ".py")).read_text() != strategy.code:
                raise ValueError("Cached evaluated record or code changed")
            if record.get("cache_import_receipt", {}).get("securities_sha256", receipt["securities_sha256"]) != receipt["securities_sha256"]:
                raise ValueError("Previously imported security order changed")
            for name, digest in record.get("cache_import_receipt", {}).get("artifacts", {}).items():
                if name != stem + ".json" and file_sha256(source / name) != digest:
                    raise ValueError("Previously imported cached artifact changed")
            unit = pd.read_parquet(source / (stem + "_unit_returns.parquet"))
            if not pd.DatetimeIndex(unit.date).equals(campaign.market.dates) or not np.isfinite(unit.returns).all():
                raise ValueError("Cached unit returns do not match the full market prefix")
            raw_formations = _read(source / (stem + "_native_forecasts.json"))
            formations = {int(day): value for day, value in raw_formations.items()}
            if set(raw_formations) != {str(day) for day in formations}:
                raise ValueError("Cached forecast day keys are not canonical")
            if any(not isinstance(t, int) or not 0 <= day < len(campaign.market.dates)
                or not 0 <= t < len(campaign.panel.dates) or campaign.panel.dates[t] > campaign.market.dates[day]
                for day, t in formations.items()):
                raise ValueError("Cached forecast formations are not causal")
            maps = {}
            for suffix, dates_name, days in (("schedule", "schedule_dates", None),
                ("native_forecasts", None, sorted(formations)), ("cio_prior_holdings", "cio_prior_dates", None)):
                if days is None:
                    dates = pd.DatetimeIndex(_read(source / (stem + "_" + dates_name + ".json")))
                    days = campaign.market.dates.get_indexer(dates).tolist()
                array = np.load(source / (stem + "_" + suffix + ".npy"), mmap_mode="r", allow_pickle=False)
                if array.shape != (len(days), len(campaign.panel.securities)) or len(set(days)) != len(days) or days != sorted(days) or any(d < 0 or d >= len(campaign.market.dates) for d in days):
                    raise ValueError("Cached array dimensions or observed decision dates changed")
                if np.isinf(array).any() or (suffix != "native_forecasts" and not np.isfinite(array).all()):
                    raise ValueError("Cached target or forecast array is invalid")
                maps[suffix] = (days, array)
            if maps["schedule"][0] != sorted(formations):
                raise ValueError("Cached schedule and native forecast days differ")
            report = record["evidence"]
            inception = pd.Timestamp(report["strategy_evidence_inception"])
            if inception not in campaign.market.dates:
                raise ValueError("Cached forecast inception is outside the market prefix")
            predictive = pd.read_parquet(source / (stem + "_predictive_monthly.parquet"))
            calibration = pd.read_parquet(source / (stem + "_predictive_calibration.parquet"))
            for frame in (predictive, calibration):
                if pd.to_datetime(frame.settlement_date).max() > pd.Timestamp(campaign.settings["end"]):
                    raise ValueError("Cached predictive evidence extends beyond the evaluated prefix")
            variant = strategy.model_dump(mode="json")
            for key in ("name", "mechanism", "falsification", "evidence_refs", "research_topic", "parent_strategy_id"):
                variant.pop(key)
            variant["universe"].pop("rationale")
            variant["code"] = ast.dump(ast.parse(strategy.code), include_attributes=False)
            variant_id = canonical_digest(variant)
            if any(t.get("variant_id") == variant_id for t in campaign.trials.values()):
                continue
            files = [*source.glob(stem + ".*"), *source.glob(stem + "_*")]
            hashes = {p.name: file_sha256(p) for p in files if p.is_file()}
            for path in files:
                if path.is_file() and path.resolve() != (output / path.name).resolve():
                    shutil.copy2(path, output / path.name)
            for path in feature_paths:
                target = output / "feature_arrays" / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(path, target)
                hashes[str(target.relative_to(output))] = file_sha256(target)
            response_bundle = bundle / file_sha256(response_path)[:16]
            response_bundle.mkdir(parents=True, exist_ok=True)
            for path in (request, response_path):
                target = response_bundle / path.name
                shutil.copy2(path, target)
                hashes[str(target.relative_to(output))] = file_sha256(target)
            record = copy.deepcopy(record)
            record.update(origin="recorded_agent_code", cache_import_receipt={"source": str(source),
                "experiment_sha256": receipt["experiment_sha256"],
                "securities_sha256": receipt["securities_sha256"], "artifacts": hashes})
            campaign.attempts.append(record)
            atomic_write_json(output / (stem + ".json"), record)
            loaded = {}
            for suffix, (days, _) in maps.items():
                values = np.load(output / (stem + "_" + suffix + ".npy"), mmap_mode="r", allow_pickle=False)
                loaded[suffix] = {day: values[row] for row, day in enumerate(days)}
            outcomes = unit_return_month_ends(campaign.panel)
            campaign.trials[identity] = {"variant_id": variant_id, "strategy": strategy, "origin": "recorded_agent_code",
                "schedule": loaded["schedule"], "native_forecasts": loaded["native_forecasts"], "native_formations": formations,
                "edge": {"returns": unit.returns.to_numpy(), "holdings": None, "prior_holdings": loaded["cio_prior_holdings"]},
                "monthly": monthly_net(unit.returns.to_numpy(), campaign.market, campaign.panel), "report": report,
                "predictive_monthly": predictive, "predictive_calibration": calibration,
                "evidence_available": outcomes >= inception, "outcome_dates": outcomes,
                "robustness_suite": campaign.robustness_suite, "funding_method": campaign.funding_method}
            receipt["imported_strategy_ids"].append(identity)
            receipt["artifacts"][identity] = hashes
        except (ValueError, KeyError, TypeError, OSError, StopIteration) as exc:
            receipt["rejected_trials"].append({"strategy_id": identity, "reason": str(exc)})
    campaign.cache_imports = [*getattr(campaign, "cache_imports", []), receipt]
    receipt["consumed_feature_sha256"] = feature_digests
    campaign._save()
    return receipt
