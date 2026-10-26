"""Reconcile saved concentration curves and frozen signal choices against their ledger."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pod_platform.application.common import atomic_write_json, file_sha256
from pod_platform.application.configuration import config_path, load_config
from pod_platform.application.edge_research import source_digest
from pod_platform.research.edge_curve import mean_evidence, summarize_prefixes


def main():
    config = load_config()
    state = config_path(config["application"]["state_dir"])
    report_path = Path(
        json.loads((state / "latest_edge_scan.json").read_text())["report"]
    )
    report = json.loads(report_path.read_text())
    fingerprint = report["fingerprints"]
    if (
        "filings_sha256" in fingerprint
        and file_sha256(config_path(config["training_lab"]["filings_path"]))
        != fingerprint["filings_sha256"]
    ):
        raise ValueError("Filing research snapshot changed")
    if source_digest() != fingerprint["source_sha256"]:
        raise ValueError(
            "Exact source version differs from the saved diagnostic"
        )
    for path, key in (
        (config["training_lab"]["panel_path"], "panel_sha256"),
        (config["training_lab"]["daily_path"], "daily_sha256"),
        (config["edge_research"]["proposal_path"], "proposal_sha256"),
    ):
        if file_sha256(config_path(path)) != fingerprint[key]:
            raise ValueError("Input or proposal snapshot changed")
    output = report_path.parent
    prefixes = pd.read_parquet(output / "prefix_monthly_returns.parquet")
    saved_curves = pd.read_csv(output / "edge_curves.csv")
    largest_difference = 0.0
    information = {}
    for identity in report["plans"]:
        recomputed = summarize_prefixes(
            prefixes.loc[prefixes.strategy_id == identity]
        )
        saved = saved_curves.loc[
            saved_curves.strategy_id == identity
        ].reset_index(drop=True)
        for column in (
            "names",
            "months",
            "mean",
            "hac_se",
            "lower_95",
            "first_half_mean",
            "second_half_mean",
        ):
            if not np.allclose(
                recomputed[column],
                saved[column],
                rtol=0,
                atol=1e-10,
                equal_nan=True,
            ):
                raise ValueError(
                    f"Saved curve does not reconcile: {identity} {column}"
                )
            difference = np.abs(
                recomputed[column].to_numpy() - saved[column].to_numpy()
            )
            finite = difference[np.isfinite(difference)]
            if len(finite):
                largest_difference = max(
                    largest_difference, float(finite.max())
                )
        information[identity] = pd.read_csv(
            output / (identity.replace(":", "_") + "_rank_ic.csv"),
            parse_dates=["formation_date", "execution_date", "settlement_date"],
        )
    for pod, evidence in report["pods"].items():
        measured = []
        for choice in evidence["signal_selection_folds"]:
            cutoff = pd.Timestamp(choice["selection_asof"])
            if cutoff != pd.Timestamp(choice["year"] - 1, 12, 31):
                raise ValueError("Invalid frozen selection boundary")
            ranked = []
            for identity, table in information.items():
                if not identity.startswith(pod + ":"):
                    continue
                prior = mean_evidence(
                    table.loc[table.settlement_date <= cutoff, "rank_ic"]
                )
                ranked.append(
                    (
                        prior["lower_95"]
                        if prior["lower_95"] is not None
                        else -np.inf,
                        identity,
                    )
                )
            if max(ranked)[1] != choice["strategy_id"]:
                raise ValueError(
                    "Signal choice used information outside its declared prior window"
                )
            table = information[choice["strategy_id"]]
            measured.extend(
                table.loc[
                    table.execution_date.dt.year == choice["year"], "rank_ic"
                ].tolist()
            )
        actual = mean_evidence(measured)
        for key in ("mean", "hac_se", "lower_95"):
            if not np.isclose(
                actual[key], evidence["frozen_signal_rank_ic"][key], atol=1e-10
            ):
                raise ValueError("Frozen signal evidence does not reconcile")
        if not 1 <= len(evidence["owned_research_strategies"]) <= 3:
            raise ValueError(
                "Every pod must own one to three research strategies"
            )
    review = {
        "source_data_and_proposal_hashes_match": True,
        "saved_curves_reconcile": True,
        "all_frozen_signal_choices_use_only_prior_settled_labels": True,
        "one_to_three_owned_research_strategies_every_pod": True,
        "capital_assumption": None,
        "prefix_monthly_observations": len(prefixes),
        "curve_points": len(saved_curves),
        "maximum_curve_discrepancy": largest_difference,
        "alpha_or_production_admission_granted": False,
    }
    atomic_write_json(output / "diagnostic_review.json", review)
    print(json.dumps(review))


if __name__ == "__main__":
    main()
