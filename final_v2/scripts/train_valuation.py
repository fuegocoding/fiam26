"""Trains the gap_micro valuation family in resumable stages.

Stage A (panel) is cached on disk; each later stage saves before the next
begins, so a lost run resumes from the last completed stage.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from pod_platform.signals import valuation as val  # noqa: E402
from pod_platform.signals.ibes_state import set_external_path  # noqa: E402

REPO = ROOT.parent
OUT = ROOT / "artifacts" / "valuation_work" / "gap_micro_best"
SMOKE = OUT / "_smoke_panel.parquet"


def main() -> None:
    set_external_path(str(REPO / "final" / "external_data" / "external.parquet"))
    OUT.mkdir(parents=True, exist_ok=True)

    if SMOKE.exists():
        panel = pd.read_parquet(SMOKE)
        print(f"[A] panel loaded from smoke cache: {len(panel):,} rows", flush=True)
    else:
        t = time.time()
        panel = val._build_panel(REPO / "final" / "data" / "chars_final_with_names.parquet", OUT)
        panel.to_parquet(SMOKE, index=False)
        print(f"[A] panel built: {len(panel):,} rows in {time.time()-t:.0f}s", flush=True)

    xs_path = OUT / val.XS_FILE
    if xs_path.exists():
        xs = pd.read_parquet(xs_path)
        print(f"[B] xs loaded from cache: {len(xs):,} rows", flush=True)
    else:
        t = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val._SHARED["device"] = "cpu"
            xs = val._cross_sectional(panel)
        xs.to_parquet(xs_path, index=False)
        print(f"[B] cross-sectional done: {len(xs):,} rows in {time.time()-t:.0f}s", flush=True)

    pooled_path = OUT / val.POOLED_FILE
    if pooled_path.exists():
        pooled = pd.read_parquet(pooled_path)
        print(f"[C] pooled loaded from cache: {len(pooled):,} rows", flush=True)
    else:
        t = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pooled, diagnostics = val._pooled(panel, xs)
        pooled.to_parquet(pooled_path, index=False)
        print(f"[C] pooled done: {len(pooled):,} rows in {time.time()-t:.0f}s", flush=True)
        (OUT / "_pooled_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

    selection = val._select_models(xs, pooled)
    if (OUT / "_pooled_diagnostics.json").exists():
        selection["pooled_diagnostics"] = json.loads((OUT / "_pooled_diagnostics.json").read_text())
    (OUT / val.SELECTION_FILE).write_text(json.dumps(selection, indent=2))
    print("[D] selection written:", flush=True)
    for year, fold in sorted(selection["folds"].items()):
        print(f"    {year}: {fold['best_pricing_model']}  mdae={min(fold['validation_mdae'].values()):.3f}", flush=True)
    print("VALUATION TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
