"""Trains the graph channel in resumable stages (faithful port pipeline).

business_text profiles -> monthly graph snapshots -> tokens -> walk-forward
ensemble (3 seeds). Each stage caches to disk and resumes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pod_platform.signals.graph import business_text, builder, dataset, train  # noqa: E402
from pod_platform.signals.ibes_state import set_external_path  # noqa: E402

REPO = ROOT.parent
OUT = ROOT / "artifacts" / "graph_work"
TEST_YEARS = tuple(range(2021, 2027))
SEEDS = (7, 19, 43)


def main() -> None:
    set_external_path(str(REPO / "external_data" / "external.parquet"))
    OUT.mkdir(parents=True, exist_ok=True)

    profiles = OUT / "text_profiles.parquet"
    t = time.time()
    business_text.build_profiles(
        REPO / "data" / "8k_20150101_20260831_identified.parquet", profiles
    )
    print(f"[1] profiles ready in {time.time()-t:.0f}s", flush=True)

    snapshots = OUT / "snapshots"
    snapshots.mkdir(exist_ok=True)
    t = time.time()
    builder.build_snapshots(
        REPO / "data" / "chars_final_with_names.parquet", profiles, snapshots
    )
    print(f"[2] snapshots ready in {time.time()-t:.0f}s", flush=True)

    t = time.time()
    dataset.build(OUT, REPO / "data" / "chars_final_with_names.parquet", snapshots)
    print(f"[3] dataset ready in {time.time()-t:.0f}s", flush=True)

    t = time.time()
    result = train.train(OUT, OUT, "cpu", years=TEST_YEARS, seeds=SEEDS)
    print(f"[4] training done in {time.time()-t:.0f}s -> {result}", flush=True)
    print("GRAPH TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
