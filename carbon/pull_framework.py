"""Carbon Arc framework puller: quote -> buy -> export.

The sanctioned data route for this account (Block/S3 is Enterprise-only,
see data/carbon/MANIFEST.md). A pull is a framework query: entities x
insight x filters. Quotes are free; purchases spend account credits.

Quote only (default, safe):
    python carbon/pull_framework.py --insight 236293 --representation ticker \\
        --name job_count --start 2015-01-01 --end 2026-08-31

Buy and download (spends credits):
    python carbon/pull_framework.py --insight 236293 --representation ticker \\
        --name job_count --start 2015-01-01 --end 2026-08-31 --buy

Output: data/carbon/raw/framework/<name>.parquet plus a MANIFEST.md entry.
Purchased framework IDs are cached in data/carbon/raw/framework/purchased.json
so re-runs download without re-buying.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "carbon"))

from client import load_key  # noqa: E402

from carbonarc import CarbonArcClient  # noqa: E402

RAW = ROOT / "data" / "carbon" / "raw" / "framework"
PURCHASED = RAW / "purchased.json"
MANIFEST = ROOT / "data" / "carbon" / "MANIFEST.md"


def client() -> CarbonArcClient:
    return CarbonArcClient(token=load_key())


def build(ex, args) -> dict:
    filters = {
        "date_resolution": args.resolution,
        "date_range": {"start_date": args.start, "end_date": args.end},
    }
    return ex.build_framework(
        entities=args.representation,
        insight=args.insight,
        filters=filters,
        aggregate=args.aggregate,
    )


def load_purchased() -> dict:
    if PURCHASED.exists():
        return json.loads(PURCHASED.read_text(encoding="utf-8"))
    return {}


def save_purchased(purchased: dict) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PURCHASED.write_text(json.dumps(purchased, indent=2), encoding="utf-8")


def manifest_note(name: str, framework_id: str, rows: int, price: float, args) -> None:
    entry = (
        f"\n## framework pull: {name}\n\n"
        f"| item | value |\n|---|---|\n"
        f"| framework_id | `{framework_id}` |\n"
        f"| insight | {args.insight} |\n"
        f"| representation | {args.representation} |\n"
        f"| window | {args.start} .. {args.end} ({args.resolution}) |\n"
        f"| rows | {rows} |\n"
        f"| price (credits) | {price} |\n"
        f"| file | `raw/framework/{name}.parquet` |\n"
    )
    with MANIFEST.open("a", encoding="utf-8") as fh:
        fh.write(entry)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--insight", type=int, required=True, help="insight id (from ontology search)")
    ap.add_argument("--representation", default="ticker", help="entity representation (ticker, company, retailer...)")
    ap.add_argument("--name", required=True, help="output name for the pull")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2026-08-31")
    ap.add_argument("--resolution", default="month", choices=["day", "week", "month", "quarter", "fiscalquarter"])
    ap.add_argument("--aggregate", default=None, choices=[None, "sum", "mean", "count_distinct"])
    ap.add_argument("--buy", action="store_true", help="purchase and download (spends credits)")
    args = ap.parse_args()

    ca = client()
    ex = ca.explorer
    purchased = load_purchased()

    cached = purchased.get(args.name)
    if cached:
        print(f"[cache] {args.name} already purchased as {cached['framework_id']} - downloading")
        df = ex.get_framework_data(cached["framework_id"])
    else:
        framework = build(ex, args)
        print("[quote] requesting price (free)...")
        try:
            q = ex.check_framework_quote(framework)
        except Exception as exc:
            print(f"  [error] quote failed: {str(exc)[:160]}")
            print(
                "  [hint] a 500 on /v2/framework/order masks 'insight blocked for user'\n"
                "         (audit 2026-10-24) - data entitlements are likely not\n"
                "         provisioned on this account yet. Check onboarding/support."
            )
            return 3
        print(f"  valid   : {q.get('valid')}")
        print(f"  rows    : {q.get('used_rows_count')}")
        print(f"  price   : {q.get('price')} credits")
        print(f"  columns : {q.get('columns')}")
        print(f"  message : {q.get('message')}")
        if not q.get("valid"):
            print("\n[stop] framework invalid or not entitled - nothing purchased")
            return 2
        if not args.buy:
            print("\n[quote only] re-run with --buy to purchase")
            return 0

        balance = ca.platform.get_balance() if hasattr(ca, "platform") else None
        price = float(q.get("price") or 0.0)
        print(f"[buy] purchasing {args.name} for {price} credits...")
        order = ex.buy_frameworks([framework])
        fid = order.get("framework_id") or (order.get("frameworks") or [{}])[0].get("id")
        print(f"  purchased: {fid}")
        purchased[args.name] = {"framework_id": fid, "price": price, "order": str(order)[:500]}
        save_purchased(purchased)
        df = ex.get_framework_data(fid)

    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"{args.name}.parquet"
    df.to_parquet(out, index=False)
    print(f"[done] {len(df):,} rows -> {out}")
    manifest_note(args.name, purchased.get(args.name, {}).get("framework_id", "?"), len(df), purchased.get(args.name, {}).get("price", 0.0), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
