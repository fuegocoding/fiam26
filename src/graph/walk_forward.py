"""Challenge-compliant walk-forward schedule, signal selection and out-of-sample evaluation.

Schedule (FIAM 2026 challenge, "Training Procedures"): an expanding training
window and a rolling two-year validation window, assigned by target return
month, refit annually. For each test year Y from 2021 through the last month:

- training:   target months 2015-02 .. (Y-3)-12   (expanding)
- validation: target months (Y-2)-01 .. (Y-1)-12  (rolling two years)
- test:       target months in year Y

A label formed at month-end t with horizon h covers target months t+1..t+h. It
belongs to training or validation only when all of its target months lie inside
that window, so labels realized in a later split never leak backwards. When
label dates are given (e.g. the announcement date of an analyst-surprise
label), a training row also needs its label to be public by the training end.
Validation rows likewise need their label public by the validation end. A row belongs to test
year Y when its first target month is in Y.

``validation_years=0`` is the user-requested prequential variant: training runs
through December of Y-1 and no validation window exists. It is a robustness
check only; the challenge specification is ``validation_years=2``.

Signal selection (``walk_forward``) for a monthly statistic series, fixed in advance:
- trade direction = sign of the training mean;
- keep if |training HAC t| >= 2 with >= 24 training labels and, when a validation
  window exists, the validation mean agrees in sign (>= 12 labels);
- kept candidates are traded in test year Y in the training direction.
Quintile legs are re-signed for reversed trades (buy Q1 = -q_short, short Q5 = -q_long).

    python -m src.graph.walk_forward --statistics path/to/monthly_statistics.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FIRST_TARGET_MONTH = pd.Timestamp("2015-02-28")
FIRST_TEST_YEAR = 2021
VALIDATION_YEARS = 2
KEYS = ["universe", "bucket", "horizon", "signal"]
PRIMARY_STATS = ("fm_factors_value", "q_spread")
MIN_TRAIN_LABELS = 24
MIN_VALIDATION_LABELS = 12
TRAIN_T = 2.0


def hac_mean(values, maxlags: int = 3) -> dict:
    """Mean with a Newey-West (Bartlett) t-statistic."""
    values = pd.Series(values).dropna().to_numpy(float)
    if not len(values):
        return {"mean": None, "hac_tstat": None, "observations": 0}
    residual = values - values.mean()
    long_run_variance = float(residual @ residual / len(values))
    for lag in range(1, min(maxlags, len(values) - 1) + 1):
        covariance = float(residual[lag:] @ residual[:-lag] / len(values))
        long_run_variance += 2 * (1 - lag / (maxlags + 1)) * covariance
    standard_error = np.sqrt(max(long_run_variance, 0) / len(values))
    return {
        "mean": float(values.mean()),
        "hac_tstat": float(values.mean() / standard_error) if standard_error > 0 else None,
        "observations": len(values),
    }


def split_masks(formation_months, horizon: int, test_year: int, label_dates=None,
                validation_years: int = VALIDATION_YEARS) -> dict[str, np.ndarray]:
    """Boolean masks over formation months for one test year, assigned by target month."""
    if validation_years < 0:
        raise ValueError("validation_years must be non-negative")
    months = pd.DatetimeIndex(formation_months)
    first_target = months + pd.offsets.MonthEnd(1)
    last_target = months + pd.offsets.MonthEnd(horizon)
    train_end = pd.Timestamp(year=test_year - validation_years - 1, month=12, day=31)
    announced = pd.DatetimeIndex(label_dates) if label_dates is not None else None
    train = np.asarray((first_target >= FIRST_TARGET_MONTH) & (last_target <= train_end))
    if announced is not None:
        train &= np.asarray(announced.notna() & (announced <= train_end))
    if validation_years:
        validation_start = pd.Timestamp(year=test_year - validation_years, month=1, day=31)
        validation_end = pd.Timestamp(year=test_year - 1, month=12, day=31)
        validation = np.asarray((first_target >= validation_start) & (last_target <= validation_end))
        if announced is not None:
            validation &= np.asarray(announced.notna() & (announced <= validation_end))
    else:
        validation = np.zeros(len(months), dtype=bool)
    return {"train": train, "validation": validation, "test": np.asarray(first_target.year == test_year)}


def primary_series(statistics: pd.DataFrame) -> pd.DataFrame:
    """Most conservative statistic per candidate: factor+value-controlled slope, else the quintile spread."""
    available = statistics[[*KEYS, "stat"]].drop_duplicates()
    has_fm = available[available.stat == PRIMARY_STATS[0]][KEYS].assign(has_fm=True)
    chosen = available.merge(has_fm, on=KEYS, how="left")
    keep = (chosen.stat == PRIMARY_STATS[0]) | ((chosen.stat == PRIMARY_STATS[1]) & chosen.has_fm.isna())
    return statistics.merge(chosen.loc[keep, [*KEYS, "stat"]], on=[*KEYS, "stat"], how="inner")


def _column(frame: pd.DataFrame, name: str) -> np.ndarray:
    return frame[name].to_numpy(dtype=float) if name in frame else np.full(len(frame), np.nan)


def walk_forward(statistics: pd.DataFrame, last_year: int,
                 validation_years: int = VALIDATION_YEARS) -> tuple[pd.DataFrame, pd.DataFrame]:
    statistics = statistics.assign(eom=pd.to_datetime(statistics.eom))
    primary = primary_series(statistics)
    legs = statistics[statistics.stat.isin(["q_long", "q_short", "q_spread"])].pivot_table(
        index=[*KEYS, "eom"], columns="stat", values="value")
    selections, oos = [], []
    for key, group in primary.groupby(KEYS, sort=True):
        horizon = int(key[2])
        series = group.set_index("eom").value.sort_index()
        stat = group.stat.iloc[0]
        for year in range(FIRST_TEST_YEAR, last_year + 1):
            masks = split_masks(series.index, horizon, year, validation_years=validation_years)
            train = hac_mean(series[masks["train"]], maxlags=horizon)
            validation = series[masks["validation"]]
            direction = float(np.sign(train["mean"])) if train["mean"] is not None else 0.0
            validation_agrees = (not validation_years) or (
                len(validation) >= MIN_VALIDATION_LABELS and direction * validation.mean() > 0)
            selected = bool(
                train["observations"] >= MIN_TRAIN_LABELS
                and train["hac_tstat"] is not None and abs(train["hac_tstat"]) >= TRAIN_T
                and validation_agrees
            )
            selections.append({
                **dict(zip(KEYS, key)), "stat": stat, "test_year": year, "direction": direction,
                "train_mean": train["mean"], "train_t": train["hac_tstat"], "train_labels": train["observations"],
                "validation_mean": validation.mean() if len(validation) else np.nan,
                "validation_labels": len(validation), "selected": selected,
            })
            test = series[masks["test"]]
            if not selected or test.empty:
                continue
            index = pd.MultiIndex.from_tuples([(*key, month) for month in test.index], names=[*KEYS, "eom"])
            leg_values = legs.reindex(index)
            frame = pd.DataFrame({**dict(zip(KEYS, key)), "stat": stat, "test_year": year, "direction": direction,
                                  "eom": test.index, "value": direction * test.to_numpy()})
            if direction > 0:
                frame["long_leg"], frame["short_leg"] = _column(leg_values, "q_long"), _column(leg_values, "q_short")
            else:
                frame["long_leg"], frame["short_leg"] = -_column(leg_values, "q_short"), -_column(leg_values, "q_long")
            frame["spread"] = direction * _column(leg_values, "q_spread")
            oos.append(frame)
    return pd.DataFrame(selections), (pd.concat(oos, ignore_index=True) if oos else pd.DataFrame())


def summarize_oos(selections: pd.DataFrame, oos: pd.DataFrame) -> dict[str, pd.DataFrame]:
    per_year = selections.groupby("test_year").agg(candidates=("selected", "size"), selected=("selected", "sum"))
    if oos.empty:
        return {"per_year": per_year}
    measures = ["value", "long_leg", "short_leg", "spread"]
    rows = []
    for key, group in oos.groupby(KEYS):
        series = group.set_index("eom").sort_index()
        row = {**dict(zip(KEYS, key)), "years_selected": group.test_year.nunique(), "months": int(series.value.notna().sum())}
        for column in measures:
            s = hac_mean(series[column].dropna(), maxlags=int(key[2]))
            row[f"{column}_mean"], row[f"{column}_t"] = s["mean"], s["hac_tstat"]
        rows.append(row)
    candidates = pd.DataFrame(rows).sort_values("value_t", ascending=False)

    strategy_rows = []
    for (universe, horizon), group in oos.groupby(["universe", "horizon"]):
        monthly = group.groupby("eom")[measures].mean().sort_index()
        row = {"universe": universe, "horizon": horizon, "months": len(monthly),
               "avg_selected_per_month": group.groupby("eom").size().mean()}
        for column in measures:
            s = hac_mean(monthly[column].dropna(), maxlags=int(horizon))
            row[f"{column}_mean"], row[f"{column}_t"] = s["mean"], s["hac_tstat"]
        strategy_rows.append(row)
    return {"per_year": per_year, "candidates": candidates, "strategy": pd.DataFrame(strategy_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--statistics", type=Path, required=True,
                        help="parquet with universe, bucket, horizon, signal, stat, eom, value")
    parser.add_argument("--validation-years", type=int, choices=(0, 2), default=VALIDATION_YEARS)
    args = parser.parse_args()
    statistics = pd.read_parquet(args.statistics)
    last_year = int(pd.to_datetime(statistics.eom).max().year)
    selections, oos = walk_forward(statistics, last_year, args.validation_years)
    output = args.statistics.parent
    selections.to_csv(output / "selections.csv", index=False)
    if not oos.empty:
        oos.to_parquet(output / "oos_series.parquet", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 200, "display.max_columns", 30):
        for name, table in summarize_oos(selections, oos).items():
            table.to_csv(output / f"oos_{name}.csv", index=name == "per_year")
            print(f"\n== {name}")
            print(table.round(2).to_string())


if __name__ == "__main__":
    main()
