"""Valuation models of log market equity over assets, selected per test year."""

import json
import multiprocessing
import pathlib
import pickle
import tempfile
import time
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

from . import ibes_state as ibes

XS_FILE = "xs_predictions.parquet"
POOLED_FILE = "pooled_predictions.parquet"
SELECTION_FILE = "selection.json"
MODELS = ["M0", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]
ENSEMBLE = ["M2", "M4", "M5", "M6", "M7", "M8"]
POOLED_MODELS = ("M6", "M8", "M9")

FIRST_MONTH = pd.Timestamp("2018-01-31")
N_FOLDS = 5
RIDGE_LAMBDA = 10.0
XS_TREES = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "objective": "reg:absoluteerror",
    "tree_method": "hist",
    "nthread": 2,
    "verbosity": 0,
}
POOLED_TREES = {
    "n_estimators": 1500,
    "max_depth": 7,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "objective": "reg:absoluteerror",
    "tree_method": "hist",
    "verbosity": 0,
}
XS_WORKERS = 6
POOLED_WORKERS = 3
MOE_EXPERTS = 4
MOE_ITERATIONS = 40
MOE_GATE_STEPS = 30
MOE_MAX_ROWS = 250_000
MOE_MIN_VARIANCE = 0.01
TEST_YEARS = list(range(2021, 2027))

LEVELS = {
    "book": "be_me",
    "net_income": "ni_me",
    "sales": "sale_me",
    "ocf": "ocf_me",
    "fcf": "fcf_me",
    "rnd": "rd_me",
    "dividends": "div12m_me",
    "net_payout": "eqnpo_me",
    "debt": "debt_me",
    "net_debt": "netdebt_me",
}
RATIOS = [
    "gp_at",
    "ebit_sale",
    "ni_be",
    "ope_be",
    "op_at",
    "cop_at",
    "ocf_at",
    "niq_at",
    "niq_be",
    "sale_gr1",
    "sale_gr3",
    "saleq_gr1",
    "at_gr1",
    "be_gr1a",
    "capx_gr1",
    "emp_gr1",
    "sale_emp_gr1",
    "niq_su",
    "saleq_su",
    "ni_inc8q",
    "rd_sale",
    "rd5_at",
    "at_turnover",
    "tangibility",
    "cash_at",
    "noa_at",
    "aliq_at",
    "at_be",
    "earnings_variability",
    "ni_ar1",
    "ni_ivol",
    "ocfq_saleq_std",
    "f_score",
    "o_score",
    "qmj_prof",
    "qmj_growth",
    "age",
    "dsale_dinv",
    "dsale_drec",
    "dgp_dsale",
    "dsale_dsga",
    "oaccruals_at",
    "taccruals_at",
    "netis_at",
    "eqnetis_at",
    "dbnetis_at",
    "niq_at_chg1",
    "niq_be_chg1",
    "ocf_at_chg1",
    "inv_gr1",
    "noa_gr1a",
    "ppeinv_gr1a",
    "capex_abn",
    "debt_gr3",
    "ebit_bev",
    "sale_bev",
    "opex_at",
    "pi_nix",
    "cop_atl1",
    "gp_atl1",
    "op_atl1",
    "ope_bel1",
]
LEVEL_FEATURES = [f"{name}_lvl_at" for name in LEVELS] + [
    "ebitda_lvl_at",
    "book_ev_lvl_at",
    "iv_lvl_at",
    "fwd_earn_lvl_at",
]
FLAGS = ["has_analyst", "loss"]
FEATURES = LEVEL_FEATURES + ["analyst_growth", "log_at"] + RATIOS + FLAGS
GATE_TRAITS = [
    "net_income_lvl_at",
    "loss",
    "op_at",
    "gp_at",
    "sale_gr1",
    "sale_gr3",
    "analyst_growth",
    "rd_sale",
    "tangibility",
    "at_be",
    "debt_lvl_at",
    "cash_at",
    "earnings_variability",
    "log_at",
    "dividends_lvl_at",
    "fwd_earn_lvl_at",
    "book_lvl_at",
    "has_analyst",
]
ANCHORS = ["a_book", "a_fwd", "a_sales", "a_ebitda", "a_assets"]
BG_LEVELS = [
    "book",
    "net_income",
    "sales",
    "ocf",
    "fcf",
    "rnd",
    "dividends",
    "debt",
    "net_debt",
    "ebitda",
]
THEORY = {
    "financials": (
        ("40",),
        [
            "book_lvl_at",
            "net_income_lvl_at",
            "dividends_lvl_at",
            "fwd_earn_lvl_at",
            "ni_be",
            "niq_be",
            "at_be",
            "earnings_variability",
            "log_at",
            "iv_lvl_at",
            "has_analyst",
            "loss",
        ],
    ),
    "utilities_real_estate": (
        ("55", "60"),
        [
            "net_income_lvl_at",
            "book_lvl_at",
            "dividends_lvl_at",
            "ocf_lvl_at",
            "fwd_earn_lvl_at",
            "debt_lvl_at",
            "net_debt_lvl_at",
            "at_be",
            "log_at",
            "iv_lvl_at",
            "has_analyst",
            "loss",
        ],
    ),
    "growth": (
        ("45", "50", "35"),
        [
            "sales_lvl_at",
            "gp_at",
            "ebit_sale",
            "sale_gr1",
            "sale_gr3",
            "saleq_gr1",
            "analyst_growth",
            "rnd_lvl_at",
            "rd5_at",
            "fwd_earn_lvl_at",
            "ocf_lvl_at",
            "cash_at",
            "log_at",
            "has_analyst",
            "loss",
        ],
    ),
    "cyclical": (
        ("10", "15", "20"),
        [
            "ebitda_lvl_at",
            "ocf_lvl_at",
            "fcf_lvl_at",
            "book_lvl_at",
            "net_debt_lvl_at",
            "ebit_sale",
            "sale_gr1",
            "tangibility",
            "at_turnover",
            "fwd_earn_lvl_at",
            "log_at",
            "has_analyst",
            "loss",
        ],
    ),
    "consumer": (
        ("25", "30"),
        [
            "net_income_lvl_at",
            "fwd_earn_lvl_at",
            "ocf_lvl_at",
            "sales_lvl_at",
            "gp_at",
            "ebit_sale",
            "sale_gr1",
            "dividends_lvl_at",
            "net_debt_lvl_at",
            "at_turnover",
            "log_at",
            "has_analyst",
            "loss",
        ],
    ),
}

# Workers load this shared state from a pickled snapshot (spawn-safe).
_SHARED: dict = {}


def _dump_shared(payload: dict) -> pathlib.Path:

    handle, name = tempfile.mkstemp(suffix=".pkl")
    with open(handle, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return pathlib.Path(name)


def _worker_init(shared_path: str) -> None:

    with open(shared_path, "rb") as fh:
        _SHARED.update(pickle.load(fh))


def directory(work_dir: pathlib.Path) -> pathlib.Path:
    """Folder holding the valuation predictions and model selection."""
    return work_dir / "gap_micro_best"


def build(
    panel_path: pathlib.Path, work_dir: pathlib.Path, device: str = "cpu"
) -> pathlib.Path:
    """Fits every valuation model and picks the best one per test year.

    Cross-sectional models (M0-M5, M7) cross-fit each month by firm. Pooled
    models (M6, M8) train on months up to December of Y-3 and predict Y-2..Y.
    The selected model has the lowest validation error over Y-2..Y-1.
    """
    output = directory(work_dir)
    if (output / SELECTION_FILE).exists():
        return output
    output.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", FutureWarning)
        _SHARED["device"] = device
        panel = _build_panel(panel_path, output)
        xs = _cross_sectional(panel)
        xs.to_parquet(output / XS_FILE, index=False)
        pooled, diagnostics = _pooled(panel, xs)
        pooled.to_parquet(output / POOLED_FILE, index=False)
    selection = _select_models(xs, pooled)
    selection["pooled_diagnostics"] = diagnostics
    (output / SELECTION_FILE).write_text(json.dumps(selection, indent=2))
    return output


def _analyst_forward(grid: pd.DataFrame, cache: pathlib.Path) -> pd.DataFrame:
    """Next-four-quarter consensus EPS and its growth over trailing EPS."""
    if cache.exists():
        return pd.read_parquet(cache)
    state = ibes.build_expectation_state(grid)
    state["eom"] = state.eom.astype("datetime64[ns]")
    rows = state[
        ["permno", "eom", "TICKER", "STATPERS", "FPEDATS", "PRICE"]
    ].rename(columns={"FPEDATS": "target_fpe"})
    consensus = ibes.load_consensus()[
        ["TICKER", "STATPERS", "FPEDATS", "MEANEST"]
    ]
    forward = rows.merge(consensus, on=["TICKER", "STATPERS"], how="inner")
    forward = forward[forward.FPEDATS >= forward.target_fpe].sort_values(
        ["permno", "eom", "FPEDATS"]
    )
    forward = forward[forward.groupby(["permno", "eom"]).cumcount() < 4]
    summed = forward.groupby(["permno", "eom"]).MEANEST.agg(["sum", "size"])
    summed = summed[summed["size"] == 4]["sum"].rename("fwd_eps").reset_index()
    surprises = ibes.load_surprises()
    surprises = surprises.sort_values(["TICKER", "anndats"]).copy()
    grouped = surprises.groupby("TICKER")
    surprises["trail_eps"] = grouped.actual.transform(
        lambda s: s.rolling(4, min_periods=4).sum()
    )
    consecutive = (surprises.fpe - grouped.fpe.shift(3)).dt.days.between(
        260, 300
    )
    surprises.loc[~consecutive, "trail_eps"] = np.nan
    trailing = pd.merge_asof(
        rows[["permno", "eom", "TICKER"]].sort_values("eom"),
        surprises[["TICKER", "anndats", "trail_eps"]]
        .assign(anndats=surprises.anndats.astype("datetime64[ns]"))
        .sort_values("anndats"),
        left_on="eom",
        right_on="anndats",
        by="TICKER",
        direction="backward",
    )
    out = (
        rows[["permno", "eom", "PRICE"]]
        .merge(summed, on=["permno", "eom"], how="left")
        .merge(
            trailing[["permno", "eom", "trail_eps"]],
            on=["permno", "eom"],
            how="left",
        )
    )
    out["analyst_growth"] = (out.fwd_eps / out.trail_eps - 1.0).where(
        out.trail_eps > 0
    )
    out = out.drop_duplicates(["permno", "eom"])[
        ["permno", "eom", "PRICE", "fwd_eps", "analyst_growth"]
    ]
    out.to_parquet(cache, index=False)
    return out


def _winsorize_by_month(
    values: np.ndarray, month_codes: np.ndarray
) -> np.ndarray:
    out = values.copy()
    for code in np.unique(month_codes):
        rows = month_codes == code
        block = values[rows]
        low, high = np.nanpercentile(block, [1, 99], axis=0)
        out[rows] = np.clip(block, low, high)
    return out


def _build_panel(
    panel_path: pathlib.Path, output: pathlib.Path
) -> pd.DataFrame:
    """Asset-scaled accounting and analyst features with the target y."""
    columns = [
        "permno",
        "eom",
        "gics",
        "prc",
        "me_company",
        "at_me",
        "ebitda_mev",
        "bev_mev",
        "intrinsic_value",
        *LEVELS.values(),
        *RATIOS,
    ]
    raw = pq.read_table(
        panel_path, columns=list(dict.fromkeys(columns))
    ).to_pandas()
    raw["eom"] = ibes.month_end(raw.eom).astype("datetime64[ns]")
    raw = raw.drop_duplicates(["permno", "eom"])
    raw = raw[(raw.at_me > 0) & (raw.me_company > 0)].reset_index(drop=True)
    at_me = raw.at_me.to_numpy(float)
    me = raw.me_company.to_numpy(float)
    ev_factor = 1.0 + raw.netdebt_me.to_numpy(float)
    feats = {
        f"{name}_lvl_at": raw[column].to_numpy(float) / at_me
        for name, column in LEVELS.items()
    }
    feats["ebitda_lvl_at"] = raw.ebitda_mev.to_numpy(float) * ev_factor / at_me
    feats["book_ev_lvl_at"] = raw.bev_mev.to_numpy(float) * ev_factor / at_me
    feats["iv_lvl_at"] = raw.intrinsic_value.to_numpy(float) / (at_me * me)
    forward = _analyst_forward(
        raw[["permno", "eom"]], output / "analyst_forward.parquet"
    )
    merged = raw[["permno", "eom", "prc"]].merge(
        forward, on=["permno", "eom"], how="left"
    )
    price = merged.prc.abs().to_numpy(float)
    same_basis = (merged.PRICE / merged.prc.abs()).between(0.5, 2.0)
    fwd_ep = np.where(
        same_basis.to_numpy(), merged.fwd_eps.to_numpy(float) / price, np.nan
    )
    feats["fwd_earn_lvl_at"] = fwd_ep / at_me
    feats["analyst_growth"] = merged.analyst_growth.to_numpy(float)
    feats["log_at"] = np.log(at_me * me)
    for column in RATIOS:
        feats[column] = raw[column].to_numpy(float)
    feats["has_analyst"] = np.isfinite(fwd_ep).astype(float)
    feats["loss"] = (raw.ni_me.to_numpy(float) < 0).astype(float)
    panel = pd.DataFrame(feats)
    gics = raw.gics.astype("string").fillna("")
    panel.insert(0, "permno", raw.permno.to_numpy())
    panel.insert(1, "eom", raw.eom.to_numpy())
    panel["sector"] = gics.str[:2].to_numpy(dtype=object)
    panel["gics4"] = gics.str[:4].to_numpy(dtype=object)
    panel["gics6"] = gics.str[:6].to_numpy(dtype=object)
    panel["y"] = -np.log(at_me)
    panel["assets"] = at_me * me
    with np.errstate(divide="ignore", invalid="ignore"):
        for name, column in [
            ("a_book", "book_lvl_at"),
            ("a_fwd", "fwd_earn_lvl_at"),
            ("a_sales", "sales_lvl_at"),
            ("a_ebitda", "ebitda_lvl_at"),
        ]:
            values = panel[column].to_numpy(float)
            panel[name] = np.where(values > 0, np.log(values), np.nan)
    panel.loc[panel.sector == "40", "a_ebitda"] = np.nan
    panel["a_assets"] = 0.0
    firm_hash = pd.util.hash_pandas_object(panel.permno, index=False)
    panel["fold"] = (firm_hash.to_numpy() % N_FOLDS).astype(int)
    codes = panel.eom.dt.year.to_numpy() * 12 + panel.eom.dt.month.to_numpy()
    continuous = [column for column in FEATURES if column not in FLAGS]
    panel[continuous] = _winsorize_by_month(
        panel[continuous].to_numpy(float), codes
    )
    return panel


def _standardize(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(train, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    train = np.where(np.isnan(train), median, train)
    test = np.where(np.isnan(test), median, test)
    mean, std = train.mean(axis=0), train.std(axis=0)
    std = np.where(std > 1e-9, std, 1.0)
    return (
        np.clip((train - mean) / std, -5, 5),
        np.clip((test - mean) / std, -5, 5),
    )


def _ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    lam: float = RIDGE_LAMBDA,
) -> np.ndarray:
    a, b = _standardize(train_x, test_x)
    centre = train_y.mean()
    weights = np.linalg.solve(
        a.T @ a + lam * np.eye(a.shape[1]), a.T @ (train_y - centre)
    )
    return b @ weights + centre


def _group_median(
    train_keys: list,
    train_values: np.ndarray,
    test_keys: list,
    minimum: int = 8,
) -> np.ndarray:
    """Median by the finest industry level with enough training firms."""
    out = np.full(len(test_keys[0]), np.nan)
    for keys_train, keys_test in zip(train_keys, test_keys, strict=True):
        stats = (
            pd.Series(train_values).groupby(keys_train).agg(["median", "size"])
        )
        stats = stats[(stats["size"] >= minimum) & (stats.index != "")]
        mapped = pd.Series(keys_test).map(stats["median"])
        out = np.where(np.isnan(out), mapped.to_numpy(dtype=float), out)
    return np.where(np.isnan(out), np.nanmedian(train_values), out)


def _sparse_columns(values: np.ndarray) -> list:
    return [
        j for j in range(values.shape[1]) if np.isnan(values[:, j]).mean() > 0.1
    ]


def _with_flags(values: np.ndarray, sparse: list) -> np.ndarray:
    if not sparse:
        return values
    return np.column_stack([values, np.isnan(values[:, sparse]).astype(float)])


def _dummies(labels: np.ndarray) -> np.ndarray:
    return pd.get_dummies(pd.Series(labels, dtype="string")).to_numpy(
        dtype=float
    )


def _trees(
    params: dict,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> np.ndarray:
    config = dict(params)
    rounds = config.pop("n_estimators")
    booster = xgb.train(
        config, xgb.DMatrix(train_x, label=train_y), num_boost_round=rounds
    )
    return booster.predict(xgb.DMatrix(test_x))


def _fit_month(month: pd.Timestamp) -> pd.DataFrame:
    """Cross-fits the cross-sectional models on one month by firm fold."""
    block = _SHARED["panel"].iloc[_SHARED["months"][month]]
    block = block.reset_index(drop=True)
    n = len(block)
    sector = block.sector.to_numpy()
    x = block[FEATURES].to_numpy(float)
    linear_x = np.column_stack(
        [
            _with_flags(x, _sparse_columns(x)),
            _dummies(sector),
            _dummies(block.gics4.to_numpy()),
        ]
    )
    coded_x = np.column_stack(
        [x, pd.factorize(block.sector)[0], pd.factorize(block.gics4)[0]]
    ).astype(float)
    y = block.y.to_numpy(float)
    low, high = np.percentile(y, [1, 99])
    y_fit = np.clip(y, low, high)
    assets = block.assets.to_numpy(float)
    levels = np.column_stack(
        [block[f"{name}_lvl_at"].to_numpy(float) * assets for name in BG_LEVELS]
    )
    levels = np.nan_to_num(_winsorize_by_month(levels, np.zeros(n)), nan=0.0)
    design = np.column_stack([np.ones(n), levels])
    market_equity = np.exp(y) * assets
    fold = block.fold.to_numpy()
    keys = [block.gics6.to_numpy(), block.gics4.to_numpy(), sector]
    group_x = {}
    for label, (sectors, columns) in THEORY.items():
        values = block[columns].to_numpy(float)
        group_x[label] = (
            np.isin(sector, sectors),
            _with_flags(values, _sparse_columns(values)),
        )
    names = ["M0", "M2", "M3", "M4", "M5", "M7", "m_med"]
    preds = {name: np.full(n, np.nan) for name in names}
    for f in range(N_FOLDS):
        tr, te = fold != f, fold == f
        if te.sum() == 0 or tr.sum() < 100:
            continue
        preds["m_med"][te] = np.median(y[tr])
        implied = []
        for anchor in ANCHORS:
            a = block[anchor].to_numpy(float)
            valid = tr & np.isfinite(a)
            if valid.sum() < 30:
                continue
            median = _group_median(
                [k[valid] for k in keys], (y - a)[valid], [k[te] for k in keys]
            )
            implied.append(median + a[te])
        preds["M0"][te] = np.nanmean(np.vstack(implied), axis=0)
        preds["M2"][te] = _ridge(linear_x[tr], y_fit[tr], linear_x[te])
        beta, *_ = np.linalg.lstsq(design[tr], market_equity[tr], rcond=None)
        preds["M3"][te] = np.log(
            np.maximum(design[te] @ beta / assets[te], 1e-3)
        )
        preds["M4"][te] = _trees(XS_TREES, x[tr], y_fit[tr], x[te])
        preds["M5"][te] = _trees(XS_TREES, coded_x[tr], y_fit[tr], coded_x[te])
        theory = preds["M2"].copy()
        for in_group, values in group_x.values():
            group_tr, group_te = tr & in_group, te & in_group
            if group_tr.sum() >= 60 and group_te.any():
                theory[group_te] = _ridge(
                    values[group_tr], y_fit[group_tr], values[group_te]
                )
        preds["M7"][te] = theory[te]
    out = block[["permno", "eom", "y"]].copy()
    for name, values in preds.items():
        out[name] = values
    return out


def _cross_sectional(panel: pd.DataFrame) -> pd.DataFrame:
    _SHARED["panel"] = panel
    _SHARED["months"] = {
        month: rows
        for month, rows in panel.groupby("eom").indices.items()
        if pd.Timestamp(month) >= FIRST_MONTH
    }
    months = sorted(_SHARED["months"])
    frames, started = [], time.time()
    shared_path = _dump_shared(
        {
            "panel": _SHARED["panel"],
            "months": _SHARED["months"],
            "device": _SHARED.get("device", "cpu"),
        }
    )
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(
        XS_WORKERS, initializer=_worker_init, initargs=(str(shared_path),)
    ) as workers:
        results = workers.imap(_fit_month, months)
        for number, (month, out) in enumerate(
            zip(months, results, strict=True)
        ):
            frames.append(out)
            if number % 12 == 0:
                print(
                    f"valuation {pd.Timestamp(month).date()}: {len(out)} rows,"
                    f" {time.time() - started:.0f}s",
                    flush=True,
                )
    return pd.concat(frames, ignore_index=True)


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def _fit_moe(x: np.ndarray, g: np.ndarray, y: np.ndarray) -> dict:
    """Linear experts mixed by a softmax gate on firm traits, fitted by EM."""
    n = len(y)
    xb, gb = np.column_stack([np.ones(n), x]), np.column_stack([np.ones(n), g])
    head = g[: min(n, 50_000)]
    _, _, vt = np.linalg.svd(head - head.mean(axis=0), full_matrices=False)
    first_pc = g @ vt[0]
    start = np.searchsorted(np.quantile(first_pc, [0.25, 0.5, 0.75]), first_pc)
    resp = np.full((n, MOE_EXPERTS), 0.1 / (MOE_EXPERTS - 1))
    resp[np.arange(n), start] = 0.9
    gate = np.zeros((gb.shape[1], MOE_EXPERTS))
    m, v = np.zeros_like(gate), np.zeros_like(gate)
    penalty = RIDGE_LAMBDA * np.eye(xb.shape[1])
    penalty[0, 0] = 0.0
    weights = np.zeros((xb.shape[1], MOE_EXPERTS))
    variance = np.ones(MOE_EXPERTS)
    history, step = [], 0
    for _ in range(MOE_ITERATIONS):
        for k in range(MOE_EXPERTS):
            weighted = xb * resp[:, k : k + 1]
            weights[:, k] = np.linalg.solve(
                weighted.T @ xb + penalty, weighted.T @ y
            )
            residual = y - xb @ weights[:, k]
            variance[k] = max(
                float((resp[:, k] * residual**2).sum() / resp[:, k].sum()),
                MOE_MIN_VARIANCE,
            )
        for _ in range(MOE_GATE_STEPS):
            step += 1
            grad = gb.T @ (_softmax(gb @ gate) - resp) / n + 1e-3 * gate
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad**2
            gate -= (
                0.05
                * (m / (1 - 0.9**step))
                / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
            )
        log_prior = np.log(_softmax(gb @ gate) + 1e-12)
        log_density = (
            -0.5 * np.log(2 * np.pi * variance)
            - 0.5 * (y[:, None] - xb @ weights) ** 2 / variance
        )
        joint = log_prior + log_density
        top = joint.max(axis=1, keepdims=True)
        log_likelihood = top[:, 0] + np.log(np.exp(joint - top).sum(axis=1))
        resp = np.exp(joint - log_likelihood[:, None])
        history.append(round(float(log_likelihood.mean()), 4))
    return {
        "weights": weights,
        "gate": gate,
        "variance": variance,
        "history": history,
    }


def _predict_moe(model: dict, x: np.ndarray, g: np.ndarray) -> np.ndarray:
    n = len(x)
    prior = _softmax(np.column_stack([np.ones(n), g]) @ model["gate"])
    experts = np.column_stack([np.ones(n), x]) @ model["weights"]
    return (prior * experts).sum(axis=1)


def _fit_fold(year: int) -> tuple[int, pd.DataFrame, dict]:
    """Pooled tree (M6) and mixture-of-experts (M8) models for one year."""
    panel, xs = _SHARED["panel"], _SHARED["xs"]
    month_median = _SHARED["month_median"]
    started = time.time()
    train = panel[panel.eom <= pd.Timestamp(f"{year - 3}-12-31")]
    target = panel[
        (panel.eom >= pd.Timestamp(f"{year - 2}-01-01"))
        & (panel.eom <= pd.Timestamp(f"{year}-12-31"))
    ]
    target = target.merge(
        xs[["permno", "eom", "m_med"]], on=["permno", "eom"], how="inner"
    )
    y = (train.y - month_median.loc[train.index]).to_numpy(float)
    low, high = np.percentile(y, [1, 99])
    y = np.clip(y, low, high)
    x_train = train[FEATURES].to_numpy(float)
    x_target = target[FEATURES].to_numpy(float)
    try:
        m6 = _trees(
            {**POOLED_TREES, "device": _SHARED["device"]}, x_train, y, x_target
        )
    except xgb.core.XGBoostError:
        m6 = _trees({**POOLED_TREES, "nthread": 6}, x_train, y, x_target)
    rows = np.random.default_rng(year).choice(
        len(train), size=min(len(train), MOE_MAX_ROWS), replace=False
    )
    sparse = _sparse_columns(x_train)
    a_train, a_target = _standardize(
        _with_flags(x_train[rows], sparse), _with_flags(x_target, sparse)
    )
    g_train, g_target = _standardize(
        train[GATE_TRAITS].to_numpy(float)[rows],
        target[GATE_TRAITS].to_numpy(float),
    )
    model = _fit_moe(a_train, g_train, y[rows])
    m8 = _predict_moe(model, a_target, g_target)
    out = target[["permno", "eom"]].copy()
    out["fold_year"] = year
    out["M6"] = m6 + target.m_med.to_numpy()
    out["M8"] = m8 + target.m_med.to_numpy()
    diagnostics = {
        "train_rows": int(len(train)),
        "moe_rows": int(len(rows)),
        "moe_loglik_first_last": [model["history"][0], model["history"][-1]],
        "seconds": round(time.time() - started),
    }
    return year, out, diagnostics


def _pooled(panel: pd.DataFrame, xs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    _SHARED.update(
        panel=panel,
        xs=xs[["permno", "eom", "m_med"]],
        month_median=panel.groupby("eom").y.transform("median"),
    )
    frames, diagnostics = [], {}
    shared_path = _dump_shared(
        {
            "panel": _SHARED["panel"],
            "xs": _SHARED["xs"],
            "month_median": _SHARED["month_median"],
            "device": _SHARED.get("device", "cpu"),
        }
    )
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(
        POOLED_WORKERS, initializer=_worker_init, initargs=(str(shared_path),)
    ) as workers:
        for year, out, diagnostic in workers.imap_unordered(
            _fit_fold, TEST_YEARS
        ):
            frames.append(out)
            diagnostics[year] = diagnostic
            print(f"valuation pooled {year}: {diagnostic}", flush=True)
    pooled = pd.concat(frames, ignore_index=True)
    return pooled, dict(sorted(diagnostics.items()))


def _select_models(xs: pd.DataFrame, pooled: pd.DataFrame) -> dict:
    """Lowest median absolute pricing error on the two validation years."""
    folds = {}
    for year in TEST_YEARS:
        block = xs.loc[
            xs.eom.between(
                pd.Timestamp(f"{year - 2}-01-01"),
                pd.Timestamp(f"{year - 1}-12-31"),
            )
        ]
        annual = pooled.loc[pooled.fold_year == year].drop(columns="fold_year")
        block = block.merge(annual, on=["permno", "eom"], how="left")
        block["M9"] = block[ENSEMBLE].mean(axis=1, skipna=False)
        block = block.dropna(subset=MODELS)
        # Rounding is part of the rule: ties go to the earlier model.
        mdae = {
            name: round(float((block[name] - block.y).abs().median()), 3)
            for name in MODELS
        }
        if not mdae or not all(np.isfinite(list(mdae.values()))):
            raise ValueError(f"No complete valuation validation data: {year}")
        folds[year] = {
            "best_pricing_model": min(mdae, key=mdae.get),
            "validation_mdae": mdae,
        }
    return {"folds": folds}
