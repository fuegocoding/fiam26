"""Point-in-time monthly company graphs, independent of prediction/backtest code.

Only month-t or earlier inputs enter a graph. The input panel must itself contain
historically available observations; month labels cannot certify source vintages.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import re

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]
# Economic characteristics only: no momentum, correlation, beta, or volatility.
DEFAULT_FUNDAMENTALS = (
    "be_me", "sale_me", "ocf_me", "ebitda_mev", "gp_at", "op_at",
    "niq_at", "ebit_sale", "at_turnover", "cash_at", "debt_me",
    "tangibility", "rd_sale", "capx_gr1", "at_gr1", "sale_gr1",
    "emp_gr1", "oaccruals_at",
)
COMPONENTS = ("industry", "business", "fundamentals", "events", "returns")
_TOKEN = re.compile(r"[a-z][a-z0-9]{2,}")
_TEXT_STOPWORDS = frozenset(
    "about above after again against all also and any are because been before being below between both business "
    "but can company corporation could did does doing during each for from further had has have having here how inc "
    "into its itself llc may more most not now other our ours out over own said same sec should such than that the "
    "their theirs them themselves then there these they this those through under until very was were what when where "
    "which while who will with would www".split()
)


@dataclass(frozen=True)
class GraphConfig:
    weights: dict[str, float] = field(default_factory=lambda: {
        "industry": 0.20, "business": 0.42, "fundamentals": 0.33,
        "events": 0.04, "returns": 0.01,
    })
    n_neighbors: int = 15
    fundamental_columns: tuple[str, ...] = DEFAULT_FUNDAMENTALS
    return_window: int = 36
    min_return_overlap: int = 12
    min_fundamental_overlap: int = 3
    fundamental_bandwidth: float = 1.0
    winsor_quantile: float = 0.01
    min_sector_observations: int = 20
    characteristic_lag_months: int = 0
    business_lookback_months: int = 60
    event_lookback_months: int = 3
    business_hash_dimensions: int = 512
    event_hash_dimensions: int = 256
    softmax_temperature: float = 0.08
    candidate_pool_size: int = 75
    component_candidate_pool_size: int = 25
    block_size: int = 256

    def normalized_weights(self) -> dict[str, float]:
        if set(self.weights) != set(COMPONENTS):
            raise ValueError(f"weights must contain exactly {COMPONENTS}")
        values = np.array([self.weights[k] for k in COMPONENTS], dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
            raise ValueError("weights must be finite, nonnegative, and have positive sum")
        values /= values.sum()
        if values[4] > 0.05 + 1e-12:
            raise ValueError("return correlation is limited to 5% of total component weight")
        if values[3] > 0.10 + 1e-12:
            raise ValueError("event text is limited to 10% of total component weight")
        if values[0] + values[1] + values[2] < 0.85 - 1e-12:
            raise ValueError("industry, business, and fundamentals must receive at least 85% of total weight")
        return dict(zip(COMPONENTS, values.tolist()))

    def __post_init__(self):
        self.normalized_weights()
        for name in ("n_neighbors", "return_window", "min_return_overlap",
                     "min_fundamental_overlap", "min_sector_observations", "event_lookback_months",
                     "business_lookback_months", "business_hash_dimensions", "event_hash_dimensions",
                     "candidate_pool_size", "component_candidate_pool_size", "block_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 10 <= self.n_neighbors <= 20:
            raise ValueError("n_neighbors must be between 10 and 20")
        if self.candidate_pool_size < self.n_neighbors:
            raise ValueError("candidate_pool_size must be at least n_neighbors")
        if not 2 <= self.min_return_overlap <= self.return_window:
            raise ValueError("min_return_overlap must be between 2 and return_window")
        if not self.fundamental_columns or len(set(self.fundamental_columns)) != len(self.fundamental_columns):
            raise ValueError("fundamental_columns must be nonempty and unique")
        if any(c in {"ret", "ret_exc", "ret_exc_lead1m", "target_month"}
               or c.startswith(("ret_", "pred_")) for c in self.fundamental_columns):
            raise ValueError("return/target columns cannot be fundamental characteristics")
        if self.min_fundamental_overlap > len(self.fundamental_columns):
            raise ValueError("min_fundamental_overlap exceeds the number of characteristics")
        if not np.isfinite(self.fundamental_bandwidth) or self.fundamental_bandwidth <= 0:
            raise ValueError("fundamental_bandwidth must be finite and positive")
        if not 0 <= self.winsor_quantile < 0.5:
            raise ValueError("winsor_quantile must be in [0, 0.5)")
        if not np.isfinite(self.softmax_temperature) or self.softmax_temperature <= 0:
            raise ValueError("softmax_temperature must be finite and positive")
        if (not isinstance(self.characteristic_lag_months, int)
                or isinstance(self.characteristic_lag_months, bool)
                or self.characteristic_lag_months < 0):
            raise ValueError("characteristic_lag_months must be a nonnegative integer")


@dataclass
class MonthlyGraph:
    month: pd.Timestamp
    nodes: pd.DataFrame
    similarity: sparse.csr_matrix  # retained combined similarity before row normalization
    adjacency: sparse.csr_matrix  # A: directed, zero diagonal, rows sum to one or zero
    features: pd.DataFrame  # X, in the same company order as adjacency
    standardized_features: pd.DataFrame
    peer_relative: pd.DataFrame
    peer_coverage: pd.DataFrame
    edges: pd.DataFrame  # selected edges with component-level evidence
    candidates: pd.DataFrame  # broader return-free shortlist for validation reweighting
    metadata: dict

    def save(self, directory: str | Path, *, include_candidates: bool = False) -> Path:
        destination = Path(directory) / self.month.strftime("%Y-%m")
        destination.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(destination / "adjacency.npz", self.adjacency)
        sparse.save_npz(destination / "similarity.npz", self.similarity)
        for name in ("nodes", "features", "standardized_features", "peer_relative",
                     "peer_coverage", "edges"):
            getattr(self, name).to_parquet(destination / f"{name}.parquet", index=True)
        if include_candidates:
            self.candidates.to_parquet(destination / "candidates.parquet", index=False)
        (destination / "metadata.json").write_text(json.dumps(self.metadata, indent=2) + "\n")
        return destination


def peer_state_features(adjacency: sparse.csr_matrix, features: pd.DataFrame):
    """Return featurewise observed-peer means, plus observed peer weight.

    For missing peer values, normalize over observed neighbors for that column.
    Zero peer coverage produces NaN, never a fake zero.
    """
    x = features.to_numpy(dtype=float)
    if adjacency.shape != (len(x), len(x)):
        raise ValueError("adjacency and features must have matching node counts")
    observed = np.isfinite(x)
    coverage = np.asarray(adjacency @ observed.astype(float))
    totals = np.asarray(adjacency @ np.where(observed, x, 0.0))
    means = np.divide(totals, coverage, out=np.full_like(totals, np.nan), where=coverage > 0)
    return (pd.DataFrame(means, index=features.index, columns=features.columns),
            pd.DataFrame(coverage, index=features.index, columns=features.columns))


def peer_relative_features(adjacency: sparse.csr_matrix, features: pd.DataFrame):
    """Return X - peer state, plus observed peer weight per feature.

    Missing own values or zero peer coverage produce NaN, never fake zeros.
    """
    x = features.to_numpy(dtype=float)
    means, coverage = peer_state_features(adjacency, features)
    mean_values = means.to_numpy()
    observed = np.isfinite(x)
    relative = np.where(observed, x - mean_values, np.nan)
    return (pd.DataFrame(relative, index=features.index, columns=features.columns),
            coverage)


def reweight_candidates(candidates: pd.DataFrame, n_nodes: int, config: GraphConfig):
    """Recombine cached components, select top-k, and apply row-wise softmax."""
    required = {"source_index", "target_index", "source_permco", "target_permco", *COMPONENTS}
    if missing := required - set(candidates.columns):
        raise ValueError(f"missing candidate columns: {sorted(missing)}")
    weights = config.normalized_weights()
    ranked = candidates.copy()
    ranked = ranked[(ranked["industry"] > 0) & (ranked["source_index"] != ranked["target_index"])].copy()
    # Reapply the hierarchical candidate rule for the trial's own k. The cache
    # was built broadly enough for k=20, so k=10/15 trials can remain narrower.
    eligible = np.zeros(len(ranked), dtype=bool)
    for positions in ranked.groupby("source_index", sort=False).indices.values():
        positions = np.asarray(positions, dtype=int)
        group_industry = ranked.iloc[positions].industry.to_numpy(float)
        needed = min(config.n_neighbors, len(positions))
        threshold = 0.25
        for level in (1.0, 0.75, 0.50, 0.25):
            if np.count_nonzero(group_industry >= level) >= needed:
                threshold = level
                break
        eligible[positions] = group_industry >= threshold
    ranked = ranked.iloc[np.flatnonzero(eligible)].copy()
    ranked["similarity"] = sum(ranked[name] * weight for name, weight in weights.items())
    ranked = ranked.sort_values(
        ["source_index", "similarity", "target_permco"], ascending=[True, False, True]
    )
    ranked = ranked.groupby("source_index", sort=False, as_index=False).head(config.n_neighbors).copy()
    ranked["rank"] = ranked.groupby("source_index", sort=False).cumcount() + 1
    row_max = ranked.groupby("source_index").similarity.transform("max")
    ranked["softmax_mass"] = np.exp((ranked.similarity - row_max) / config.softmax_temperature)
    ranked["weight"] = ranked.softmax_mass / ranked.groupby("source_index").softmax_mass.transform("sum")
    shape = (n_nodes, n_nodes)
    similarity = sparse.csr_matrix(
        (ranked.similarity.to_numpy(float),
         (ranked.source_index.to_numpy(int), ranked.target_index.to_numpy(int))), shape=shape,
    )
    adjacency = sparse.csr_matrix(
        (ranked.weight.to_numpy(float),
         (ranked.source_index.to_numpy(int), ranked.target_index.to_numpy(int))), shape=shape,
    )
    return ranked.reset_index(drop=True), similarity, adjacency


def _standardize(frame: pd.DataFrame, quantile: float) -> pd.DataFrame:
    x = frame.astype(float).replace([np.inf, -np.inf], np.nan)
    x = x.clip(lower=x.quantile(quantile), upper=x.quantile(1 - quantile), axis=1)
    scale = x.std(ddof=0)
    # Constant / entirely missing dimensions contain no similarity information.
    return (x - x.mean()) / scale.where(scale > 1e-12)


def _sector_labels(snapshot: pd.DataFrame) -> pd.Series:
    gics = snapshot.get("gics", pd.Series(pd.NA, index=snapshot.index)).astype("string")
    gics = gics.str.replace(r"\.0$", "", regex=True)
    sector = gics.where(gics.str.fullmatch(r"[0-9]{2,8}") & ~gics.str.startswith("00")).str[:2]
    ff = pd.to_numeric(snapshot.get("ff49", pd.Series(np.nan, index=snapshot.index)), errors="coerce")
    fallback = ff.where(ff.between(1, 49)).map(lambda x: f"ff49:{int(x)}" if pd.notna(x) else pd.NA)
    return sector.fillna(fallback).astype("string")


def _sector_standardize(frame: pd.DataFrame, snapshot: pd.DataFrame, config: GraphConfig) -> pd.DataFrame:
    """Winsorize and standardize fundamentals within contemporaneous sectors."""
    global_z = _standardize(frame, config.winsor_quantile)
    result = global_z.copy()
    sectors = _sector_labels(snapshot)
    for _, positions in sectors.groupby(sectors, dropna=True).groups.items():
        positions = list(positions)
        if len(positions) < config.min_sector_observations:
            continue
        local = _standardize(frame.loc[positions], config.winsor_quantile)
        # Use local scaling wherever a sector has enough observations and dispersion;
        # retain the global fallback for a constant or nearly empty sector dimension.
        for column in frame.columns:
            if local[column].notna().sum() >= config.min_sector_observations:
                result.loc[positions, column] = local[column]
    return result


def _hashed_tfidf(documents: list[str], dimensions: int) -> sparse.csr_matrix:
    """Deterministic dependency-free word/bigram TF-IDF with unit-length rows."""
    rows, columns, values = [], [], []
    for row, document in enumerate(documents):
        tokens = [token for token in _TOKEN.findall((document or "").lower())
                  if token not in _TEXT_STOPWORDS and not token.isdigit()]
        terms = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        counts = Counter(
            int.from_bytes(hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest(), "little") % dimensions
            for term in terms
        )
        for column, count in counts.items():
            rows.append(row); columns.append(column); values.append(1.0 + np.log(count))
    matrix = sparse.csr_matrix((values, (rows, columns)), shape=(len(documents), dimensions), dtype=float)
    if matrix.nnz == 0:
        return matrix
    document_frequency = np.bincount(matrix.indices, minlength=dimensions)
    idf = np.log((1.0 + len(documents)) / (1.0 + document_frequency)) + 1.0
    matrix = matrix @ sparse.diags(idf)
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    return (sparse.diags(np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)) @ matrix).tocsr()


def _fundamental_similarity(z: np.ndarray, start: int, stop: int, config: GraphConfig):
    mask = np.isfinite(z).astype(float)
    values = np.nan_to_num(z, nan=0.0)
    overlap = mask[start:stop] @ mask.T
    distances = ((values[start:stop] ** 2) @ mask.T
                 + mask[start:stop] @ (values ** 2).T
                 - 2 * values[start:stop] @ values.T)
    mean_distance = np.divide(np.maximum(distances, 0), overlap,
                              out=np.zeros_like(distances), where=overlap > 0)
    informative = max(int(np.isfinite(z).any(axis=0).sum()), 1)
    similarity = np.exp(-mean_distance / (2 * config.fundamental_bandwidth**2))
    similarity *= overlap / informative
    similarity[overlap < config.min_fundamental_overlap] = 0.0
    return np.clip(similarity, 0, 1)


def _industry_similarity(snapshot: pd.DataFrame, start: int, stop: int):
    """Hierarchical GICS; use FF49, then SIC when a pair lacks usable GICS."""
    n = len(snapshot)
    gics = snapshot.get("gics", pd.Series(pd.NA, index=snapshot.index)).astype("string")
    gics = gics.str.replace(r"\.0$", "", regex=True)
    codes = [gics.where(gics.str.fullmatch(r"[0-9]{" + str(length) + r",8}")
                        & ~gics.str.startswith("00")).str[:length].fillna("").to_numpy(str)
             for length in (2, 4, 6, 8)]
    result = np.zeros((stop - start, n))
    for code, score in zip(codes, (0.25, 0.50, 0.75, 1.0)):
        same = (code[start:stop, None] == code[None, :]) & (code[start:stop, None] != "")
        result = np.maximum(result, same * score)
    valid_gics = codes[0] != ""
    fallback = ~(valid_gics[start:stop, None] & valid_gics[None, :])
    ff = pd.to_numeric(snapshot.get("ff49", pd.Series(np.nan, index=snapshot.index)), errors="coerce")
    ff = ff.where(ff.between(1, 49)).to_numpy(float)
    ff_pair = np.isfinite(ff[start:stop, None]) & np.isfinite(ff[None, :])
    result = np.where(fallback & ff_pair, (ff[start:stop, None] == ff[None, :]) * 0.75, result)
    sic = pd.to_numeric(snapshot.get("sic", pd.Series(np.nan, index=snapshot.index)), errors="coerce")
    sic = sic.where(sic.between(100, 9999)).to_numpy(float)
    sic_score = ((sic[start:stop, None] // 100 == sic[None, :] // 100) * 0.50)
    sic_score = np.maximum(sic_score, (sic[start:stop, None] == sic[None, :]) * 0.75)
    return np.where(fallback & ~ff_pair, sic_score, result)


def _return_similarity(history: np.ndarray, start: int, stop: int, minimum: int):
    """Pairwise-complete Pearson correlation; calendar gaps stay missing."""
    mask = np.isfinite(history).astype(float)
    values = np.where(np.isfinite(history), history, 0.0)
    a, ma = values[start:stop], mask[start:stop]
    overlap = ma @ mask.T
    divisor = np.maximum(overlap, 1)
    sx, sy = a @ mask.T, ma @ values.T
    cov = a @ values.T - sx * sy / divisor
    vx = (a * a) @ mask.T - sx * sx / divisor
    vy = ma @ (values * values).T - sy * sy / divisor
    denom = np.sqrt(np.maximum(vx, 0) * np.maximum(vy, 0))
    valid = (overlap >= minimum) & (denom > 1e-12)
    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=valid)
    # Negative correlation is not economic similarity. Unavailable pairs score 0.
    return np.where(valid, np.clip(corr, 0, 1), 0.0), overlap


class EconomicGraphBuilder:
    """Build arbitrary months from the supplied company/security panel.

    Each PERMCO gets one node. For multiple share classes choose primary_sec,
    then largest contemporaneous market equity, then smallest PERMNO. Return
    history uses the representative selected independently in each past month.
    """

    def __init__(self, panel: pd.DataFrame, config: GraphConfig | None = None,
                 text_evidence: pd.DataFrame | None = None):
        self.config = config or GraphConfig()
        required = {"permco", "permno", "eom", "ret", "me", *self.config.fundamental_columns}
        if missing := required - set(panel.columns):
            raise ValueError(f"missing panel columns: {sorted(missing)}")
        optional = [c for c in ("primary_sec", "gics", "ff49", "sic", "ticker",
                                "company_name", "gvkey") if c in panel]
        # Explicit projection: future labels cannot influence selection or scaling.
        p = panel[sorted(required) + optional].copy()
        p["eom"] = pd.to_datetime(p["eom"])
        if p[["permco", "permno", "eom"]].isna().any().any():
            raise ValueError("company, security and month keys cannot be missing")
        if not p.eom.dt.is_month_end.all() or not (p.eom == p.eom.dt.normalize()).all():
            raise ValueError("eom must be a calendar month-end date")
        if p.duplicated(["permno", "eom"]).any():
            raise ValueError("duplicate security-month observations")
        if "primary_sec" not in p:
            p["primary_sec"] = 0
        p = p.sort_values(["eom", "permco", "primary_sec", "me", "permno"],
                          ascending=[True, True, False, False, True], na_position="last")
        self.panel = p.drop_duplicates(["permco", "eom"]).reset_index(drop=True)
        self.months = pd.DatetimeIndex(sorted(self.panel.eom.unique()))
        self.text_evidence = None
        weights = self.config.normalized_weights()
        if (weights["business"] > 0 or weights["events"] > 0) and text_evidence is None:
            raise ValueError(
                "positive business/event weight requires point-in-time text evidence; "
                "run python -m src.graph.business_text_profiles"
            )
        if text_evidence is not None:
            self.text_evidence = self._prepare_text_evidence(text_evidence)

    @staticmethod
    def _prepare_text_evidence(evidence):
        required = {"permco", "document_id", "available_at", "business_description", "event_text"}
        if missing := required - set(evidence.columns):
            raise ValueError(f"missing text evidence columns: {sorted(missing)}")
        e = evidence[sorted(required)].copy()
        if e[["permco", "document_id"]].isna().any().any():
            raise ValueError("text evidence company/document keys cannot be missing")
        if e.duplicated(["permco", "document_id"]).any():
            raise ValueError("duplicate company-document text evidence")
        e["available_at"] = pd.to_datetime(e["available_at"], utc=True, errors="raise", format="ISO8601")
        if e["available_at"].isna().any():
            raise ValueError("available_at cannot be missing")
        for column in ("business_description", "event_text"):
            e[column] = e[column].fillna("").astype(str)
        return e.sort_values(["available_at", "document_id"]).reset_index(drop=True)

    def _text_vectors(self, month, companies):
        """Return latest business profiles and recent-event vectors known by month-end."""
        if self.text_evidence is None:
            empty_business = sparse.csr_matrix((len(companies), self.config.business_hash_dimensions))
            empty_events = sparse.csr_matrix((len(companies), self.config.event_hash_dimensions))
            return empty_business, empty_events, 0, 0
        end = (month + pd.Timedelta(days=1)).tz_localize("UTC")
        evidence = self.text_evidence
        known = evidence[(evidence.available_at < end) & evidence.permco.isin(companies)]
        business_begin = (month.to_period("M") - (self.config.business_lookback_months - 1)).start_time.tz_localize("UTC")
        descriptions = known[(known.available_at >= business_begin) & (known.business_description.str.len() > 0)]
        business_map = descriptions.groupby("permco", sort=False).business_description.agg(" ".join)
        business_docs = business_map.reindex(companies).fillna("").tolist()

        begin = (month.to_period("M") - (self.config.event_lookback_months - 1)).start_time.tz_localize("UTC")
        recent = known[(known.available_at >= begin) & (known.event_text.str.len() > 0)]
        event_map = recent.groupby("permco", sort=False).event_text.agg(" ".join)
        event_docs = event_map.reindex(companies).fillna("").tolist()
        return (
            _hashed_tfidf(business_docs, self.config.business_hash_dimensions),
            _hashed_tfidf(event_docs, self.config.event_hash_dimensions),
            int(business_map.index.nunique()), len(recent),
        )

    def build_month(self, month: str | pd.Timestamp) -> MonthlyGraph:
        config = self.config
        month = pd.Timestamp(month).to_period("M").to_timestamp("M")
        current = self.panel[self.panel.eom == month].sort_values("permco").reset_index(drop=True)
        if current.empty:
            raise ValueError(f"no company observations for {month.date()}")
        companies = current.permco.to_numpy()
        index = pd.Index(companies, name="permco")
        cutoff = (month.to_period("M") - config.characteristic_lag_months).to_timestamp("M")
        if config.characteristic_lag_months:
            # Exact lagged snapshot; do not carry indefinitely stale characteristics.
            snapshot = self.panel[self.panel.eom == cutoff].set_index("permco").reindex(index)
        else:
            snapshot = current.set_index("permco")
        features = snapshot[list(config.fundamental_columns)].astype(float).replace([np.inf, -np.inf], np.nan)
        z = _sector_standardize(features, snapshot, config)
        periods = pd.period_range(end=month.to_period("M"), periods=config.return_window, freq="M")
        h = self.panel[self.panel.eom.between(periods[0].to_timestamp("M"), month)]
        history = h.pivot(index="permco", columns="eom", values="ret").reindex(
            index=index, columns=periods.to_timestamp("M")).to_numpy(float)
        weights = config.normalized_weights()
        business_vectors, event_vectors, business_count, event_count = self._text_vectors(month, companies)
        candidate_rows = []
        for start in range(0, len(current), config.block_size):
            stop = min(start + config.block_size, len(current))
            industry = _industry_similarity(snapshot, start, stop)
            fundamentals = _fundamental_similarity(z.to_numpy(), start, stop, config)
            returns, overlap = _return_similarity(history, start, stop, config.min_return_overlap)
            business = np.clip((business_vectors[start:stop] @ business_vectors.T).toarray(), 0, 1)
            events = np.clip((event_vectors[start:stop] @ event_vectors.T).toarray(), 0, 1)
            semantic = (weights["industry"] * industry + weights["business"] * business
                        + weights["fundamentals"] * fundamentals + weights["events"] * events)
            # Industry classifications define the candidate set. Text, accounting
            # resemblance, or return co-movement cannot bridge unrelated sectors.
            semantic[industry <= 0] = 0
            semantic[np.arange(stop - start), np.arange(start, stop)] = 0
            for local, row in enumerate(semantic):
                # Cache enough of the narrowest tier that can supply the maximum
                # supported k=20. Reweighting reapplies this rule for its trial k.
                # FF49/SIC fallbacks use the same score levels.
                required = min(20, max(len(current) - 1, 0))
                threshold = 0.25
                for level in (1.0, 0.75, 0.50, 0.25):
                    tier = (industry[local] >= level)
                    tier[start + local] = False
                    if tier.sum() >= required:
                        threshold = level
                        break
                eligible = np.flatnonzero(industry[local] >= threshold)
                eligible = eligible[eligible != start + local]
                # Returns cannot admit candidates. Cache a broad shortlist ranked
                # entirely by economic/text evidence for cheap validation searches.
                order = np.lexsort((companies[eligible], -row[eligible]))
                chosen = set(eligible[order[:config.candidate_pool_size]].tolist())
                for component in (industry, business, fundamentals, events):
                    informative = eligible[component[local, eligible] > 0]
                    component_order = np.lexsort(
                        (companies[informative], -component[local, informative])
                    )
                    chosen.update(
                        informative[component_order[:config.component_candidate_pool_size]].tolist()
                    )
                for neighbor in sorted(chosen):
                    candidate_rows.append((start + local, neighbor,
                                           industry[local, neighbor], business[local, neighbor],
                                           fundamentals[local, neighbor], events[local, neighbor],
                                           returns[local, neighbor], int(overlap[local, neighbor])))
        candidates = pd.DataFrame(candidate_rows, columns=[
            "source_index", "target_index", "industry", "business", "fundamentals",
            "events", "returns", "return_overlap",
        ])
        if len(candidates):
            candidates["source_permco"] = companies[candidates.source_index.to_numpy(int)]
            candidates["target_permco"] = companies[candidates.target_index.to_numpy(int)]
        else:
            candidates["source_permco"] = pd.Series(dtype=np.int64)
            candidates["target_permco"] = pd.Series(dtype=np.int64)
        edges, similarity, adjacency = reweight_candidates(candidates, len(current), config)
        relative, coverage = peer_relative_features(adjacency, features)
        node_columns = [c for c in ("permco", "permno", "gvkey", "ticker", "company_name") if c in current]
        nodes = current[node_columns].copy()
        nodes.index.name = "node_index"
        nodes["eom"] = month
        nodes["characteristic_eom"] = cutoff
        nodes["neighbor_count"] = adjacency.getnnz(axis=1)
        metadata = {"month": str(month.date()), "characteristic_cutoff": str(cutoff.date()),
                    "config": asdict(config), "normalized_weights": weights,
                    "nodes": len(nodes), "edges": adjacency.nnz,
                    "companies_with_business_profiles": business_count,
                    "eligible_recent_event_filings": event_count, "node_key": "permco",
                    "candidate_edges": len(candidates),
                    "directed": True, "row_normalized": True, "weight_transform": "softmax",
                    "input_availability_assumption": "Panel observations were available by their eom; source vintages are not independently certified."}
        return MonthlyGraph(
            month, nodes, similarity, adjacency, features, z, relative, coverage,
            edges, candidates, metadata,
        )

    def iter_months(self, start=None, end=None):
        lo = pd.Timestamp(start).to_period("M").to_timestamp("M") if start else self.months.min()
        hi = pd.Timestamp(end).to_period("M").to_timestamp("M") if end else self.months.max()
        for month in self.months[(self.months >= lo) & (self.months <= hi)]:
            yield self.build_month(month)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=ROOT / "data/raw/chars_final_with_names.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "output/economic_graph_refined")
    parser.add_argument("--config", type=Path, help="JSON GraphConfig overrides; weights are selected outside this pipeline")
    parser.add_argument(
        "--text-evidence", type=Path, default=ROOT / "output/business_text_evidence.parquet",
        help="Point-in-time business descriptions and recent event text parquet",
    )
    parser.add_argument("--start", help="First graph month, e.g. 2015-01")
    parser.add_argument("--end", help="Last graph month, e.g. 2026-08")
    parser.add_argument("--save-candidates", action="store_true",
                        help="save broad component cache for weight validation")
    args = parser.parse_args()
    overrides = json.loads(args.config.read_text()) if args.config else {}
    if "fundamental_columns" in overrides:
        overrides["fundamental_columns"] = tuple(overrides["fundamental_columns"])
    config = GraphConfig(**overrides)
    import pyarrow.parquet as pq
    available = set(pq.ParquetFile(args.panel).schema_arrow.names)
    columns = {"permco", "permno", "eom", "ret", "me", *config.fundamental_columns}
    columns |= available & {"primary_sec", "gics", "ff49", "sic", "ticker", "company_name", "gvkey"}
    panel = pd.read_parquet(args.panel, columns=sorted(columns))
    text_evidence = pd.read_parquet(args.text_evidence) if args.text_evidence.exists() else None
    builder = EconomicGraphBuilder(panel, config, text_evidence)
    count = 0
    for graph in builder.iter_months(args.start, args.end):
        path = graph.save(args.output, include_candidates=args.save_candidates)
        print(f"{graph.month:%Y-%m}: {len(graph.nodes):,} companies, {graph.adjacency.nnz:,} edges -> {path}", flush=True)
        count += 1
    if not count:
        parser.error("no panel months in the requested range")


if __name__ == "__main__":
    main()
