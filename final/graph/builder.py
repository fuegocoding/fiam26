"""Monthly point-in-time company graphs from industry, text and fundamentals."""

import collections
import dataclasses
import hashlib
import json
import pathlib
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

# Economic characteristics only: no momentum, correlation, beta or volatility.
DEFAULT_FUNDAMENTALS = (
    "be_me",
    "sale_me",
    "ocf_me",
    "ebitda_mev",
    "gp_at",
    "op_at",
    "niq_at",
    "ebit_sale",
    "at_turnover",
    "cash_at",
    "debt_me",
    "tangibility",
    "rd_sale",
    "capx_gr1",
    "at_gr1",
    "sale_gr1",
    "emp_gr1",
    "oaccruals_at",
)
COMPONENTS = ("industry", "business", "fundamentals", "events", "returns")
_OPTIONAL_COLUMNS = (
    "primary_sec",
    "gics",
    "ff49",
    "sic",
    "ticker",
    "company_name",
    "gvkey",
)
_INDUSTRY_LEVELS = (1.0, 0.75, 0.50, 0.25)
_TOKEN = re.compile(r"[a-z][a-z0-9]{2,}")
_TEXT_STOPWORDS = frozenset(
    "about above after again against all also and any are because been before"
    " being below between both business but can company corporation could did"
    " does doing during each for from further had has have having here how inc"
    " into its itself llc may more most not now other our ours out over own"
    " said same sec should such than that the their theirs them themselves"
    " then there these they this those through under until very was were what"
    " when where which while who will with would www".split()
)


@dataclasses.dataclass(frozen=True)
class GraphConfig:
    """Component weights and limits of the monthly peer graph.

    Industry, business text and fundamentals carry most of the weight; event
    text and return co-movement only refine the ranking of candidates.
    """

    weights: dict[str, float] = dataclasses.field(
        default_factory=lambda: {
            "industry": 0.20,
            "business": 0.40,
            "fundamentals": 0.35,
            "events": 0.04,
            "returns": 0.01,
        }
    )
    n_neighbors: int = 15
    fundamental_columns: tuple[str, ...] = DEFAULT_FUNDAMENTALS
    return_window: int = 36
    min_return_overlap: int = 12
    min_fundamental_overlap: int = 3
    fundamental_bandwidth: float = 1.0
    winsor_quantile: float = 0.01
    min_sector_observations: int = 20
    business_lookback_months: int = 60
    event_lookback_months: int = 3
    business_hash_dimensions: int = 512
    event_hash_dimensions: int = 256
    softmax_temperature: float = 0.08
    candidate_pool_size: int = 75
    component_candidate_pool_size: int = 25
    block_size: int = 256

    def normalized_weights(self) -> dict[str, float]:
        """Component weights scaled to sum to one."""
        values = np.array([self.weights[k] for k in COMPONENTS], dtype=float)
        return dict(zip(COMPONENTS, (values / values.sum()).tolist(), strict=True))


@dataclasses.dataclass
class MonthlyGraph:
    """One month's directed peer graph over companies."""

    month: pd.Timestamp
    nodes: pd.DataFrame
    adjacency: sparse.csr_matrix  # Zero diagonal; rows sum to one or zero.
    metadata: dict

    def save(self, directory: pathlib.Path) -> pathlib.Path:
        """Writes the graph to `directory`/YYYY-MM and returns that folder."""
        destination = directory / self.month.strftime("%Y-%m")
        destination.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(destination / "adjacency.npz", self.adjacency)
        self.nodes.to_parquet(destination / "nodes.parquet", index=True)
        (destination / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2) + "\n"
        )
        return destination


def build_snapshots(
    panel_path: pathlib.Path,
    text_path: pathlib.Path,
    output_dir: pathlib.Path,
    config: GraphConfig | None = None,
) -> pathlib.Path:
    """Builds and saves every panel month whose snapshot is still missing.

    Args:
        panel_path: Characteristics panel parquet.
        text_path: Dated business and event text from `business_text`.
        output_dir: Directory receiving one YYYY-MM folder per month.
        config: Graph settings; defaults to `GraphConfig()`.

    Returns:
        The output directory.
    """
    config = config or GraphConfig()
    months = pd.DatetimeIndex(
        pd.read_parquet(panel_path, columns=["eom"]).eom.unique()
    ).sort_values()
    missing = [
        month
        for month in months
        if not (output_dir / f"{month:%Y-%m}" / "metadata.json").exists()
    ]
    if not missing:
        return output_dir
    available = set(pq.ParquetFile(panel_path).schema_arrow.names)
    columns = {"permco", "permno", "eom", "ret", "me"}
    columns |= set(config.fundamental_columns)
    columns |= available & set(_OPTIONAL_COLUMNS)
    panel = pd.read_parquet(panel_path, columns=sorted(columns))
    builder = EconomicGraphBuilder(panel, pd.read_parquet(text_path), config)
    for month in missing:
        graph = builder.build_month(month)
        graph.save(output_dir)
        print(
            f"Graph {graph.month:%Y-%m}: {len(graph.nodes):,} companies,"
            f" {graph.adjacency.nnz:,} edges",
            flush=True,
        )
    return output_dir


def reweight_candidates(
    candidates: pd.DataFrame, n_nodes: int, config: GraphConfig
) -> sparse.csr_matrix:
    """Recombines component scores, keeps the top k and applies a softmax."""
    weights = config.normalized_weights()
    ranked = candidates[
        (candidates.industry > 0) & (candidates.source_index != candidates.target_index)
    ].copy()
    eligible = np.zeros(len(ranked), dtype=bool)
    sources = ranked.groupby("source_index", sort=False).indices
    for positions in sources.values():
        positions = np.asarray(positions, dtype=int)
        industry = ranked.iloc[positions].industry.to_numpy(float)
        needed = min(config.n_neighbors, len(positions))
        threshold = 0.25
        for level in _INDUSTRY_LEVELS:
            if np.count_nonzero(industry >= level) >= needed:
                threshold = level
                break
        eligible[positions] = industry >= threshold
    ranked = ranked.iloc[np.flatnonzero(eligible)].copy()
    ranked["similarity"] = sum(
        ranked[name] * weight for name, weight in weights.items()
    )
    ranked = ranked.sort_values(
        ["source_index", "similarity", "target_permco"],
        ascending=[True, False, True],
    )
    ranked = (
        ranked.groupby("source_index", sort=False, as_index=False)
        .head(config.n_neighbors)
        .copy()
    )
    row_max = ranked.groupby("source_index").similarity.transform("max")
    ranked["softmax_mass"] = np.exp(
        (ranked.similarity - row_max) / config.softmax_temperature
    )
    ranked["weight"] = ranked.softmax_mass / ranked.groupby(
        "source_index"
    ).softmax_mass.transform("sum")
    index = (
        ranked.source_index.to_numpy(int),
        ranked.target_index.to_numpy(int),
    )
    return sparse.csr_matrix(
        (ranked.weight.to_numpy(float), index), shape=(n_nodes, n_nodes)
    )


class EconomicGraphBuilder:
    """Builds month graphs over companies, one node per PERMCO.

    For multiple share classes the representative is the primary security,
    then the largest market equity, then the smallest PERMNO.
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        text_evidence: pd.DataFrame,
        config: GraphConfig | None = None,
    ) -> None:
        """Keeps one representative row per company and month.

        Args:
            panel: Security-month panel with keys, returns and fundamentals.
            text_evidence: Dated text from `business_text`.
            config: Graph settings; defaults to `GraphConfig()`.
        """
        self.config = config or GraphConfig()
        required = {
            "permco",
            "permno",
            "eom",
            "ret",
            "me",
            *self.config.fundamental_columns,
        }
        optional = [c for c in _OPTIONAL_COLUMNS if c in panel]
        p = panel[sorted(required) + optional].copy()
        p["eom"] = pd.to_datetime(p["eom"])
        if "primary_sec" not in p:
            p["primary_sec"] = 0
        p = p.sort_values(
            ["eom", "permco", "primary_sec", "me", "permno"],
            ascending=[True, True, False, False, True],
            na_position="last",
        )
        self.panel = p.drop_duplicates(["permco", "eom"]).reset_index(drop=True)
        columns = [
            "available_at",
            "business_description",
            "document_id",
            "event_text",
            "permco",
        ]
        e = text_evidence[columns].copy()
        e["available_at"] = pd.to_datetime(
            e["available_at"], utc=True, errors="raise", format="ISO8601"
        )
        for column in ("business_description", "event_text"):
            e[column] = e[column].fillna("").astype(str)
        self.text_evidence = e.sort_values(["available_at", "document_id"]).reset_index(
            drop=True
        )

    def _text_vectors(
        self, month: pd.Timestamp, companies: np.ndarray
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """Latest business profiles and recent event text known by month-end."""
        config = self.config
        end = (month + pd.Timedelta(days=1)).tz_localize("UTC")
        evidence = self.text_evidence
        known = evidence[
            (evidence.available_at < end) & evidence.permco.isin(companies)
        ]
        period = month.to_period("M")
        business_begin = (
            period - (config.business_lookback_months - 1)
        ).start_time.tz_localize("UTC")
        descriptions = known[
            (known.available_at >= business_begin)
            & (known.business_description.str.len() > 0)
        ]
        business_map = descriptions.groupby(
            "permco", sort=False
        ).business_description.agg(" ".join)
        begin = (period - (config.event_lookback_months - 1)).start_time.tz_localize(
            "UTC"
        )
        recent = known[(known.available_at >= begin) & (known.event_text.str.len() > 0)]
        event_map = recent.groupby("permco", sort=False).event_text.agg(" ".join)
        return (
            _hashed_tfidf(
                business_map.reindex(companies).fillna("").tolist(),
                config.business_hash_dimensions,
            ),
            _hashed_tfidf(
                event_map.reindex(companies).fillna("").tolist(),
                config.event_hash_dimensions,
            ),
        )

    def build_month(self, month: str | pd.Timestamp) -> MonthlyGraph:
        """Builds one month from that month's (or earlier) inputs only."""
        config = self.config
        month = pd.Timestamp(month).to_period("M").to_timestamp("M")
        current = self.panel[self.panel.eom == month].sort_values("permco")
        current = current.reset_index(drop=True)
        companies = current.permco.to_numpy()
        index = pd.Index(companies, name="permco")
        snapshot = current.set_index("permco")
        features = (
            snapshot[list(config.fundamental_columns)]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
        )
        z = _sector_standardize(features, snapshot, config).to_numpy()
        periods = pd.period_range(
            end=month.to_period("M"), periods=config.return_window, freq="M"
        )
        window = self.panel[self.panel.eom.between(periods[0].to_timestamp("M"), month)]
        history = (
            window.pivot(index="permco", columns="eom", values="ret")
            .reindex(index=index, columns=periods.to_timestamp("M"))
            .to_numpy(float)
        )
        weights = config.normalized_weights()
        business_vectors, event_vectors = self._text_vectors(month, companies)
        candidate_rows = []
        for start in range(0, len(current), config.block_size):
            stop = min(start + config.block_size, len(current))
            industry = _industry_similarity(snapshot, start, stop)
            fundamentals = _fundamental_similarity(z, start, stop, config)
            returns = _return_similarity(
                history, start, stop, config.min_return_overlap
            )
            business = np.clip(
                (business_vectors[start:stop] @ business_vectors.T).toarray(),
                0,
                1,
            )
            events = np.clip(
                (event_vectors[start:stop] @ event_vectors.T).toarray(), 0, 1
            )
            semantic = (
                weights["industry"] * industry
                + weights["business"] * business
                + weights["fundamentals"] * fundamentals
                + weights["events"] * events
            )
            # Industry defines the candidate set; text, accounting or return
            # resemblance cannot bridge unrelated sectors.
            semantic[industry <= 0] = 0
            semantic[np.arange(stop - start), np.arange(start, stop)] = 0
            for local, row in enumerate(semantic):
                required = min(20, max(len(current) - 1, 0))
                threshold = 0.25
                for level in _INDUSTRY_LEVELS:
                    tier = industry[local] >= level
                    tier[start + local] = False
                    if tier.sum() >= required:
                        threshold = level
                        break
                eligible = np.flatnonzero(industry[local] >= threshold)
                eligible = eligible[eligible != start + local]
                order = np.lexsort((companies[eligible], -row[eligible]))
                chosen = set(eligible[order[: config.candidate_pool_size]].tolist())
                for component in (industry, business, fundamentals, events):
                    informative = eligible[component[local, eligible] > 0]
                    component_order = np.lexsort(
                        (companies[informative], -component[local, informative])
                    )
                    chosen.update(
                        informative[
                            component_order[: config.component_candidate_pool_size]
                        ].tolist()
                    )
                for neighbor in sorted(chosen):
                    candidate_rows.append(
                        (
                            start + local,
                            neighbor,
                            industry[local, neighbor],
                            business[local, neighbor],
                            fundamentals[local, neighbor],
                            events[local, neighbor],
                            returns[local, neighbor],
                        )
                    )
        candidates = pd.DataFrame(
            candidate_rows,
            columns=[
                "source_index",
                "target_index",
                "industry",
                "business",
                "fundamentals",
                "events",
                "returns",
            ],
        )
        candidates["target_permco"] = companies[candidates.target_index.to_numpy(int)]
        adjacency = reweight_candidates(candidates, len(current), config)
        nodes = current[["permco", "permno"]].copy()
        nodes.index.name = "node_index"
        nodes["neighbor_count"] = adjacency.getnnz(axis=1)
        metadata = {
            "month": str(month.date()),
            # Characteristics and text are those known at the month-end.
            "characteristic_cutoff": str(month.date()),
            "nodes": len(nodes),
            "edges": int(adjacency.nnz),
            "candidate_edges": len(candidates),
        }
        return MonthlyGraph(month, nodes, adjacency, metadata)


def _standardize(frame: pd.DataFrame, quantile: float) -> pd.DataFrame:
    x = frame.astype(float).replace([np.inf, -np.inf], np.nan)
    x = x.clip(lower=x.quantile(quantile), upper=x.quantile(1 - quantile), axis=1)
    scale = x.std(ddof=0)
    return (x - x.mean()) / scale.where(scale > 1e-12)


def _sector_labels(snapshot: pd.DataFrame) -> pd.Series:
    gics = snapshot.get("gics", pd.Series(pd.NA, index=snapshot.index))
    gics = gics.astype("string").str.replace(r"\.0$", "", regex=True)
    valid = gics.str.fullmatch(r"[0-9]{2,8}") & ~gics.str.startswith("00")
    sector = gics.where(valid).str[:2]
    ff = pd.to_numeric(
        snapshot.get("ff49", pd.Series(np.nan, index=snapshot.index)),
        errors="coerce",
    )
    fallback = ff.where(ff.between(1, 49)).map(
        lambda x: f"ff49:{int(x)}" if pd.notna(x) else pd.NA
    )
    return sector.fillna(fallback).astype("string")


def _sector_standardize(
    frame: pd.DataFrame, snapshot: pd.DataFrame, config: GraphConfig
) -> pd.DataFrame:
    """Winsorizes and standardizes fundamentals within each sector."""
    result = _standardize(frame, config.winsor_quantile)
    sectors = _sector_labels(snapshot)
    for _, positions in sectors.groupby(sectors, dropna=True).groups.items():
        positions = list(positions)
        if len(positions) < config.min_sector_observations:
            continue
        local = _standardize(frame.loc[positions], config.winsor_quantile)
        for column in frame.columns:
            if local[column].notna().sum() >= config.min_sector_observations:
                result.loc[positions, column] = local[column]
    return result


def _hashed_tfidf(documents: list[str], dimensions: int) -> sparse.csr_matrix:
    """Deterministic hashed word and bigram TF-IDF with unit-length rows."""
    rows, columns, values = [], [], []
    for row, document in enumerate(documents):
        tokens = [
            token
            for token in _TOKEN.findall((document or "").lower())
            if token not in _TEXT_STOPWORDS and not token.isdigit()
        ]
        bigrams = zip(tokens, tokens[1:], strict=False)
        terms = tokens + [f"{a}_{b}" for a, b in bigrams]
        counts = collections.Counter(
            int.from_bytes(
                hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest(),
                "little",
            )
            % dimensions
            for term in terms
        )
        for column, count in counts.items():
            rows.append(row)
            columns.append(column)
            values.append(1.0 + np.log(count))
    matrix = sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(len(documents), dimensions),
        dtype=float,
    )
    if matrix.nnz == 0:
        return matrix
    document_frequency = np.bincount(matrix.indices, minlength=dimensions)
    idf = np.log((1.0 + len(documents)) / (1.0 + document_frequency)) + 1.0
    matrix = matrix @ sparse.diags(idf)
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    inverse = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
    return (sparse.diags(inverse) @ matrix).tocsr()


def _fundamental_similarity(
    z: np.ndarray, start: int, stop: int, config: GraphConfig
) -> np.ndarray:
    mask = np.isfinite(z).astype(float)
    values = np.nan_to_num(z, nan=0.0)
    overlap = mask[start:stop] @ mask.T
    distances = (
        (values[start:stop] ** 2) @ mask.T
        + mask[start:stop] @ (values**2).T
        - 2 * values[start:stop] @ values.T
    )
    mean_distance = np.divide(
        np.maximum(distances, 0),
        overlap,
        out=np.zeros_like(distances),
        where=overlap > 0,
    )
    informative = max(int(np.isfinite(z).any(axis=0).sum()), 1)
    similarity = np.exp(-mean_distance / (2 * config.fundamental_bandwidth**2))
    similarity *= overlap / informative
    similarity[overlap < config.min_fundamental_overlap] = 0.0
    return np.clip(similarity, 0, 1)


def _industry_similarity(snapshot: pd.DataFrame, start: int, stop: int) -> np.ndarray:
    """Hierarchical GICS; FF49, then SIC, when a pair lacks usable GICS."""
    n = len(snapshot)
    gics = snapshot.get("gics", pd.Series(pd.NA, index=snapshot.index))
    gics = gics.astype("string").str.replace(r"\.0$", "", regex=True)
    codes = [
        gics.where(
            gics.str.fullmatch(r"[0-9]{" + str(length) + r",8}")
            & ~gics.str.startswith("00")
        )
        .str[:length]
        .fillna("")
        .to_numpy(str)
        for length in (2, 4, 6, 8)
    ]
    result = np.zeros((stop - start, n))
    for code, score in zip(codes, (0.25, 0.50, 0.75, 1.0), strict=True):
        same = (code[start:stop, None] == code[None, :]) & (
            code[start:stop, None] != ""
        )
        result = np.maximum(result, same * score)
    valid_gics = codes[0] != ""
    fallback = ~(valid_gics[start:stop, None] & valid_gics[None, :])
    ff = pd.to_numeric(
        snapshot.get("ff49", pd.Series(np.nan, index=snapshot.index)),
        errors="coerce",
    )
    ff = ff.where(ff.between(1, 49)).to_numpy(float)
    ff_pair = np.isfinite(ff[start:stop, None]) & np.isfinite(ff[None, :])
    result = np.where(
        fallback & ff_pair,
        (ff[start:stop, None] == ff[None, :]) * 0.75,
        result,
    )
    sic = pd.to_numeric(
        snapshot.get("sic", pd.Series(np.nan, index=snapshot.index)),
        errors="coerce",
    )
    sic = sic.where(sic.between(100, 9999)).to_numpy(float)
    sic_score = (sic[start:stop, None] // 100 == sic[None, :] // 100) * 0.50
    sic_score = np.maximum(sic_score, (sic[start:stop, None] == sic[None, :]) * 0.75)
    return np.where(fallback & ~ff_pair, sic_score, result)


def _return_similarity(
    history: np.ndarray, start: int, stop: int, minimum: int
) -> np.ndarray:
    """Pairwise-complete positive return correlation; gaps stay missing."""
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
    return np.where(valid, np.clip(corr, 0, 1), 0.0)
