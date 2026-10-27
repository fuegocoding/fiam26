"""Observable, agent-declared cohorts shared by fitting, scoring and trading."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator

from ..contracts.base import StrictModel, canonical_digest
from .input_permissions import COMPETITION_CHARACTERISTICS


class ObservableRange(StrictModel):
    feature: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    minimum: float | None = Field(default=None, allow_inf_nan=False)
    maximum: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered_bounds(self):
        if self.minimum is None and self.maximum is None:
            raise ValueError("A cohort filter needs at least one bound")
        if self.minimum is not None and self.maximum is not None and self.minimum >= self.maximum:
            raise ValueError("Cohort minimum must be below maximum")
        return self


class StrategyUniverse(StrictModel):
    rationale: str = Field(min_length=20, max_length=600)
    min_market_cap_usd: float = Field(default=0, ge=0, allow_inf_nan=False)
    max_market_cap_usd: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    gics_prefixes: list[str] = Field(default_factory=list, max_length=32)
    filters: list[ObservableRange] = Field(default_factory=list, max_length=8)

    @field_validator("gics_prefixes")
    @classmethod
    def normalized_sectors(cls, values):
        return sorted(set(values))

    @field_validator("filters")
    @classmethod
    def normalized_filters(cls, values):
        return sorted(values, key=lambda f: f.feature)

    @model_validator(mode="after")
    def observable_scope(self):
        if self.max_market_cap_usd is not None and self.max_market_cap_usd <= self.min_market_cap_usd:
            raise ValueError("Market-cap maximum must exceed minimum")
        if any(len(s) not in {2, 4, 6, 8} or not s.isdigit() for s in self.gics_prefixes):
            raise ValueError("Use two-, four-, six- or eight-digit GICS prefixes")
        if len({f.feature for f in self.filters}) != len(self.filters):
            raise ValueError("Combine bounds for a feature into one cohort filter")
        return self


def cohort_mask(frame, universe):
    """Use raw contemporaneous characteristics, never outcomes or future members."""
    cap = frame.me.to_numpy(dtype=float) * 1e6
    mask = np.isfinite(cap) & (cap >= universe.min_market_cap_usd)
    if universe.max_market_cap_usd is not None:
        mask &= cap < universe.max_market_cap_usd
    if universe.gics_prefixes:
        sectors = frame.gics.fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
        mask &= sectors.str.startswith(tuple(universe.gics_prefixes)).to_numpy()
    for condition in universe.filters:
        # Frame columns are source characteristics. Targets, identifiers and
        # labels are deliberately not part of the cohort language.
        if condition.feature not in COMPETITION_CHARACTERISTICS or condition.feature not in frame:
            raise ValueError(f"Cohort filter requires an observable source characteristic: {condition.feature}")
        values = frame[condition.feature].to_numpy(dtype=float)
        mask &= np.isfinite(values)
        if condition.minimum is not None:
            mask &= values >= condition.minimum
        if condition.maximum is not None:
            mask &= values < condition.maximum
    return mask


def scoped_panel(panel, universe):
    signature = canonical_digest(universe.model_dump(mode="json"))
    if panel.metadata.get("strategy_universe_digest") == signature:
        return panel
    cache = getattr(panel, "_strategy_scope_cache", {})
    if signature in cache:
        return cache[signature]
    frames = panel.frames
    missing = {condition.feature for condition in universe.filters if condition.feature not in frames[0]}
    if not missing <= set(COMPETITION_CHARACTERISTICS):
        raise ValueError("Cohort filters may use only observable source characteristics")
    if missing:
        source = panel.metadata.get("raw_characteristic_source")
        if source is None:
            raise ValueError("Raw source columns are required to evaluate the declared cohort filters")
        raw = pd.read_parquet(source, columns=["eom", "permno", *sorted(missing)],
            filters=[("eom", ">=", panel.dates[0]), ("eom", "<=", panel.dates[-1])])
        raw["eom"] = pd.to_datetime(raw.eom)
        values = {name: raw.pivot(index="eom", columns="permno", values=name)
            .reindex(index=panel.dates, columns=panel.securities).to_numpy(dtype=np.float32)
            for name in missing}
        frames = [frame.assign(**{name: array[t] for name, array in values.items()})
            for t, frame in enumerate(frames)]
    masks = np.stack([cohort_mask(f, universe) for f in frames]) & panel.mask
    result = replace(panel, mask=masks, frames=frames, metadata={**panel.metadata,
        "strategy_universe_digest": signature,
        "strategy_universe": universe.model_dump(mode="json")})
    cache[signature] = result
    panel._strategy_scope_cache = cache
    return result


def scoped_design(design, frame, universe):
    """Enforce membership even when a caller supplies a broad risk design."""
    from .training_search import RiskDesign

    if getattr(design, "declared_universe_digest", None) == canonical_digest(universe.model_dump(mode="json")):
        return design
    mask = design.mask & cohort_mask(frame, universe)
    if np.array_equal(mask, design.mask):
        return design
    local = mask[design.mask]
    matrix = design.matrix[local]
    return RiskDesign(mask, matrix, np.linalg.pinv(matrix), design.shortable[local])


def scope_diagnostics(panel, universe, formations):
    """Report coverage and, for an empty scope, aggregate source-unit ranges."""
    formations = np.asarray(formations, dtype=int)
    counts = panel.mask[formations].sum(axis=1)
    result = {"minimum_names": int(counts.min()) if len(counts) else 0,
        "mean_names": float(counts.mean()) if len(counts) else 0.0,
        "maximum_names": int(counts.max()) if len(counts) else 0,
        "formation_months": len(counts)}
    if not counts.any() and universe.filters:
        bounds = universe.model_copy(update={"filters": []})
        ranges = {}
        for condition in universe.filters:
            parts = [panel.frames[t].loc[cohort_mask(panel.frames[t], bounds), condition.feature]
                .to_numpy(dtype=float) for t in formations]
            values = np.concatenate(parts) if parts else np.array([])
            values = values[np.isfinite(values)]
            ranges[condition.feature] = dict(zip(("p05", "median", "p95"),
                np.quantile(values, [.05, .5, .95]).tolist())) if len(values) else None
        result["observed_raw_filter_ranges"] = ranges
    return result
