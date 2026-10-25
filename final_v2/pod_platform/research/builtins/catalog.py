"""Reviewed recipe catalog: the only feature/target/baseline definitions an executor can run.

A recipe is a fixed, reviewed combination of cross-sectional transforms of operator-declared columns. LLM
parameters pick a recipe id and one point from a tiny grid (method, feature set, ridge penalty); they never add
features, change targets or reach code. Transforms are rank/lag/change/growth/sector-rank/rank-product only, and
every transform uses the same month or earlier months (never later rows).

Strategy bridge honesty: each recipe's belief target is evaluated against a declared cheap baseline, and the
return bridge (next-month excess-return IC, tail spread, costs, turnover) is evaluated separately.
"""

from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ...contracts.base import canonical_digest
from ...strategies.base import Charter
from .errors import BuiltinConfigError

LABEL = "@label"  # symbolic: the dataset's declared label column
FEATURE_OPS = (
    "rank",
    "lag_rank",
    "change_rank",
    "growth_rank",
    "sector_rank",
    "rank_product",
)
TARGET_KINDS = ("lead", "cum_lead")
METHODS = ("ridge", "rank_blend")
RIDGE_LAMBDAS = (0.01, 0.05, 0.25, 1.0)
DEFAULT_RIDGE_LAMBDA = 0.05
ALLOWED_PARAMS = frozenset(
    {
        "dataset_id",
        "recipe_id",
        "method_id",
        "feature_set",
        "ridge_lambda",
        "partition",
    }
)
_IDENT = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    op: str
    column: str
    column2: Optional[str] = None
    months: int = 0
    sign: int = (
        1  # ex-ante direction used by rank_blend; ridge learns its own signs
    )


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    kind: str
    column: str  # physical column or LABEL
    months: int
    demean: str  # "none" | "sector"
    description: str


@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    charter: str
    mechanism_family: str
    summary: str
    target: TargetSpec
    baseline: FeatureSpec
    baseline_description: str
    feature_sets: tuple[tuple[str, tuple[FeatureSpec, ...]], ...]
    bridge_sign: int
    declared_direction: str
    proxy_labels: tuple[str, ...]
    limitations: tuple[str, ...]

    def features(self, feature_set: str) -> tuple[FeatureSpec, ...]:
        for name, feats in self.feature_sets:
            if name == feature_set:
                return feats
        raise BuiltinConfigError(
            f"recipe {self.recipe_id} has no feature_set {feature_set!r}; known: {[n for n, _ in self.feature_sets]}"
        )

    def feature_set_names(self) -> tuple[str, ...]:
        return tuple(n for n, _ in self.feature_sets)

    def digest(self) -> str:
        return canonical_digest(dataclasses.asdict(self))

    def uses_sector(self) -> bool:
        used = [f.op for _, feats in self.feature_sets for f in feats] + [
            self.baseline.op
        ]
        return "sector_rank" in used or self.target.demean == "sector"

    def logical_columns(self, label_column: str) -> set[str]:
        """Every column the recipe reads, before operator aliases are applied (label included for the bridge)."""
        cols = {label_column}
        specs = [self.baseline] + [
            f for _, feats in self.feature_sets for f in feats
        ]
        for spec in specs:
            cols.add(spec.column)
            if spec.column2:
                cols.add(spec.column2)
        cols.add(
            label_column if self.target.column == LABEL else self.target.column
        )
        return cols


def _fs(
    name: str,
    op: str,
    column: str,
    sign: int = 1,
    *,
    column2: Optional[str] = None,
    months: int = 0,
) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        op=op,
        column=column,
        column2=column2,
        months=months,
        sign=sign,
    )


def _issuance_recipe(column: str) -> Recipe:
    return Recipe(
        recipe_id=f"events_net_issuance_{column.split('_')[0]}_v1",
        charter=Charter.CORPORATE_EVENTS.value,
        mechanism_family="net_issuance",
        summary=f"Does {column} plus valuation and run-up context forecast issuance intensity six months ahead better than current intensity alone?",
        target=TargetSpec(
            f"{column}_lead_6m",
            "lead",
            column,
            6,
            "none",
            f"panel {column} six monthly rows ahead (future financing/dilution intensity)",
        ),
        baseline=_fs(f"{column}_magnitude_only", "rank", column),
        baseline_description="size-normalized event magnitude alone (current net-issuance intensity rank)",
        feature_sets=(
            (
                "core",
                (
                    _fs(f"{column}_rank", "rank", column, 1),
                    _fs("be_me_rank", "rank", "be_me", -1),
                    _fs("ret_12_1_rank", "rank", "ret_12_1", 1),
                ),
            ),
            (
                "extended",
                (
                    _fs(f"{column}_rank", "rank", column, 1),
                    _fs("be_me_rank", "rank", "be_me", -1),
                    _fs("ret_12_1_rank", "rank", "ret_12_1", 1),
                    _fs("ni_me_rank", "rank", "ni_me", -1),
                    _fs(f"{column}_chg6", "change_rank", column, 1, months=6),
                ),
            ),
        ),
        bridge_sign=-1,
        declared_direction="higher predicted issuance intensity implies LOWER next-month excess return (dilution underappreciated); bridge sign -1",
        proxy_labels=(
            "MECHANISM_TRANSFER: panel net-issuance intensity stands in for discrete announced/completed issuance events; filing dates, event type and announcement-vs-completion are not observed",
            "PROXY: belief target is the panel measure six rows ahead, not a timestamped financing event",
        ),
        limitations=(
            "Unsigned financing intensity is given a direction only by the declared hypothesis sign; it is not an event-study.",
            "Issuance, size and run-up are correlated with common styles; exposure_* metrics must be read before any claim of independence.",
        ),
    )


RECIPES_LIST: tuple[Recipe, ...] = (
    Recipe(
        recipe_id="fund_earnings_innovation_v1",
        charter=Charter.FUNDAMENTALS_EXPECTATIONS.value,
        mechanism_family="earnings_surprise",
        summary="Do sales growth and earnings yield add to surprise persistence when forecasting the next earnings innovation?",
        target=TargetSpec(
            "niq_su_lead_3m",
            "lead",
            "niq_su",
            3,
            "none",
            "panel niq_su three monthly rows ahead (next reported earnings innovation proxy)",
        ),
        baseline=_fs("niq_su_persistence", "rank", "niq_su"),
        baseline_description="seasonal-random-walk surprise persistence (current niq_su rank)",
        feature_sets=(
            (
                "core",
                (
                    _fs("niq_su_rank", "rank", "niq_su", 1),
                    _fs("sale_gr1_rank", "rank", "sale_gr1", 1),
                    _fs("ni_me_rank", "rank", "ni_me", 1),
                ),
            ),
            (
                "extended",
                (
                    _fs("niq_su_rank", "rank", "niq_su", 1),
                    _fs("sale_gr1_rank", "rank", "sale_gr1", 1),
                    _fs("ni_me_rank", "rank", "ni_me", 1),
                    _fs("niq_su_chg3", "change_rank", "niq_su", 1, months=3),
                    _fs("ret_12_1_rank", "rank", "ret_12_1", 1),
                ),
            ),
        ),
        bridge_sign=1,
        declared_direction="higher predicted next earnings innovation implies higher next-month excess return; bridge sign +1",
        proxy_labels=(
            "PROXY: the target is the panel's niq_su three monthly rows ahead; the true release/acceptance timestamp is not in the panel",
            "PROXY: niq_su is a seasonal-random-walk surprise, not an analyst-consensus surprise",
        ),
        limitations=(
            "Announcement-reaction baseline and guidance/revision fields are not available in this panel.",
            "Post-announcement drift and pre-report forecasting are distinct models; only the pre-report belief is evaluated here.",
        ),
    ),
    _issuance_recipe("eqnetis_at"),
    _issuance_recipe("netis_at"),
    Recipe(
        recipe_id="valuation_residual_return_v1",
        charter=Charter.VALUATION_RELATIVE_VALUE.value,
        mechanism_family="value_spread",
        summary="Does valuation plus earnings yield forecast three-month sector-residual returns beyond sector-relative value alone?",
        target=TargetSpec(
            "sector_residual_excess_return_3m",
            "cum_lead",
            LABEL,
            3,
            "sector",
            "sum of three monthly excess returns after the formation month, demeaned within sector",
        ),
        baseline=_fs("sector_relative_value", "sector_rank", "be_me"),
        baseline_description="sector-relative book-to-market rank (cheapest common valuation control)",
        feature_sets=(
            (
                "core",
                (
                    _fs("be_me_rank", "rank", "be_me", 1),
                    _fs("ni_me_rank", "rank", "ni_me", 1),
                    _fs("be_me_sector_rank", "sector_rank", "be_me", 1),
                ),
            ),
            (
                "extended",
                (
                    _fs("be_me_rank", "rank", "be_me", 1),
                    _fs("ni_me_rank", "rank", "ni_me", 1),
                    _fs("be_me_sector_rank", "sector_rank", "be_me", 1),
                    _fs("niq_su_rank", "rank", "niq_su", 1),
                    _fs("sale_gr1_rank", "rank", "sale_gr1", 1),
                ),
            ),
        ),
        bridge_sign=1,
        declared_direction="higher predicted residual return implies higher next-month excess return; bridge sign +1",
        proxy_labels=(
            "PROXY: the belief target is a return target (sum of monthly excess returns, not compounded); the charter's profitability-persistence targets are in valuation_cash_conversion_v1",
            "PROXY: sector peers are the first sector_digits of the panel's gics code, not economically matched peers",
        ),
        limitations=(
            "A cheap multiple is not proof of misvaluation; this recipe cannot separate value traps from mispricing.",
            "Belief and bridge share the first forward month; they differ in residualization, horizon, sample rules, costs and short-eligibility.",
        ),
    ),
    Recipe(
        recipe_id="valuation_cash_conversion_v1",
        charter=Charter.VALUATION_RELATIVE_VALUE.value,
        mechanism_family="intrinsic_value",
        summary="Do cash-flow level, earnings yield and book-to-market forecast six-month-ahead operating cash flow better than persistence?",
        target=TargetSpec(
            "ocf_at_lead_6m",
            "lead",
            "ocf_at",
            6,
            "none",
            "panel ocf_at six monthly rows ahead (cash-conversion persistence)",
        ),
        baseline=_fs("ocf_at_persistence", "rank", "ocf_at"),
        baseline_description="current operating cash flow to assets rank",
        feature_sets=(
            (
                "core",
                (
                    _fs("ocf_at_rank", "rank", "ocf_at", 1),
                    _fs("ni_me_rank", "rank", "ni_me", 1),
                    _fs("be_me_rank", "rank", "be_me", 1),
                ),
            ),
            (
                "extended",
                (
                    _fs("ocf_at_rank", "rank", "ocf_at", 1),
                    _fs("ni_me_rank", "rank", "ni_me", 1),
                    _fs("be_me_rank", "rank", "be_me", 1),
                    _fs("ocf_at_chg6", "change_rank", "ocf_at", 1, months=6),
                    _fs("sale_gr1_rank", "rank", "sale_gr1", 1),
                ),
            ),
        ),
        bridge_sign=1,
        declared_direction="higher predicted cash conversion implies higher next-month excess return; bridge sign +1",
        proxy_labels=(
            "PROXY: ocf_at is a trailing accounting ratio carried monthly; its panel date is not the filing timestamp",
        ),
        limitations=(
            "A good company is not necessarily an underpriced stock; the bridge tests that translation separately.",
        ),
    ),
    Recipe(
        recipe_id="dynamics_reversal_liquidity_v1",
        charter=Charter.MARKET_DYNAMICS.value,
        mechanism_family="short_interest_liquidity",
        summary="Do prior-month return, 12-1 momentum and dollar volume forecast next-month sector-residual returns beyond simple reversal?",
        target=TargetSpec(
            "sector_residual_excess_return_1m",
            "cum_lead",
            LABEL,
            1,
            "sector",
            "next-month excess return demeaned within sector",
        ),
        baseline=_fs("short_term_reversal", "rank", "ret_1_0", -1),
        baseline_description="simple short-term reversal (negative prior-month return rank)",
        feature_sets=(
            (
                "core",
                (
                    _fs("ret_1_0_rank", "rank", "ret_1_0", -1),
                    _fs("ret_12_1_rank", "rank", "ret_12_1", 1),
                    _fs("dolvol_126d_rank", "rank", "dolvol_126d", -1),
                ),
            ),
            (
                "extended",
                (
                    _fs("ret_1_0_rank", "rank", "ret_1_0", -1),
                    _fs("ret_12_1_rank", "rank", "ret_12_1", 1),
                    _fs("dolvol_126d_rank", "rank", "dolvol_126d", -1),
                    _fs(
                        "reversal_x_volume",
                        "rank_product",
                        "ret_1_0",
                        -1,
                        column2="dolvol_126d",
                    ),
                ),
            ),
        ),
        bridge_sign=1,
        declared_direction="higher predicted residual return implies higher next-month excess return; bridge sign +1",
        proxy_labels=(
            "MECHANISM_TRANSFER: the monthly panel horizon stands in for the charter's 5- and 20-day dynamics; same-day leakage cannot occur but intramonth pressure is not observed",
            "PROXY: dolvol_126d is a trailing dollar-volume average, not an intraday liquidity-shock measure",
        ),
        limitations=(
            "Microstructure artifacts and untradeable turnover are only approximated by the flat one-way cost.",
            "Industry/network lead-lag and ETF-flow mechanisms are out of scope for this recipe.",
        ),
    ),
)

RECIPES: dict[str, Recipe] = {r.recipe_id: r for r in RECIPES_LIST}
RECIPES_BY_CHARTER: dict[str, tuple[Recipe, ...]] = {
    c.value: tuple(r for r in RECIPES_LIST if r.charter == c.value)
    for c in Charter
}
DEFAULT_RECIPE: dict[str, str] = {
    Charter.FUNDAMENTALS_EXPECTATIONS.value: "fund_earnings_innovation_v1",
    Charter.CORPORATE_EVENTS.value: "events_net_issuance_eqnetis_v1",
    Charter.VALUATION_RELATIVE_VALUE.value: "valuation_residual_return_v1",
    Charter.MARKET_DYNAMICS.value: "dynamics_reversal_liquidity_v1",
}


@dataclass(frozen=True)
class VariantConfig:
    dataset_id: Optional[str]
    recipe_id: str
    method_id: str
    feature_set: str
    ridge_lambda: Optional[float]
    partition: str = "design"

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def parse_variant_params(
    params: Mapping[str, Any], charter: str
) -> VariantConfig:
    """Whitelist-parse LLM/campaign scalars. Unknown keys, paths and out-of-grid values are rejected."""
    unknown = sorted(set(params) - ALLOWED_PARAMS)
    if unknown:
        raise BuiltinConfigError(
            f"unsupported variant parameter(s) {unknown}; allowed: {sorted(ALLOWED_PARAMS)}"
        )
    for key in (
        "dataset_id",
        "recipe_id",
        "method_id",
        "feature_set",
        "partition",
    ):
        value = params.get(key)
        if value is not None and (
            not isinstance(value, str) or not _IDENT.match(value)
        ):
            raise BuiltinConfigError(
                f"parameter {key} must be a plain identifier (paths and code are never accepted)"
            )
    recipe_id = params.get("recipe_id") or DEFAULT_RECIPE[charter]
    recipe = RECIPES.get(recipe_id)
    if recipe is None or recipe.charter != charter:
        raise BuiltinConfigError(
            f"recipe {recipe_id!r} is not a reviewed recipe of this executor's charter"
        )
    method = params.get("method_id", "ridge")
    if method not in METHODS:
        raise BuiltinConfigError(f"method_id must be one of {METHODS}")
    feature_set = params.get("feature_set", "core")
    recipe.features(feature_set)  # validates
    lam: Optional[float] = None
    if method == "ridge":
        raw = params.get("ridge_lambda", DEFAULT_RIDGE_LAMBDA)
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(raw)
        ):
            raise BuiltinConfigError(
                "ridge_lambda must be a finite number from the reviewed grid"
            )
        match = [g for g in RIDGE_LAMBDAS if abs(g - float(raw)) < 1e-12]
        if not match:
            raise BuiltinConfigError(
                f"ridge_lambda must be one of {RIDGE_LAMBDAS}"
            )
        lam = match[0]
    elif "ridge_lambda" in params:
        raise BuiltinConfigError(
            "ridge_lambda is only valid with method_id=ridge"
        )
    return VariantConfig(
        dataset_id=params.get("dataset_id"),
        recipe_id=recipe_id,
        method_id=method,
        feature_set=feature_set,
        ridge_lambda=lam,
        partition=params.get("partition", "design"),
    )
