"""Contemporary LLM hypotheses, capacity curves and permanent pod research books.

Models propose bounded economic recipes. Supplied pre-2021 outcomes determine
their evidence, and evidence determines funding independently of pod ownership.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from pod_platform.contracts.base import StrictModel
from pod_platform.research.training_search import (
    RiskDesign,
    make_weights,
    smooth_signal,
)

from .input_permissions import AVAILABLE_RESEARCH_FEATURES


class SignalTerm(StrictModel):
    feature: str
    other_feature: str | None
    operation: Literal["identity", "product", "difference"]
    sign: Literal[-1, 1]

    @model_validator(mode="after")
    def supported(self):
        allowed = AVAILABLE_RESEARCH_FEATURES
        if (
            self.feature not in allowed
            or self.other_feature is not None
            and self.other_feature not in allowed
        ):
            raise ValueError(
                "Only supplied, named training characteristics are executable"
            )
        if (self.operation == "identity") != (self.other_feature is None):
            raise ValueError(
                "Identity uses no second feature; binary operators require one"
            )
        return self


class StrategyPlan(StrictModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    economic_mechanism: str = Field(min_length=30, max_length=1200)
    falsification: str = Field(min_length=20, max_length=800)
    terms: list[SignalTerm] = Field(min_length=1, max_length=6)
    review_months: Literal[1, 3, 6]
    smoothing_months: Literal[1, 3, 6]
    controls: Literal["dollar", "sector", "sector_beta"]
    entry_rank: float = Field(ge=0, le=0.9)
    exit_rank: float = Field(ge=0, le=0.5)
    expected_holding_months: int = Field(ge=1, le=12)

    @model_validator(mode="after")
    def thresholds(self):
        if self.entry_rank and self.exit_rank >= self.entry_rank:
            raise ValueError("Exit threshold must be below entry threshold")
        if not self.entry_rank and self.exit_rank:
            raise ValueError(
                "Continuous sizing has no membership exit threshold"
            )
        return self


class PodPlans(StrictModel):
    diagnosis: str = Field(min_length=30, max_length=1600)
    plans: list[StrategyPlan] = Field(min_length=1, max_length=3)


PLAN_PROMPT = """You are a quantitative pod researcher. Return 1–3 economically distinct,
falsifiable stock strategies in the exact JSON schema. The pod must own research
strategies even when its allocated capital is zero. Never claim that a proposal
is profitable or validated. Do not fill a slot with a losing rule labeled good.
Diagnose signal, turnover, capacity and construction separately. Pick mechanism
and holding period before looking for profitable parameter choices. The host
measures return edge per unit of exposure without a dollar capital assumption.
Only afterward a continuous allocator funds the strategy until marginal edge,
liquidity, impact or borrow constraints bind. There is no preset capital grid.
You cannot approve capital.
All 147 supplied characteristics are available across pods. Default pod feature
lists are examples, not access restrictions. Follow the pod's economic mission,
and use any available characteristic or registered research feature that helps
express it. Supplied 8-K metadata and text are available through the dated local
filing reader; disclose the corpus's survivor bias and uncertain historical maps.
Ranked numeric features are centered in [-1,1]. A product is an interaction
of two centered ranks, not a Boolean AND. An identity term has other_feature=null.
Use sign +1 for positively rewarded evidence, -1 for negatively rewarded evidence.
Different risk controls express different economic hypotheses: sector_beta removes
both; sector permits deliberate beta exposure; dollar permits deliberate sector
and beta exposures but keeps net zero. State that exposure in the mechanism.
Every plan needs a concrete falsification test, sensible review/holding period,
and entry/exit ranks (0/0 means continuous sizing). Missing event/analyst/flow
data cannot be invented. Filing access is research access, not evidence of an
unbiased text corpus or financial admission. No security IDs, historical winners, dates of known crashes, arbitrary
code, file paths or additional data sources. This is contemporary research on
2015–2020 development data. Never describe it as an untouched historical test.
"""


def construction_designs(panel, controls):
    """Apply the plan's declared controls without modifying forecast ranks."""
    designs = []
    for frame, mask in zip(panel.frames, panel.mask):
        local = frame.loc[mask]
        if controls == "dollar":
            matrix = np.ones((len(local), 1))
        else:
            sectors = local.gics.fillna("unknown").astype(str).str[:2]
            matrix = pd.get_dummies(sectors, dtype=float).to_numpy()
            if controls == "sector_beta":
                beta = local.beta_60m.fillna(1).to_numpy(dtype=float)
                beta = (beta - beta.mean()) / max(beta.std(), 1e-8)
                matrix = np.column_stack([matrix, beta])
        designs.append(
            RiskDesign(
                mask,
                matrix,
                np.linalg.pinv(matrix),
                local.me.to_numpy() >= 500,
                np.ones(len(local)),
                np.ones(len(local)),
            )
        )
    return designs


def plan_signal(plan, panel, *, apply_thresholds=True):
    """Causal signal and membership, before any position count or capital choice."""
    terms = []
    for term in plan.terms:
        first = panel.features[term.feature]
        if term.operation == "identity":
            value = first.copy()
        elif term.operation == "product":
            value = first * panel.features[term.other_feature]
        else:
            value = first - panel.features[term.other_feature]
        terms.append(term.sign * value)
    stack = np.stack(terms)
    count = np.isfinite(stack).sum(axis=0)
    values = np.divide(
        np.nansum(stack, axis=0),
        count,
        out=np.full(stack.shape[1:], np.nan),
        where=count > 0,
    )
    values = smooth_signal(values, plan.smoothing_months)
    if plan.entry_rank and apply_thresholds:
        ranks = pd.DataFrame(values).rank(axis=1, pct=True).to_numpy() * 2 - 1
        previous = np.zeros(values.shape[1])
        for t, score in enumerate(ranks):
            previous[(previous > 0) & (score < plan.exit_rank)] = 0
            previous[(previous < 0) & (score > -plan.exit_rank)] = 0
            previous[score >= plan.entry_rank] = 1
            previous[score <= -plan.entry_rank] = -1
            previous[~panel.mask[t] | ~np.isfinite(score)] = 0
            values[t] = np.where(
                previous > 0,
                np.maximum(score, plan.exit_rank),
                np.where(previous < 0, np.minimum(score, -plan.exit_rank), 0),
            )
    return values


def plan_weights(plan, panel, settings):
    """Construct unit weights from the independently defined signal."""
    values = plan_signal(plan, panel)
    return make_weights(
        values, construction_designs(panel, plan.controls), settings
    )
