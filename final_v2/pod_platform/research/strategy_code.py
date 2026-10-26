"""Inspectable research code with a deliberately small, IO-free Python surface.

The evaluator supplies only observations known at the decision close. Outcomes,
capital, credentials, filesystem and provider clients never enter this runtime.
"""

from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field

from ..contracts.base import StrictModel
from .filing_text import FilingMeasurement
from .pod_plans import construction_designs
from .training_search import make_weights
from .training_tactical import threshold_positions

FUNCTIONS = {
    name: getattr(np, name)
    for name in (
        "abs",
        "sign",
        "sqrt",
        "log1p",
        "minimum",
        "maximum",
        "clip",
        "where",
        "isfinite",
        "nan_to_num",
        "nanmean",
        "nanstd",
        "nansum",
        "mean",
        "std",
        "sum",
        "diff",
        "tanh",
        "exp",
    )
}
SAFE_NP = SimpleNamespace(**FUNCTIONS)


class CodeStrategy(StrictModel):
    pod: Literal["fundamentals", "events", "valuation", "dynamics"]
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    mechanism: str = Field(min_length=30, max_length=1800)
    falsification: str = Field(min_length=20, max_length=1000)
    evidence_refs: list[str] = Field(default_factory=list, max_length=16)
    code: str = Field(min_length=20, max_length=12000)
    controls: Literal["dollar", "sector", "sector_beta"]
    review: Literal["daily", "weekly", "monthly"]
    signal_update_months: int = Field(default=1, ge=1, le=12)
    entry_rank: float = Field(default=0, ge=0, le=0.95)
    exit_rank: float = Field(default=0, ge=0, le=0.9)
    expected_holding_days: int = Field(ge=1, le=504)


class CodeBatch(StrictModel):
    diagnosis: str = Field(min_length=20, max_length=2000)
    strategies: list[CodeStrategy] = Field(min_length=1, max_length=3)
    filing_measurements: list[FilingMeasurement] = Field(
        default_factory=list, max_length=12
    )


def compile_strategy(source, feature_names):
    """Validate syntax before compilation; no imports, loops, attributes or IO.

    Mathematical assignments and return expressions are actual agent-written
    Python. Attributes are permitted exclusively on the finite numpy facade.
    This is a language restriction, not a claim of an OS process sandbox.
    """
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Supply only def score(features, history, daily)")
    function = tree.body[0]
    if (
        function.name != "score"
        or [a.arg for a in function.args.args]
        != ["features", "history", "daily"]
        or function.args.posonlyargs
        or function.args.kwonlyargs
        or function.args.vararg
        or function.args.kwarg
        or function.args.defaults
        or function.decorator_list
        or function.returns is not None
        or any(a.annotation is not None for a in function.args.args)
    ):
        raise ValueError(
            "Use the exact score(features, history, daily) signature"
        )
    allowed = (
        ast.Module,
        ast.FunctionDef,
        ast.arguments,
        ast.arg,
        ast.Assign,
        ast.Return,
        ast.Expr,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.Store,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.BoolOp,
        ast.IfExp,
        ast.Subscript,
        ast.Slice,
        ast.Tuple,
        ast.List,
        ast.Call,
        ast.keyword,
        ast.Attribute,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.USub,
        ast.UAdd,
        ast.Invert,
        ast.BitAnd,
        ast.BitOr,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Eq,
        ast.NotEq,
        ast.And,
        ast.Or,
        ast.Not,
    )
    assigned = {"features", "history", "daily", "np"}
    if len(list(ast.walk(tree))) > 1000:
        raise ValueError("Strategy exceeds syntax resource bound")
    for statement in function.body:
        if not isinstance(statement, (ast.Assign, ast.Return, ast.Expr)):
            raise ValueError("Use assignments and a return expression")
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(
                statement.targets[0], ast.Name
            ):
                raise ValueError("Assignments must create local names")
            name = statement.targets[0].id
            if name in assigned or name.startswith("_"):
                raise ValueError("Do not overwrite inputs or existing locals")
            assigned.add(name)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(
                f"Unsupported strategy syntax: {type(node).__name__}"
            )
        if isinstance(node, ast.Name) and node.id not in assigned:
            raise ValueError(f"Unknown strategy name: {node.id}")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str) and node.value not in feature_names:
                raise ValueError(f"Unknown research feature: {node.value}")
            if isinstance(node.value, (float, int)) and abs(node.value) > 10000:
                raise ValueError("Excessive numeric literal")
        if isinstance(node, ast.Attribute):
            if (
                not isinstance(node.value, ast.Name)
                or node.value.id != "np"
                or node.attr not in FUNCTIONS
            ):
                raise ValueError(
                    "Only approved numpy functions have attributes"
                )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Attribute):
                raise ValueError("Only approved numpy functions may be called")
            if any(k.arg is None for k in node.keywords):
                raise ValueError("Expanded keyword calls are forbidden")
        if isinstance(node, ast.Subscript) and any(
            isinstance(c, ast.Constant) and c.value is None
            for c in ast.walk(node.slice)
        ):
            raise ValueError("New axes are forbidden")
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in {"features", "history"}
        ):
            if (
                not isinstance(node.slice, ast.Constant)
                or node.slice.value not in feature_names
            ):
                raise ValueError(
                    "Feature access requires a literal registered name"
                )
    if not isinstance(function.body[-1], ast.Return):
        raise ValueError("End the strategy with return score_vector")
    namespace = {}
    exec(
        compile(tree, "<research_strategy>", "exec"),
        {"__builtins__": {}, "np": SAFE_NP},
        namespace,
    )
    return namespace["score"]


def required_features(source):
    return sorted(
        {
            n.slice.value
            for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.Subscript)
            and isinstance(n.value, ast.Name)
            and n.value.id in {"features", "history"}
            and isinstance(n.slice, ast.Constant)
            and isinstance(n.slice.value, str)
        }
    )


def code_signals(strategy, panel, market):
    """Replay code at each review using strictly past data and stable columns."""
    if strategy.entry_rank and strategy.exit_rank >= strategy.entry_rank:
        raise ValueError("Exit must be below entry threshold")
    function = compile_strategy(strategy.code, set(panel.features))
    inputs = required_features(strategy.code)
    scores, schedules = {}, {}
    previous = np.zeros(len(panel.securities))
    cached, last_formation = None, -1
    monthly = {
        int(market.dates.searchsorted(day)): t
        for t, day in enumerate(panel.execution_dates)
    }
    for execution in range(1, len(market.dates)):
        d = execution - 1
        date = market.dates[d]
        t = monthly.get(execution, int(market.formation[d]))
        if t < 0:
            continue
        month_close = execution in monthly
        weekly = (
            date.isocalendar()[:2] != market.dates[execution].isocalendar()[:2]
        )
        due = (
            strategy.review == "daily"
            or (strategy.review == "weekly" and weekly)
            or (strategy.review == "monthly" and month_close)
        )
        if not due:
            continue
        update = (
            cached is None
            or strategy.review != "monthly"
            or (t != last_formation and t % strategy.signal_update_months == 0)
        )
        if update:
            features = {f: panel.features[f][t].copy() for f in inputs}
            history = {
                f: panel.features[f][max(0, t - 11) : t + 1].copy()
                for f in inputs
            }
            trailing = market.returns[max(0, d - 251) : d + 1].view()
            for array in [*features.values(), *history.values(), trailing]:
                array.setflags(write=False)
            with np.errstate(all="ignore"):
                value = np.asarray(
                    function(features, history, trailing), dtype=float
                )
            if value.shape != previous.shape or np.isinf(value).any():
                raise ValueError(
                    "score must return one finite-or-missing value per security"
                )
            cached, last_formation = value.copy(), t
        value = cached.copy()
        value[~panel.mask[t]] = np.nan
        ranks = pd.Series(value).rank(pct=True).to_numpy() * 2 - 1
        if strategy.entry_rank:
            previous = threshold_positions(
                ranks, previous, strategy.entry_rank, strategy.exit_rank
            )
            value = np.where(previous == 0, 0, ranks)
        scores[t] = value
        if update or strategy.review != "monthly" or strategy.entry_rank:
            schedules[execution] = (t, value)
    return scores, schedules


def unit_schedule(strategy, decisions, panel, settings, name_limit):
    designs = construction_designs(panel, strategy.controls)
    schedule = {}
    for day, (formation, score) in decisions.items():
        schedule[day] = make_weights(
            score[None, :],
            [designs[formation]],
            {**settings, "universe_size": name_limit},
        )[0]
    return schedule
