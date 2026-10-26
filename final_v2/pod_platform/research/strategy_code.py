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
from pydantic import Field, model_validator

from ..contracts.base import StrictModel
from .filing_text import FilingMeasurement
from .forecast_construction import construct_target
from .pod_plans import construction_designs
from .score_worker import ScoreWorker
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
        "isnan",
        "nan",
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
        "copy",
        "argsort",
        "nanmin",
        "nanmax",
        "prod",
        "nanprod",
        "nanmedian",
        "nanpercentile",
        "nanquantile",
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
    weighting: Literal["score", "equal_tail"] = "score"
    hedge_policy: Literal["paired_signal", "liquid_sector"] = "paired_signal"
    construction: Literal[
        "unscaled", "liquidity_tilt", "liquidity_borrow_tilt"
    ] = "unscaled"

    forecast_model: Literal["code", "ridge"] = "code"
    model_features: list[str] = Field(default_factory=list, max_length=147)
    model_penalty: float = Field(default=1.0, gt=0, le=100.0)
    model_lookback_months: int = Field(default=36, ge=12, le=120)

    @model_validator(mode="after")
    def explicit_hedge_controls(self):
        if self.hedge_policy == "liquid_sector" and self.controls != "sector":
            raise ValueError(
                "liquid_sector hedge implements sector/dollar matching; choose controls=sector and report residual beta"
            )
        return self


class CodeBatch(StrictModel):
    diagnosis: str = Field(min_length=20, max_length=2000)
    strategies: list[CodeStrategy] = Field(min_length=1, max_length=3)
    filing_measurements: list[FilingMeasurement] = Field(
        default_factory=list, max_length=12
    )


class NumericalMethods(ast.NodeTransformer):
    """Translate finite ndarray convenience methods to the same numpy facade."""

    methods = {"copy", "mean", "std", "sum", "argsort"}

    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body.pop(0)
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in self.methods
            and not (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "np"
            )
        ):
            return ast.copy_location(
                ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="np", ctx=ast.Load()),
                        attr=node.func.attr,
                        ctx=ast.Load(),
                    ),
                    args=[node.func.value, *node.args],
                    keywords=node.keywords,
                ),
                node,
            )
        return node


def compile_strategy(source, feature_names):
    """Validate syntax before compilation; no imports, loops, attributes or IO.

    Numerical assignments, local array masks, finite conditionals and acyclic
    pure helpers are actual agent-written Python. Attributes are permitted
    exclusively on the finite numpy facade (array convenience methods lower to
    the same facade).
    This is a language restriction, not a claim of an OS process sandbox.
    """
    tree = ast.fix_missing_locations(
        NumericalMethods().visit(ast.parse(source))
    )
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
        ast.AugAssign,
        ast.If,
        ast.In,
        ast.NotIn,
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
    reserved = {"features", "history", "daily", "np"}
    functions = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ]
    helpers = {node.name for node in functions[1:]}
    if (
        len(functions) > 9
        or len(helpers) != len(functions) - 1
        or any(
            name in reserved | {"score"} or name.startswith("_")
            for name in helpers
        )
    ):
        raise ValueError("Use at most eight distinct pure numerical helpers")
    assigned = reserved | helpers
    for helper in functions[1:]:
        arguments = helper.args
        if (
            not 1 <= len(arguments.args) <= 4
            or arguments.posonlyargs
            or arguments.kwonlyargs
            or arguments.vararg
            or arguments.kwarg
            or arguments.defaults
            or helper.decorator_list
            or helper.returns is not None
            or any(
                arg.annotation is not None
                or arg.arg in reserved | helpers
                or arg.arg.startswith("_")
                for arg in arguments.args
            )
        ):
            raise ValueError(
                "Helpers use 1-4 plain local arguments with no defaults or decorators"
            )
        assigned.update(arg.arg for arg in arguments.args)
    if len(list(ast.walk(tree))) > 1800:
        raise ValueError("Strategy exceeds syntax resource bound")
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AugAssign))
    ]
    for statement in assignments:
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        if len(targets) != 1:
            raise ValueError("Assignments must use one explicit target")
        target = targets[0]
        if isinstance(target, ast.Name):
            if target.id in reserved | helpers or target.id.startswith("_"):
                raise ValueError(
                    "Do not overwrite inputs, helpers or numpy namespace"
                )
            assigned.add(target.id)
        elif not isinstance(target, ast.Subscript):
            raise ValueError(
                "Assignments create local names or update local arrays"
            )
    for statement in assignments:
        target = (
            statement.targets[0]
            if isinstance(statement, ast.Assign)
            else statement.target
        )
        if isinstance(target, ast.Subscript) and (
            not isinstance(target.value, ast.Name)
            or target.value.id not in assigned
            or target.value.id in reserved | helpers
        ):
            raise ValueError(
                "Array assignment is permitted only on local arrays; inputs stay read-only"
            )

    def own_nodes(node):
        yield node
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef):
                continue
            yield from own_nodes(child)

    graph = {
        node.name: {
            child.func.id
            for statement in node.body
            for child in own_nodes(statement)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        }
        for node in functions
    }

    def check_calls(name, stack):
        if name in stack:
            raise ValueError("Recursive numerical helpers are forbidden")
        for called in graph.get(name, ()):
            if called not in helpers:
                raise ValueError(
                    "Only approved numpy functions or pure declared helpers may be called"
                )
            check_calls(called, stack | {name})

    for name in graph:
        check_calls(name, set())
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(
                f"Unsupported strategy syntax: {type(node).__name__}"
            )
        if isinstance(node, ast.Name) and node.id not in assigned:
            raise ValueError(f"Unknown strategy name: {node.id}")
        if isinstance(node, ast.Tuple):
            parent = parents.get(node)
            in_slice = (
                isinstance(parent, ast.Subscript) and parent.slice is node
            )
            in_axis = (
                isinstance(parent, ast.keyword)
                and parent.arg == "axis"
                and all(
                    isinstance(x, ast.Constant) and isinstance(x.value, int)
                    for x in node.elts
                )
            )
            if not in_slice and not in_axis:
                raise ValueError(
                    "Tuples are permitted only for slices or reduction axes"
                )
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                parent = parents.get(node)
                key = (
                    isinstance(parent, ast.Subscript)
                    and parent.slice is node
                    and isinstance(parent.value, ast.Name)
                    and parent.value.id in {"features", "history"}
                )
                membership = (
                    isinstance(parent, ast.Compare)
                    and parent.left is node
                    and len(parent.ops) == 1
                    and isinstance(parent.ops[0], (ast.In, ast.NotIn))
                    and len(parent.comparators) == 1
                    and isinstance(parent.comparators[0], ast.Name)
                    and parent.comparators[0].id in {"features", "history"}
                )
                if node.value not in feature_names or not (key or membership):
                    raise ValueError(
                        "Strings are permitted only as registered feature keys"
                    )
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
            if not isinstance(node.func, ast.Attribute) and not (
                isinstance(node.func, ast.Name) and node.func.id in helpers
            ):
                raise ValueError(
                    "Only approved numpy functions or pure helpers may be called"
                )
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
    namespace["score"]._strategy_source = source
    namespace["score"]._strategy_feature_names = tuple(feature_names)
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


def native_review_calendar(strategy, panel, market):
    """Yield decision timing without computing or inspecting any forecast."""
    monthly = {
        int(market.dates.searchsorted(day)): t
        for t, day in enumerate(panel.execution_dates)
    }
    last_update = None
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
            last_update is None
            or strategy.review != "monthly"
            or (t != last_update and t % strategy.signal_update_months == 0)
        )
        if update:
            last_update = t
        yield execution, t, update


def code_signals(strategy, panel, market):
    """Replay code at each review using strictly past data and stable columns."""
    if strategy.entry_rank and strategy.exit_rank >= strategy.entry_rank:
        raise ValueError("Exit must be below entry threshold")
    function = compile_strategy(strategy.code, set(panel.features))
    inputs = required_features(strategy.code)
    scores, schedules = {}, {}
    previous = np.zeros(len(panel.securities))
    cached = None
    with ScoreWorker(function) as worker:
        for execution, t, update in native_review_calendar(
            strategy, panel, market
        ):
            d = execution - 1
            if update:
                features = {f: panel.features[f][t].copy() for f in inputs}
                history = {
                    f: panel.features[f][max(0, t - 11) : t + 1].copy()
                    for f in inputs
                }
                trailing = market.returns[max(0, d - 251) : d + 1].view()
                for array in [*features.values(), *history.values(), trailing]:
                    array.setflags(write=False)
                value = worker.score(features, history, trailing)
                if value.shape != previous.shape or np.isinf(value).any():
                    raise ValueError(
                        "score must return one finite-or-missing value per security"
                    )
                cached = value.copy()
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
    """Use the identical supported constructor for every native review."""
    designs = construction_designs(panel, strategy.controls)
    return {
        day: construct_target(
            strategy,
            score,
            designs[formation],
            panel.frames[formation],
            settings,
            name_limit,
        )
        for day, (formation, score) in decisions.items()
    }
