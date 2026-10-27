"""Inspectable research code with a deliberately small, IO-free Python surface.

The evaluator supplies only observations known at the decision close. Outcomes,
capital, credentials, filesystem and provider clients never enter this runtime.
"""

from __future__ import annotations

import ast
import json
import datetime as dt
import re
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field, ValidationError, model_validator

from ..contracts.base import StrictModel
from .filing_text import FilingMeasurement
from .forecast_construction import construct_target
from .pod_plans import construction_designs
from .score_worker import ScoreWorker
from .training_tactical import threshold_positions
from .strategy_scope import StrategyUniverse, scoped_panel

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
        "concatenate",
        "full",
        "full_like",
        "zeros_like",
        "roll",
        "unique",
        "size",
    )
}
SAFE_NP = SimpleNamespace(**FUNCTIONS)


class TreeModelParameters(StrictModel):
    rounds: int = Field(ge=1, le=2000)
    max_depth: int = Field(ge=1, le=8)
    learning_rate: float = Field(gt=0, le=1)
    min_child_weight: float = Field(gt=0)
    subsample: float = Field(gt=0, le=1)
    column_subsample: float = Field(gt=0, le=1)
    label: Literal["rank", "return"]


class CodeStrategy(StrictModel):
    pod: Literal["fundamentals", "events", "valuation", "dynamics"]
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    mechanism: str = Field(min_length=30, max_length=1800)
    falsification: str = Field(min_length=20, max_length=1000)
    evidence_refs: list[str] = Field(default_factory=list, max_length=16)
    code: str = Field(min_length=20, max_length=12000)
    controls: Literal["none", "dollar", "sector", "sector_beta"]
    review: Literal["daily", "weekly", "monthly"]
    signal_update_months: int = Field(default=1, ge=1, le=12)
    entry_rank: float = Field(default=0, ge=0, le=0.95)
    exit_rank: float = Field(default=0, ge=0, le=0.9)
    expected_holding_days: int = Field(ge=1, le=504)
    weighting: Literal["score", "equal_tail"] = "score"
    hedge_policy: Literal["paired_signal", "liquid_sector", "long_only"] = "paired_signal"
    construction: Literal["unscaled"] = "unscaled"
    universe: StrategyUniverse
    evidence_lookback_months: int = Field(ge=12, le=120)
    validation_window_months: int = Field(ge=3, le=24)
    research_topic: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    parent_strategy_id: str | None = None

    forecast_model: Literal["code", "ridge", "xgboost"] = "code"
    model_features: list[str] = Field(default_factory=list, max_length=147)
    model_penalty: float = Field(default=1.0, gt=0, le=100.0)
    model_lookback_months: int = Field(default=36, ge=12, le=120)
    model_parameters: TreeModelParameters | None = Field(default=None, exclude_if=lambda v: v is None)

    @model_validator(mode="after")
    def explicit_hedge_controls(self):
        if self.validation_window_months * 2 > self.evidence_lookback_months:
            raise ValueError("Choose an evidence lookback containing at least two chronological validation windows")
        if self.hedge_policy == "long_only" and self.controls != "none":
            raise ValueError("A long-only portfolio cannot be dollar/sector neutral; choose controls=none")
        if (self.entry_rank == 0 and self.exit_rank != 0) or (
            self.entry_rank > 0 and self.exit_rank >= self.entry_rank
        ):
            raise ValueError("Use entry_rank=exit_rank=0 for continuous sizing, or 0<=exit_rank<entry_rank for hysteresis; the exit threshold must be lower than entry")
        if self.forecast_model != "code" and not self.model_features:
            raise ValueError("Fitted proposals must explicitly choose model_features; no default feature model is supplied")
        if self.forecast_model == "xgboost" and self.model_parameters is None:
            raise ValueError("XGBoost proposals must explicitly choose model_parameters")
        if self.forecast_model != "xgboost" and self.model_parameters is not None:
            raise ValueError("model_parameters applies only to the XGBoost forecaster")
        if self.forecast_model != "code":
            try:
                inputs = required_features(self.code)
            except (SyntaxError, ValueError):
                inputs = None  # The numerical evaluator records the exact syntax failure.
            if inputs is not None and "model_forecast" not in inputs:
                raise ValueError("Fitted code must return features['model_forecast']; handwritten feature weights are not a fitted model")
        if self.hedge_policy == "liquid_sector" and self.controls != "sector":
            raise ValueError(
                "liquid_sector hedge implements sector/dollar matching; choose controls=sector and report residual beta"
            )
        if self.hedge_policy == "paired_signal" and re.search(
            r"long[- ]only|avoid(?:ing)? short|no short|without (?:a |the )?short", self.mechanism, re.I
        ):
            raise ValueError("A long-only hypothesis must declare hedge_policy=long_only and controls=none; paired_signal creates short positions")
        return self


class ResearchTopic(StrictModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    mechanism: str = Field(min_length=30, max_length=800)


class TopicPlan(StrictModel):
    diagnosis: str = Field(min_length=20, max_length=2000)
    research_action: Literal["plan"]
    research_topics: list[ResearchTopic] = Field(min_length=2, max_length=8)


class AlternativeResearchSearch(StrictModel):
    entity_query: str = Field(min_length=2, max_length=100)
    metric_query: str = Field(min_length=2, max_length=100)


class AlternativeResearchRequest(StrictModel):
    discovery_id: str = Field(pattern=r"^discovery_[0-9a-f]{24}$")
    entity_id: int = Field(gt=0)
    metric_id: int = Field(gt=0)
    dataset_id: str = Field(min_length=1, max_length=100)
    start: dt.date
    end: dt.date


class CodeBatch(StrictModel):
    diagnosis: str = Field(min_length=20, max_length=8000)
    research_action: Literal["plan", "propose", "search", "retain"] = "propose"
    research_topics: list[ResearchTopic] = Field(default_factory=list, max_length=8)
    research_queries: list[str] = Field(default_factory=list, max_length=3)
    alternative_searches: list[AlternativeResearchSearch] = Field(default_factory=list, max_length=3)
    alternative_requests: list[AlternativeResearchRequest] = Field(default_factory=list, max_length=3)
    strategies: list[CodeStrategy] = Field(default_factory=list, max_length=3)
    filing_measurements: list[FilingMeasurement] = Field(
        default_factory=list, max_length=12
    )

    @model_validator(mode="after")
    def explicit_research_action(self):
        if self.research_action == "plan" and (len(self.research_topics) < 2 or self.strategies):
            raise ValueError("Plan at least two distinct economic topics before writing strategies")
        if self.research_action == "propose" and not self.strategies:
            raise ValueError("Propose executable strategies, request a search, or explicitly retain the supported book")
        if self.research_action == "search" and not (
            self.research_queries or self.alternative_searches or self.alternative_requests
        ):
            raise ValueError("A search action requires paper queries, alternative searches or alternative-data requests")
        if any(not 5 <= len(query.strip()) <= 240 for query in self.research_queries):
            raise ValueError("Research queries must contain 5-240 characters")
        return self

    @classmethod
    def validate_partial(cls, text):
        """Preserve valid siblings when an individual strategy fails validation."""
        text = text.strip()
        if text.startswith(("```json\n", "```\n")) and text.endswith("\n```"):
            text = text.split("\n", 1)[1].rsplit("\n", 1)[0]
        raw = json.loads(text)
        # Some JSON-mode providers repeat empty batch fields inside a strategy.
        # Empty containers carry no instruction or economic content. Nonempty
        # misplaced fields remain errors rather than silently losing research.
        if isinstance(raw, dict) and isinstance(raw.get("strategies"), list):
            for strategy in raw["strategies"]:
                if isinstance(strategy, dict):
                    for key in ("filing_measurements", "research_queries", "alternative_searches", "alternative_requests"):
                        if strategy.get(key) == []:
                            strategy.pop(key)
        try:
            return cls.model_validate(raw), []
        except ValidationError as exc:
            errors = exc.errors(include_input=False, include_url=False, include_context=False)
            if not all(len(e["loc"]) >= 2 and e["loc"][0] == "strategies"
                       and isinstance(e["loc"][1], int) for e in errors):
                raise
            bad = {e["loc"][1] for e in errors}
            rejected = [{"strategy": value if isinstance(value, dict) else None,
                         "errors": [e for e in errors if e["loc"][1] == i]}
                        for i, value in enumerate(raw["strategies"]) if i in bad]
            kept = [value for i, value in enumerate(raw["strategies"]) if i not in bad]
            if not kept:
                return None, rejected
            return cls.model_validate({**raw, "strategies": kept}), rejected


class ProposalBatch(CodeBatch):
    research_action: Literal["propose", "search", "retain"]


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
        if isinstance(node.func, ast.Name) and node.func.id == "len":
            if len(node.args) != 1 or node.keywords:
                raise ValueError("len() requires one numerical array")
            return ast.copy_location(ast.Call(
                func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="size", ctx=ast.Load()),
                args=node.args, keywords=[ast.keyword(arg="axis", value=ast.Constant(value=0))],
            ), node)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in {"features", "history"}
            and node.func.attr == "get"
        ):
            if (
                len(node.args) not in {1, 2}
                or node.keywords
                or not isinstance(node.args[0], ast.Constant)
                or not isinstance(node.args[0].value, str)
                or (len(node.args) == 2 and not (
                    isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, (int, float, type(None)))
                    or isinstance(node.args[1], ast.Attribute)
                    and isinstance(node.args[1].value, ast.Name)
                    and node.args[1].value.id == "np"
                    and node.args[1].attr == "nan"
                ))
            ):
                raise ValueError("Feature get() requires a literal registered key and an optional scalar default")
            # Registered keys are always supplied. Lower this familiar spelling
            # to the same validated access; arbitrary dictionary methods stay forbidden.
            return ast.copy_location(ast.Subscript(
                value=node.func.value, slice=node.args[0], ctx=ast.Load()
            ), node)
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
            in_concatenation = (
                isinstance(parent, ast.Call)
                and isinstance(parent.func, ast.Attribute)
                and isinstance(parent.func.value, ast.Name)
                and parent.func.value.id == "np"
                and parent.func.attr == "concatenate"
                and parent.args and parent.args[0] is node
                and 1 <= len(node.elts) <= 12
            )
            if not in_slice and not in_axis and not in_concatenation:
                raise ValueError(
                    "Tuples are permitted only for slices, reduction axes or bounded concatenation"
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
            for n in ast.walk(NumericalMethods().visit(ast.parse(source)))
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
    for execution in range(len(market.dates)):
        d = execution
        date = market.dates[d]
        t = monthly.get(execution, int(market.formation[d]))
        if t < 0:
            continue
        month_close = execution in monthly
        weekly = (
            execution + 1 < len(market.dates)
            and date.isocalendar()[:2] != market.dates[execution + 1].isocalendar()[:2]
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


def code_signals(strategy, panel, market, *, readiness=None, decision_feature_provider=None):
    """Replay code at each review using strictly past data and stable columns."""
    panel = scoped_panel(panel, strategy.universe)
    if strategy.entry_rank and strategy.exit_rank >= strategy.entry_rank:
        raise ValueError("Exit must be below entry threshold")
    function = compile_strategy(strategy.code, set(panel.features))
    inputs = required_features(strategy.code)
    scores, schedules = {}, {}
    previous = np.zeros(len(panel.securities))
    cached = None
    previous_scope = None
    with ScoreWorker(function) as worker:
        for execution, t, update in native_review_calendar(
            strategy, panel, market
        ):
            d = execution
            if update:
                columns = np.flatnonzero(panel.mask[t])
                features = {f: panel.features[f][t, columns].copy() for f in inputs}
                if decision_feature_provider is not None:
                    # Date-only filing availability is end-of-day. Use the
                    # known start-of-session snapshot for a same-session fill.
                    as_of = market.dates[d].tz_localize("America/New_York")
                    features.update({f: values[columns].copy() for f, values in
                        decision_feature_provider(as_of, inputs, panel.mask[t]).items()})
                history = {
                    f: panel.features[f][max(0, t - 11) : t + 1, columns].copy()
                    for f in inputs
                }
                # Give code only current members, including their own past
                # observations before cohort entry. Outside stocks cannot alter
                # cross-sectional means, array lengths or numerical defaults.
                trailing = market.returns[max(0, d - 252) : d, columns].copy()
                for array in [*features.values(), *history.values(), trailing]:
                    array.setflags(write=False)
                value = (worker.score(features, history, trailing)
                    if len(columns) else np.zeros(0))
                if value.shape != (len(columns),) or np.isinf(value).any():
                    raise ValueError(
                        "score must return one finite-or-missing value per security"
                    )
                cached = np.full(len(panel.securities), np.nan)
                cached[columns] = value
            value = cached.copy()
            value[~panel.mask[t]] = np.nan
            if readiness is not None:
                readiness[execution] = bool(np.isfinite(value).any())
            ranks = pd.Series(value).rank(pct=True).to_numpy() * 2 - 1
            if strategy.entry_rank:
                previous = threshold_positions(
                    ranks, previous, strategy.entry_rank, strategy.exit_rank
                )
                value = np.where(previous == 0, 0, ranks)
            scores[t] = value
            scope_changed = previous_scope is None or not np.array_equal(previous_scope, panel.mask[t])
            if update or strategy.review != "monthly" or strategy.entry_rank or scope_changed:
                schedules[execution] = (t, value)
                previous_scope = panel.mask[t].copy()
    return scores, schedules


def unit_schedule(strategy, decisions, panel, settings, name_limit):
    """Use the identical supported constructor for every native review."""
    panel = scoped_panel(panel, strategy.universe)
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
