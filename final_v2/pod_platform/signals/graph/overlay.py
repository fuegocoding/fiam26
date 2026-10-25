"""Blends the sleeve score with the graph EPS forecast through monthly ranks."""

import pandas as pd

GRAPH_WEIGHT = 0.05


def company_forecasts(
    forecasts: pd.DataFrame, formation: pd.DataFrame
) -> pd.DataFrame:
    """Next-month graph forecasts keyed by return month and company.

    Args:
        forecasts: permno, eom, target_month and graph_attention_eps.
        formation: Formation panel with eom (monthly period), permno, permco.

    Returns:
        Frame with return_month, permco and graph_attention_eps.
    """
    graph = forecasts[
        ["permno", "eom", "target_month", "graph_attention_eps"]
    ].copy()
    graph["eom"] = pd.to_datetime(graph.eom).dt.to_period("M")
    graph["return_month"] = pd.to_datetime(graph.target_month).dt.to_period("M")
    graph = graph.loc[graph.return_month.eq(graph.eom + 1)].copy()
    graph = graph.merge(
        formation[["eom", "permno", "permco"]],
        on=["eom", "permno"],
        how="left",
    )
    return graph[["return_month", "permco", "graph_attention_eps"]]


def monthly_rank(values: pd.Series, months: pd.Series) -> pd.Series:
    """Percentile rank within each month, averaging ties."""
    return values.groupby(months, sort=False).rank(method="average", pct=True)


def apply(
    scores: pd.DataFrame, graph: pd.DataFrame, weight: float = GRAPH_WEIGHT
) -> pd.DataFrame:
    """Adds graph and final scores to company-mapped sleeve scores.

    The graph never routes among sleeves; it only tilts the blended rank.
    Stocks without a graph forecast keep their sleeve rank.
    """
    result = scores.merge(graph, on=["return_month", "permco"], how="left")
    months = result.return_month
    result["sleeve_rank"] = monthly_rank(result.sleeve_score, months)
    result["graph_rank"] = monthly_rank(result.graph_attention_eps, months)
    available = result.graph_attention_eps.notna()
    result["final_score"] = (
        1 - weight
    ) * result.sleeve_rank + weight * result.graph_rank
    result.loc[~available, "final_score"] = result.sleeve_rank[~available]
    result["graph_available"] = available
    return result
