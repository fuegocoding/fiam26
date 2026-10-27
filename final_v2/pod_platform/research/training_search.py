"""Shared risk projections and descriptive return metrics."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .desk_metrics import compute_desk_metrics


@dataclass
class RiskDesign:
    """Formation controls and construction-only liquidity/short restrictions."""

    mask: np.ndarray
    matrix: np.ndarray
    inverse: np.ndarray
    shortable: np.ndarray

    def __iter__(self):
        return iter((self.mask, self.matrix, self.inverse))



def risk_designs(panel, settings=None):
    """Precompute formation-time sector, dollar, beta and size projections."""
    result = []
    settings = settings or {}
    for frame, mask in zip(panel.frames, panel.mask):
        current = frame.loc[mask]
        sectors = current.gics.fillna("unknown").astype(str).str[:2]
        sector = pd.get_dummies(sectors, dtype=float).to_numpy()
        beta = current.beta_60m.fillna(1.0).to_numpy(dtype=float)
        size = np.log(current.me.to_numpy(dtype=float))
        controls = np.column_stack(
            [beta, size] if settings.get("neutralize_size", True) else [beta]
        )
        controls -= controls.mean(axis=0)
        controls /= np.maximum(controls.std(axis=0), 1e-8)
        design = np.column_stack([sector, controls])
        result.append(
            RiskDesign(
                mask,
                design,
                np.linalg.pinv(design),
                current.me.to_numpy(dtype=float) >= 500,
            )
        )
    return result



def summarize(net, market=None, **kwargs):
    """Reuse the platform's PSR, DSR, drawdown, HAC and stability metrics."""
    metrics = compute_desk_metrics(net, periods_per_year=12, **kwargs)
    if market is not None:
        design = np.column_stack([np.ones(len(market)), market])
        intercept, beta = np.linalg.lstsq(design, net, rcond=None)[0]
        metrics["market_alpha_annualized"] = float(intercept * 12)
        metrics["realized_market_beta"] = float(beta)
    return metrics
