"""Formation-time controls for agent-proposed portfolios."""

import numpy as np
import pandas as pd

from .training_search import RiskDesign


def construction_designs(panel, controls):
    """Apply the plan's declared controls without modifying forecast ranks."""
    designs = []
    for frame, mask in zip(panel.frames, panel.mask):
        local = frame.loc[mask]
        if controls == "none":
            matrix = np.empty((len(local), 0))
        elif controls == "dollar":
            matrix = np.ones((len(local), 1))
        else:
            sectors = local.gics.fillna("unknown").astype(str).str[:2]
            matrix = pd.get_dummies(sectors, dtype=float).to_numpy()
            if controls == "sector_beta":
                beta = local.beta_60m.fillna(1).to_numpy(dtype=float)
                beta = (beta - beta.mean()) / max(beta.std(), 1e-8)
                matrix = np.column_stack([matrix, beta])
        design = RiskDesign(
                mask,
                matrix,
                np.linalg.pinv(matrix),
                local.me.to_numpy() >= 500,
            )
        design.declared_universe_digest = panel.metadata.get("strategy_universe_digest")
        designs.append(design)
    return designs
