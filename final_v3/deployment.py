"""Deploy own capital while respecting per-name capacity and book limits."""
import numpy as np
import pandas as pd
from scipy.optimize import linprog


def resize(weights, frame, nav, long_target, short_target, *,
           long_participation, short_participation, execution_days, short_caps=None):
    """Resize selected names; never relax liquidity to meet a cash target."""
    w = pd.Series(weights, dtype=float)
    meta = frame.set_index('permno').reindex(w.index)
    signs = np.sign(w.to_numpy())
    basis = w.abs().to_numpy()
    caps = np.minimum(.05 - 1e-10, meta.dolvol_126d.to_numpy(float) * execution_days
                      * np.where(signs > 0, long_participation, short_participation) / nav)
    if short_caps is not None:
        caps = np.minimum(caps, short_caps.reindex(w.index).fillna(np.inf).to_numpy(float))
    if not np.isfinite(caps).all() or (caps <= 0).any():
        raise ValueError('Invalid stock capacity for Treasury-cap sizing')
    target = np.zeros(len(w))
    for side, total in ((1, long_target), (-1, short_target)):
        mask = signs == side
        if caps[mask].sum() < total - 1e-10:
            raise ValueError(f'Treasury cap infeasible: selected {side:+} leg capacity below {total:.6f} NAV')
        free = mask.copy()
        remaining = total
        while free.any():
            proposed = remaining * basis[free] / basis[free].sum()
            hit = proposed > caps[free]
            ids = np.flatnonzero(free)
            if not hit.any():
                target[ids] = proposed
                break
            fixed = ids[hit]
            target[fixed] = caps[fixed]
            remaining -= caps[fixed].sum()
            free[fixed] = False
    sectors = meta.sector.astype(str).to_numpy()
    sector_matrix = np.array([(sectors == sector) * signs for sector in sorted(set(sectors))])
    # The usual proportional sizing needs no solver. Use a bounded L1
    # projection only if sector limits require redistribution within the names.
    if (np.abs(sector_matrix @ target) > .20 - 1e-9).any():
        n = len(w)
        zeros = np.zeros_like(sector_matrix)
        inequalities = np.vstack([np.c_[sector_matrix, zeros], np.c_[-sector_matrix, zeros],
                                   np.c_[np.eye(n), -np.eye(n)], np.c_[-np.eye(n), -np.eye(n)]])
        upper = np.r_[np.full(2*len(sector_matrix), .20 - 1e-8), target, -target]
        equality = np.c_[np.array([signs > 0, signs < 0], dtype=float), np.zeros((2,n))]
        solution = linprog(np.r_[np.zeros(n), np.ones(n)], A_ub=inequalities, b_ub=upper,
            A_eq=equality, b_eq=[long_target, short_target],
            bounds=[(min(1e-6, cap/2), cap) for cap in caps] + [(0,None)]*n, method='highs')
        if not solution.success:
            raise ValueError('Treasury cap infeasible under sector/capacity bounds: '+solution.message)
        target = solution.x[:n]
    sized = target * signs
    if (target > caps + 1e-9).any() or (target <= 1e-10).any():
        raise ValueError('Treasury sizing violates capacity or removes selected names')
    if target.sum() > 2.0 + 1e-10 or abs(sized.sum()) > .5 + 1e-10:
        raise ValueError('Treasury sizing violates gross/net exposure limits')
    if abs(target[signs > 0].sum() - long_target) > 1e-9:
        raise ValueError('Treasury sizing failed to deploy the required own capital')
    return dict(zip(w.index, sized, strict=True))
