"""Project a target onto a dollar turnover budget using formation data only."""

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import lil_matrix, vstack


class AnnualBudget:
    """Calendar-year one-way turnover, charged from settled realized trades.

    Future calendar months keep a paced share of the remaining budget. The
    formation-only fallback may borrow from that share for necessary trades;
    no holding-month outcomes enter the sizing decision.
    """

    def __init__(self, limit, reserve=.05):
        self.limit = float(limit)
        self.reserve = float(reserve)
        if not np.isfinite(self.limit) or self.limit <= 0 or self.reserve < 0:
            raise ValueError('Invalid annual turnover budget')
        self.spent = {}

    def limits(self, month):
        remaining = self.limit-self.spent.get(month.year,0.0)-1e-7
        if remaining <= 0:
            raise ValueError(f'{month}: annual turnover budget exhausted')
        preferred = max(remaining/(13-month.month)-self.reserve,0.0)
        return preferred,remaining

    def settle(self, month, total_one_way, exempt_deployment=0.0):
        charged = float(total_one_way)-float(exempt_deployment)
        if not np.isfinite(charged) or charged < -1e-9:
            raise ValueError('Invalid realized annual turnover charge')
        charged = max(charged,0.0)
        spent = self.spent.get(month.year,0.0)+charged
        if spent >= self.limit-1e-9:
            raise ValueError(f'{month}: realized calendar-year turnover {spent:.6%} reaches annual cap {self.limit:.2%}')
        self.spent[month.year] = spent
        return {'turnover_calendar_year':month.year,
                'turnover_annual_charged_this_month':charged,
                'turnover_annual_year_to_date':spent,
                'turnover_annual_remaining_after':self.limit-spent,
                'turnover_initial_deployment_excluded':float(exempt_deployment)}


def calendar_totals(ledger, exempt_initial=True):
    """Actual annual turnover; initial funding excluded, forced exits included."""
    charged = ledger.traded.astype(float)/2
    charged = charged.copy()
    if exempt_initial and len(charged):
        # The initial book starts from cash: half gross is deployment. Any
        # further within-month liquidation remains chargeable to the year.
        charged.iloc[0] -= float(ledger.gross.iloc[0])/2
        charged.iloc[0] = max(float(charged.iloc[0]),0.0)
    return charged.groupby(charged.index.year).sum()


def constrain(target, reference, frame, drift, caps, limit, names_per_side, *, minimize_trades=False):
    """Keep target sector name counts and select/sizes names within a hard budget.

    reference includes eligible incumbents as well as the desired new names.
    The signed pretrade book is drifted using the preceding settled return.
    Initial funding is handled explicitly by the caller, never hidden here.
    """
    ids = pd.Index(sorted(reference))
    meta = frame.set_index('permno').reindex(ids)
    signs = np.sign(pd.Series(reference).reindex(ids).to_numpy(float))
    desired = pd.Series(target).reindex(ids, fill_value=0).abs().to_numpy(float)
    old = pd.Series(drift, dtype=float).reindex(ids, fill_value=0).to_numpy(float)
    capacity = caps.reindex(ids).to_numpy(float)
    if not np.isfinite(capacity).all() or (capacity <= 0).any():
        raise ValueError('Turnover projection has invalid per-name capacity')
    forced = sum(abs(value) for p,value in drift.items() if p not in ids)
    budget = 2*limit - forced - 1e-8
    if budget < 0:
        raise ValueError('Turnover cap infeasible: mandatory exits exhaust budget')
    n = len(ids)
    # Variables: absolute weight, absolute trade, target deviation, selected.
    rows, lower, upper = [], [], []

    def add(values, lo=-np.inf, hi=np.inf):
        rows.append(values); lower.append(lo); upper.append(hi)

    for i in range(n):
        add({i:signs[i], n+i:-1}, hi=old[i])
        add({i:-signs[i], n+i:-1}, hi=-old[i])
        add({i:1, 2*n+i:-1}, hi=desired[i])
        add({i:-1, 2*n+i:-1}, hi=-desired[i])
        add({i:1, 3*n+i:-capacity[i]}, hi=0)
        add({i:-1, 3*n+i:1e-6}, hi=0)
    add({n+i:1 for i in range(n)}, hi=budget)
    long_target = sum(v for v in target.values() if v>0)
    add({i:1 for i in range(n) if signs[i]>0}, long_target, long_target)
    short_target = -sum(v for v in target.values() if v<0)
    add({i:1 for i in range(n) if signs[i]<0}, short_target, short_target)
    target_series = pd.Series(target, dtype=float)
    sectors = meta.sector.astype(str).to_numpy()
    for sector in sorted(set(sectors)):
        indices = np.flatnonzero(sectors==sector)
        add({int(i):signs[i] for i in indices}, -.20+1e-8, .20-1e-8)
        for sign in (1,-1):
            target_ids = target_series.index[np.sign(target_series)==sign]
            count = int(frame.set_index('permno').sector.reindex(target_ids).eq(sector).sum())
            add({3*n+int(i):1 for i in indices if signs[i]==sign}, count, count)
    matrix = lil_matrix((len(rows),4*n))
    for row,values in enumerate(rows):
        for col,value in values.items(): matrix[row,col] = value
    objective = np.r_[np.zeros(n),np.full(n,1 if minimize_trades else 1e-5),
                      np.full(n,1e-5 if minimize_trades else 1),np.zeros(n)]
    result = milp(objective,
        integrality=np.r_[np.zeros(3*n),np.ones(n)],
        bounds=Bounds(np.zeros(4*n),np.r_[capacity,np.full(2*n,np.inf),np.ones(n)]),
        constraints=LinearConstraint(matrix.tocsr(),lower,upper),
        options={'time_limit':20, 'mip_rel_gap':.01})
    if result.x is None:
        raise ValueError('Turnover cap infeasible or solver found no feasible book: '+result.message)
    chosen = result.x[3*n:] > .5
    # Polish the chosen membership as an LP: MIP feasibility tolerances can
    # otherwise leave tiny selected weights at zero or budgets a little over.
    sparse = matrix.tocsr()
    finite_lo,finite_hi = np.isfinite(lower),np.isfinite(upper)
    polished = linprog(objective,
        A_ub=vstack([sparse[finite_hi],-sparse[finite_lo]]),
        b_ub=np.r_[np.asarray(upper)[finite_hi],-np.asarray(lower)[finite_lo]],
        bounds=[(0,cap) for cap in capacity]+[(0,None)]*(2*n)
            +[(int(value),int(value)) for value in chosen], method='highs',
        options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not polished.success:
        raise ValueError('Selected turnover membership cannot be polished: '+polished.message)
    weights = dict(zip(ids[chosen],(polished.x[:n]*signs)[chosen],strict=True))
    traded = sum(abs(weights.get(p,0)-drift.get(p,0)) for p in set(weights)|set(drift))
    if traded/2 > limit+1e-9:
        raise ValueError('Turnover solver exceeded hard one-way budget')
    w = pd.Series(weights)
    if (sum(w>0)!=names_per_side or sum(w<0)!=names_per_side
        or w.abs().sum()>2+1e-9 or abs(w.sum())>.5+1e-9
        or abs(w.clip(lower=0).sum()-long_target)>1e-9
        or abs(-w.clip(upper=0).sum()-short_target)>1e-9
        or (w.abs()>caps.reindex(w.index)+1e-9).any()):
        raise ValueError('Turnover solver violated book or individual capacity bounds')
    return weights, {'rebalance_one_way_turnover':traded/2,
        'turnover_solver_status':int(result.status),
        'turnover_solver_gap':float(result.mip_gap)}
