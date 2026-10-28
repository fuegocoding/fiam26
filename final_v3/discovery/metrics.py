"""Fixed prototype admission evidence; recent strength rolls with the year."""

import numpy as np
import pandas as pd

def cohort(frame, proposal):
    declared = proposal['cohort']
    mask = frame.size_grp.isin(declared['size_groups'])
    if declared['gics2']:
        sector = frame.gics.astype('string').str.replace(r'\.0$', '', regex=True).str[:2]
        mask &= sector.isin(declared['gics2'])
    return mask.to_numpy(bool)


def scores(frame, proposal, native_only):
    eligible = frame.prc.ge(1).to_numpy(copy=True)  # Observed positive price, no abs-price substitution.
    if native_only:
        eligible &= cohort(frame, proposal)
    result = np.zeros(len(frame), dtype=np.float32)
    complete = eligible.copy()
    for item in proposal['inputs']:
        values = frame[item['source'] + ':' + item['feature'] + ':' + item.get('transform', 'level')].where(eligible)
        values = values.where(np.isfinite(values))
        ranked = values.groupby(frame.return_month).rank(pct=True)
        complete &= ranked.notna().to_numpy()
        result += item['sign'] * ranked.fillna(0).to_numpy(dtype=np.float32)
    result /= len(proposal['inputs'])
    result[~complete] = np.nan
    return result


def nw_t(values):
    values = np.asarray(values, float)
    if not len(values):
        return np.nan
    centered = values - values.mean()
    variance = (centered @ centered + centered[1:] @ centered[:-1]) / len(values)
    if variance <= 0:
        return float(np.sign(values.mean()) * np.inf) if values.mean() else 0.0
    return float(values.mean() / np.sqrt(variance / len(values)))


def rank_ic(x, y):
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])


def evaluate(frame, proposal, year):
    if not frame.return_month.lt(pd.Period(f'{year}-01')).all():
        raise ValueError('Evidence includes test-year labels')
    score = scores(frame, proposal, True)
    rows = []
    for month, indices in frame.groupby('return_month', sort=True).indices.items():
        x, y = score[indices].astype(float), frame.ret_exc_lead1m.to_numpy()[indices].astype(float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        ic = rank_ic(x, y)
        if not np.isfinite(ic):
            continue
        lower, upper = np.quantile(x, [.05, .95])
        trim = (x >= lower) & (x <= upper)
        rank = pd.Series(x).rank(pct=True).to_numpy()
        top, bottom = y[rank > .8], y[rank <= .2]
        spread = float(top.mean() - bottom.mean()) if len(top) and len(bottom) else np.nan
        rng = np.random.default_rng(np.random.SeedSequence([7, month.ordinal]))
        xr, yr = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
        xr, yr = xr - xr.mean(), yr - yr.mean()
        norm = np.linalg.norm(xr) * np.linalg.norm(yr)
        shuffled = [float(rng.permutation(xr) @ yr / norm) for _ in range(32)]
        rows.append([month, ic, rank_ic(x[trim], y[trim]), spread, *shuffled])
    evidence = pd.DataFrame(rows, columns=['return_month', 'ic', 'trimmed', 'spread', *range(32)])
    metrics = dict(P=np.nan, C=np.nan, R=np.nan, B=np.nan, S=np.nan,
                   trimmed_check=False, halves_check=False, spread_check=False,
                   shuffled_check=False, ic_mean=evidence.ic.mean(), ic_t=nw_t(evidence.ic),
                   spread_mean=evidence.spread.mean(),
                   spread_t=nw_t(evidence.spread.dropna()), months=len(evidence),
                   spread_months=int(evidence.spread.notna().sum()))
    if len(evidence) >= 12:
        ic = evidence.ic.to_numpy()
        recent = evidence.loc[evidence.return_month >= pd.Period(f'{year}-01') - 12, 'ic']
        if not recent.empty:
            half = len(ic) // 2
            checks = [evidence.trimmed.mean() > 0,
                      ic[:half].mean() > 0 and ic[half:].mean() > 0,
                      evidence.spread.mean() > 0,
                      ic.mean() > np.quantile(evidence[list(range(32))].mean(), .95)]
            p, c = float(np.clip(nw_t(ic) / 3, 0, 1)), float((ic > 0).mean())
            r, b = float(np.clip(.5 + recent.mean() / .10, 0, 1)), float(np.mean(checks))
            metrics.update(P=p, C=c, R=r, B=b, S=.40*p + .25*c + .20*r + .15*b)
            metrics.update(zip(['trimmed_check', 'halves_check', 'spread_check', 'shuffled_check'], map(bool, checks)))
    return metrics, evidence.set_index('return_month').spread


