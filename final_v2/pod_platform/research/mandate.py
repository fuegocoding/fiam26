"""Shared Part 2 mandate definitions for construction and observed holdings."""

from __future__ import annotations

import numpy as np
import pandas as pd

GICS = ('10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60')
POSITION_EPS = 1e-10
LIMIT_EPS = 1e-7
DEFAULTS = {'min_sectors': 5, 'sector_net_limit': .20, 'volatility_limit': .12}
NEUTRALITY = 'Hard GICS sector net bounds of +/-20% of NAV; exact dollar neutrality is not guaranteed'


def validate_part2_settings(settings):
    """Allow stricter policies, never a silent relaxation of the chosen mandate."""
    breadth = settings.get('min_sectors', 5)
    sector = settings.get('sector_net_limit', .2)
    volatility = settings.get('volatility_limit', .12)
    if (not isinstance(breadth, int) or isinstance(breadth, bool) or not 5 <= breadth <= 11
        or not np.isfinite(sector) or not 0 < sector <= .2
        or not np.isfinite(volatility) or not 0 < volatility <= .12):
        raise ValueError('Part 2 requires 5-11 sectors, sector net limit <=20%, and volatility limit <=12%')


def sector_codes(values):
    """Recognize current GICS sectors from sector or full classification codes."""
    codes = pd.Series(values, dtype='string').str.replace(r'\.0$', '', regex=True).str[:2]
    return codes.where(codes.isin(GICS), 'unknown').fillna('unknown').to_numpy(str)


def frame_sectors(frame, names):
    return sector_codes(frame.gics if frame is not None and 'gics' in frame else ['unknown'] * names)


def sector_rows(weights, sectors):
    weights = np.asarray(weights, float)
    sectors = sector_codes(sectors)
    rows = []
    for code in (*GICS, 'unknown'):
        w = weights[sectors == code]
        rows.append({'sector': code, 'long_weight': float(np.maximum(w, 0).sum()),
                     'short_weight': float(-np.minimum(w, 0).sum()),
                     'net_weight': float(w.sum()), 'gross_weight': float(np.abs(w).sum()),
                     'position_count': int((np.abs(w) > POSITION_EPS).sum())})
    return rows


def check_book(weights, sectors, settings):
    """No empty-book exemption: cash is research evidence, not mandate compliance."""
    w = np.asarray(weights, float)
    if w.ndim != 1 or not np.isfinite(w).all():
        return {'pass': False, 'failures': ['nonfinite_weights']}
    rows = sector_rows(w, sectors)
    count = int((np.abs(w) > POSITION_EPS).sum())
    breadth = sum(r['position_count'] > 0 for r in rows if r['sector'] in GICS)
    maximum = max(abs(r['net_weight']) for r in rows)
    checks = {
        'minimum_positions': count >= settings.get('min_positions', 200),
        'maximum_positions': count <= settings.get('universe_size', 500),
        'gross': np.abs(w).sum() <= settings.get('gross_limit', 2) + LIMIT_EPS,
        'net': abs(w.sum()) <= .5 + LIMIT_EPS,
        'single_name': np.abs(w).max(initial=0) <= settings.get('single_name_limit', .05) + LIMIT_EPS,
        'sector_breadth': breadth >= settings.get('min_sectors', 5),
        'sector_net': maximum <= settings.get('sector_net_limit', .2) + LIMIT_EPS,
        'sector_classification': rows[-1]['position_count'] == 0,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {'pass': not failures, 'failures': failures, 'positions': count,
            'sectors': breadth, 'maximum_absolute_sector_net': maximum}


def volatility_check(returns, ceiling=.12):
    values = np.asarray(returns, float)
    valid = len(values) >= 2 and np.isfinite(values).all()
    volatility = float(np.std(values, ddof=1) * np.sqrt(252)) if valid else None
    return {'pass': bool(valid and volatility <= ceiling + 1e-12),
            'annualized_realized_volatility': volatility, 'ceiling': ceiling,
            'frequency': 'daily', 'periods_per_year': 252, 'ddof': 1,
            'observations': len(values)}


def audit_holdings(dates, holdings, market, active_indices, settings, rebalance_days):
    """Audit actual daily holdings using each session's formation snapshot."""
    rows, failures = [], []
    breadth, maxima = [], []
    for date, weights, day in zip(dates, holdings, active_indices):
        formation = int(market.formation[day])
        frame = market.frames[formation] if market.frames is not None and formation >= 0 else None
        sectors = frame_sectors(frame, len(weights))
        result = check_book(weights, sectors, settings)
        if frame is None or 'me' not in frame:
            result['failures'].append('short_eligibility_unknown')
        else:
            cap = frame.me.to_numpy(float)
            if np.any((weights < -POSITION_EPS) & (~np.isfinite(cap) | (cap < 500))):
                result['failures'].append('short_eligibility')
        breadth.append(result['sectors'])
        maxima.append(result['maximum_absolute_sector_net'])
        if result['failures']:
            failures.append({'date': str(date.date()), 'rebalance': int(day) in rebalance_days,
                             'failures': result['failures']})
        rows.extend({'date': str(date.date()), 'is_rebalance': int(day) in rebalance_days, **r}
                    for r in sector_rows(weights, sectors))
    return pd.DataFrame(rows), {'pass': bool(len(dates) and not failures),
        'breaches': failures, 'minimum_sector_breadth': min(breadth, default=0),
        'maximum_absolute_sector_net': max(maxima, default=0),
        'neutrality_definition': NEUTRALITY}
