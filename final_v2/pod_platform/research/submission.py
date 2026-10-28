"""Reconciled Part 2 reporting from the executed campaign, never a legacy book."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .campaign_metrics import frequency_metrics
from .mandate import POSITION_EPS, audit_holdings, volatility_check

ARCANA_EXHIBITS = [
    'Summary table', 'Idiosyncratic vs Factor', 'Decomposition',
    'Long vs Short spread', 'Concentration', 'Earnings P&L',
    'Exposures (STYLE EXPOSURE)', 'Drawdown (S&P 500 benchmark)',
    'Sharpe ratio', 'Hit rate', 'Turnover',
]


def coverage_check(dates, evidence_scope, expected_sessions=None):
    dates = pd.DatetimeIndex(dates)
    expected = pd.period_range('2021-01', '2026-08', freq='M')
    observed = dates.to_period('M').unique()
    missing = expected.difference(observed).astype(str).tolist()
    bounds = bool(len(dates) and dates[0] <= pd.Timestamp('2021-01-04')
                  and dates[-1] >= pd.Timestamp('2026-08-31'))
    expected_dates = pd.DatetimeIndex(expected_sessions) if expected_sessions is not None else pd.DatetimeIndex([])
    expected_dates = expected_dates[(expected_dates >= '2021-01-01') & (expected_dates <= '2026-08-31')]
    calendar_verified = bool(len(expected_dates) and expected_dates.is_unique and expected_dates.is_monotonic_increasing
        and expected_dates.to_period('M').nunique() == 68
        and expected_dates[0] == pd.Timestamp('2021-01-04') and expected_dates[-1] == pd.Timestamp('2026-08-31'))
    missing_sessions = expected_dates.difference(dates).astype(str).tolist()
    valid = bool(len(dates) and dates.is_unique and dates.is_monotonic_increasing)
    return {'pass': bool(valid and bounds and not missing and calendar_verified and not missing_sessions
                        and dates.isin(expected_dates).all() and evidence_scope == 'prior_settled'),
            'missing_months': missing, 'boundary_sessions_present': bounds,
            'session_calendar_verified':calendar_verified,'missing_sessions':missing_sessions,
            'evidence_scope': evidence_scope, 'required_start': '2021-01-01',
            'required_end': '2026-08-31', 'required_months': 68}


def _drawdown(values):
    wealth = np.cumprod(1 + np.asarray(values))
    high = np.maximum.accumulate(np.r_[1., wealth])[1:]
    return float(np.max(1 - wealth / high))


def standard_table(daily, monthly, held, nav, turnover, rebalances, caps, execution, initial_nav):
    """Monthly arithmetic moments, daily beta/loss/drawdown, explicitly labelled."""
    strategy = frequency_metrics(monthly.total, monthly.cash, monthly.market, 12)
    benchmark_returns = monthly.cash + .04 / 12
    reference = frequency_metrics(benchmark_returns, monthly.cash, monthly.market, 12)
    daily_stats = frequency_metrics(daily.total, daily.cash, daily.market, 252)
    daily_benchmark = daily.cash + .04 / 252
    counts = (np.abs(held) > POSITION_EPS).sum(axis=1)
    gross, net = np.abs(held).sum(axis=1), held.sum(axis=1)
    active = monthly.total - benchmark_returns
    percent = lambda x: float(x) * 100 if x is not None else None
    uncertainty = lambda x, se: f'{percent(x):.4f} (+/- {percent(se):.4f})' if x is not None and se is not None else 'Unavailable'
    coefficient = lambda x, se: f'{x:.4f} (+/- {se:.4f})' if x is not None and se is not None else 'Unavailable'
    one_way = np.asarray(turnover) / 2
    rebalance_count = int(np.count_nonzero(rebalances))
    short_caps = caps[held < -POSITION_EPS]
    rows = [
        ('Ending value of the endowment', 'US$ billion', float(nav[-1]/1e9), float(initial_nav*np.prod(1+benchmark_returns)/1e9)),
        ('Cumulative return over the period', '%', percent(np.prod(1+daily.total)-1), percent(np.prod(1+benchmark_returns)-1)),
        ('Annualised return', '% per year; monthly arithmetic', percent(strategy['annualized_arithmetic_return']), percent(reference['annualized_arithmetic_return'])),
        ('CAGR', '% per year; monthly compounding', percent(strategy['cagr']), percent(reference['cagr'])),
        ('Annualised volatility', '% per year; monthly', percent(strategy['annualized_volatility']), percent(reference['annualized_volatility'])),
        ('Annualised alpha vs S&P 500', '% per year (+ s.e.); monthly HC1', uncertainty(strategy['annualized_alpha'],strategy['annualized_alpha_se']), 'n/a'),
        ('Beta vs S&P 500 - monthly', 'coefficient (+ s.e.); HC1', coefficient(strategy['beta'],strategy['beta_se']), 'n/a'),
        ('Beta vs S&P 500 - daily', 'coefficient (+ s.e.); HC1', coefficient(daily_stats['beta'],daily_stats['beta_se']), 'n/a'),
        ('Sharpe ratio', 'annualised; monthly', strategy['sharpe'], reference['sharpe']),
        ('Information ratio vs T-bill + 4%', 'annualised; monthly', strategy['net_information_ratio'], 'n/a'),
        ('Tracking error', '% per year; monthly', percent(active.std(ddof=1)*np.sqrt(12)), 'n/a'),
        ('Maximum drawdown', '%; daily / monthly benchmark', percent(_drawdown(daily.total)),percent(_drawdown(benchmark_returns))),
        ('Maximum one-month loss', '%; nonnegative loss magnitude', percent(max(0.,-monthly.total.min())),percent(max(0.,-benchmark_returns.min()))),
        ('Maximum one-day loss', '%; daily benchmark convention', percent(max(0.,-daily.total.min())),percent(max(0.,-daily_benchmark.min()))),
        ('Positions held', 'average / min / max', f'{counts.mean():.1f} / {counts.min()} / {counts.max()}', 'n/a'),
        ('Gross exposure', 'average / maximum, % NAV', f'{gross.mean()*100:.3f} / {gross.max()*100:.3f}', 'n/a'),
        ('Net exposure', 'average / min / max, % NAV', f'{net.mean()*100:.3f} / {net.min()*100:.3f} / {net.max()*100:.3f}', 'n/a'),
        ('One-way turnover', '% per rebalance / annualised (observed frequency)', f'{one_way[rebalances].mean()*100 if rebalance_count else 0:.3f} / {one_way.sum()*252/len(daily)*100:.3f}', 'n/a'),
        ('Average holding period', 'closed-notional-weighted trading days; open positions excluded', execution.get('average_closed_notional_holding_trading_days'), 'n/a'),
        ('Average position size', 'US$ million; daily position observations', float((np.abs(held)*nav[:,None]).sum()/max(1,counts.sum())/1e6), 'n/a'),
        ('Short book median market cap', 'US$ million; daily position observations', float(np.nanmedian(short_caps)) if len(short_caps) else None, 'n/a'),
        ('Trading cost', '% NAV per year', 'Not modelled', 'n/a'),
        ('Borrow cost', '% NAV per year', 'Not modelled', 'n/a'),
        ('Information ratio, net of all costs', 'annualised', 'Not modelled', 'n/a'),
    ]
    return pd.DataFrame(rows,columns=['metric','units','strategy','benchmark']).set_index('metric')


def earnings_attribution(dates, securities, contributions, events):
    if events is None:
        return {'status':'pending','reason':'No dated earnings calendar supplied; 8-K presence is not an earnings calendar'}
    required = {'permno','announcement_date'}
    if not required <= set(events):
        raise ValueError('Earnings calendar needs permno and announcement_date')
    mask = np.zeros(contributions.shape, bool)
    dates = pd.DatetimeIndex(dates)
    columns = {int(p): i for i,p in enumerate(securities)}
    for row in events.itertuples():
        if int(row.permno) not in columns:
            continue
        date = pd.Timestamp(row.announcement_date)
        if pd.isna(date):
            raise ValueError('Missing earnings announcement date')
        # Weekend announcements map to the next observed session.
        day = dates.searchsorted(date.normalize())
        if day >= len(dates) or date < dates[0] - pd.Timedelta(days=4):
            continue
        mask[max(0,day-1):min(len(dates),day+2),columns[int(row.permno)]] = True
    inside, outside = float(contributions[mask].sum()), float(contributions[~mask].sum())
    if not np.isclose(inside+outside, contributions.sum(),atol=1e-12):
        raise ValueError('Earnings attribution failed to reconcile')
    return {'status':'computed_coverage_unverified','earnings_contribution':inside,
            'ex_earnings_contribution':outside,'window_trading_sessions':[-1,1],
            'basis':'linked contribution to return on initial NAV; stock P&L only',
            'announcement_rows':len(events),'earnings_security_sessions':int(mask.sum()),
            'event_time_policy':'date-only calendar; no intraday timing claim'}


def _write_json(path, value):
    def clean(item):
        if isinstance(item, dict): return {k:clean(v) for k,v in item.items()}
        if isinstance(item, list): return [clean(v) for v in item]
        if isinstance(item, (np.floating,float)) and not np.isfinite(item): return None
        return item
    path.write_text(json.dumps(clean(value),indent=2,default=str,allow_nan=False),encoding='utf-8')


def write_bundle(output, dates, held, market, active_indices, securities, daily, monthly,
                 turnover, execution_targets, settings, policy, execution):
    """Write local evidence and a draft deck; external verification stays deferred."""
    output = Path(output)
    dates = pd.DatetimeIndex(dates)
    if not len(dates):
        raise ValueError('Submission reporting requires observed sessions')
    rebalances = np.array([int(i) in execution_targets for i in active_indices])
    sector_history, audit = audit_holdings(dates,held,market,active_indices,settings,set(execution_targets))
    sector_history.to_csv(output/'sector_exposures.csv',index=False)
    sector_history.loc[sector_history.is_rebalance].to_csv(output/'sector_exposures_rebalances.csv',index=False)
    volatility = volatility_check(daily.total,settings.get('volatility_limit',.12))
    calendar_path = policy.get('submission',{}).get('expected_sessions_path')
    expected_sessions = pd.read_csv(calendar_path)['date'] if calendar_path else None
    coverage = coverage_check(dates,policy.get('portfolio_evidence_scope','prior_settled'),expected_sessions)
    initial_nav = float(settings.get('fund_capital_usd',1e9))
    nav = initial_nav*np.cumprod(1+daily.total.to_numpy())
    caps, volume = np.full(held.shape,np.nan), np.full(held.shape,np.nan)
    tickers, companies = {}, {}
    for local, day in enumerate(active_indices):
        formation = int(market.formation[day])
        if market.frames is None or formation < 0: continue
        frame = market.frames[formation]
        if 'me' in frame: caps[local] = frame.me.to_numpy(float)
        if 'dolvol_126d' in frame: volume[local] = frame.dolvol_126d.to_numpy(float)
        for name, dest in [('ticker',tickers),('company_name',companies)]:
            if name in frame:
                dest[local] = frame[name].fillna('').astype(str).to_numpy()
    table = standard_table(daily,monthly,held,nav,turnover,rebalances,caps,execution,initial_nav)
    table.to_csv(output/'performance_table.csv',na_rep='Unavailable')
    counts = (np.abs(held)>POSITION_EPS).sum(axis=1)
    exposures = pd.DataFrame({'date':dates,'nav_usd':nav,'positions':counts,
        'gross':np.abs(held).sum(axis=1),'net':held.sum(axis=1),'one_way_turnover':np.asarray(turnover)/2})
    exposures.to_csv(output/'portfolio_history.csv',index=False)
    row,col = np.nonzero(np.abs(held)>POSITION_EPS)
    positions = pd.DataFrame({'date':dates[row],'permno':np.asarray(securities)[col],
        'weight':held[row,col],'ticker':[tickers.get(int(r),['']*len(securities))[c] for r,c in zip(row,col)],
        'company_name':[companies.get(int(r),['']*len(securities))[c] for r,c in zip(row,col)],
        'market_cap_millions':caps[row,col],'dollar_volume':volume[row,col],
        'position_days_adv':np.divide(np.abs(held[row,col])*nav[row],volume[row,col],
            out=np.full(len(row),np.nan),where=volume[row,col]>0)})
    positions.to_csv(output/'holdings_history.csv',index=False)
    # Canonical history is intentionally not advertised as an Arcana upload schema.
    average = pd.DataFrame({'permno':securities,'average_weight':held.mean(axis=0)})
    pd.concat([average.nlargest(10,'average_weight').assign(group='top'),
               average.nsmallest(10,'average_weight').assign(group='bottom')]).to_csv(output/'average_weight_extremes.csv',index=False)
    prior = np.zeros_like(held,dtype=float)
    # Holdings before the OOS boundary are included when the run has earlier trading.
    for local, day in enumerate(active_indices):
        if local and int(active_indices[local-1]) == int(day)-1: prior[local]=held[local-1]
    raw = market.returns[active_indices]
    contributions = prior*np.where(np.isfinite(raw)&(raw>=-1),raw,0)
    linked = contributions*np.r_[1.,np.cumprod(1+daily.total.to_numpy())[:-1]][:,None]
    earnings_path = policy.get('submission',{}).get('earnings_calendar_path')
    events = pd.read_csv(earnings_path) if earnings_path else None
    earnings = earnings_attribution(dates,securities,linked,events)
    _write_json(output/'earnings_attribution.json',earnings)
    contributors = pd.DataFrame({'permno':securities,'linked_return_contribution':linked.sum(axis=0)})
    pd.concat([contributors.nlargest(10,'linked_return_contribution').assign(group='positive'),
               contributors.nsmallest(10,'linked_return_contribution').assign(group='negative')]).to_csv(output/'return_contributors.csv',index=False)
    years = pd.DataFrame({'strategy':daily.total,'benchmark':daily.cash+.04/252,'sp500':daily.market}).groupby(dates.year).agg(lambda x:np.prod(1+x)-1)
    years['cost_status']='No trading or borrow costs modelled; before/after identical by assumption'
    years.to_csv(output/'calendar_year_returns.csv',index_label='year')
    # Historical monthly outcomes are useful local context, not Arcana stress scenarios.
    monthly[['total','market','cash']].to_csv(output/'historical_monthly_outcomes.csv',index_label='date')
    arcana = {'status':'deferred','reason':'User deferred Arcana; credentials unavailable',
              'required_tab':'HISTORICAL','upload_route':'Upload History',
              'upload_schema':'Unverified; obtain account example before converting canonical history',
              'required_exhibits_in_order':ARCANA_EXHIBITS,
              'stress_scenarios':{'status':'pending','required_minimum':51},
              'attribution_reconciliation':{'status':'pending'}}
    identifiers_complete = bool(len(positions) and positions.ticker.ne('').all() and positions.company_name.ne('').all())
    reporting_complete = bool(daily.market.notna().all() and identifiers_complete
        and positions.market_cap_millions.notna().all() and positions.position_days_adv.notna().all())
    constraints_pass = audit['pass'] and volatility['pass']
    result = {'constraints_pass':constraints_pass,'sector_compliance':audit,
        'volatility':volatility,'coverage':coverage,'arcana':arcana,'earnings':earnings,
        'reporting_inputs_complete':reporting_complete,'identifiers_complete':identifiers_complete,
        'submission_ready':False,
        'pending_requirements':['Arcana historical verification, exhibits and >50 scenarios',
            'Cost sensitivity and net-of-all-cost results: no cost model authorised',
            'Agent construction/risk counterfactual and worked decision example',
            'Final deck review and data-screening answers']}
    if not coverage['pass']: result['pending_requirements'].append('Complete untouched 68-month OOS coverage')
    if not reporting_complete: result['pending_requirements'].append('Complete identifiers, market, cap and ADV inputs')
    if earnings['status'] != 'verified': result['pending_requirements'].append('Dated earnings calendar with coverage verification')
    if not constraints_pass: result['pending_requirements'].append('Mandate failures: see sector and volatility audits')
    _write_json(output/'submission_status.json',result)
    _write_json(output/'arcana_requirements.json',arcana)
    _write_deck(output,table,exposures,sector_history,monthly,result)
    paths = ['config.json','experiment.json','daily_ledger.csv','actual_holdings.parquet',
             'source_snapshot/prompts.txt']
    paths += [p.name for p in output.iterdir() if p.is_file() and p.name in {
        'sector_exposures.csv','sector_exposures_rebalances.csv','performance_table.csv',
        'portfolio_history.csv','holdings_history.csv','average_weight_extremes.csv',
        'return_contributors.csv','calendar_year_returns.csv','historical_monthly_outcomes.csv',
        'earnings_attribution.json','submission_status.json','arcana_requirements.json','submission_deck.html'}]
    _write_json(output/'submission_manifest.json',{'artifacts':{
        name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in paths if (output/name).is_file()},
        'source_inputs':'See experiment.json for input/configuration/source hashes',
        'status':'draft_pending_external_verification'})
    return result


def _write_deck(output, table, exposures, sectors, monthly, status):
    """Portable, printable local review slides with linked evidence; no fake screenshots."""
    def slide(title, body): return f'<section><h1>{html.escape(title)}</h1>{body}</section>'
    def text(value): return '<p>'+html.escape(value)+'</p>'
    def chart(series, title):
        values=np.asarray(series,float)
        valid=np.isfinite(values)
        if not valid.any(): return text(title+': unavailable')
        low,high=float(values[valid].min()),float(values[valid].max())
        span=max(high-low,1e-9)
        points=' '.join(f'{30+i*840/max(1,len(values)-1):.1f},{220-(v-low)*170/span:.1f}' for i,v in enumerate(values) if np.isfinite(v))
        return f'<h3>{html.escape(title)}</h3><svg viewBox="0 0 900 250"><path d="M30 20V220H880" fill="none" stroke="#94a3b8"/><polyline points="{points}" fill="none" stroke="#087f8c" stroke-width="2"/><text x="30" y="242">{html.escape(str(exposures.date.iloc[0].date()))}</text><text x="750" y="242">{html.escape(str(exposures.date.iloc[-1].date()))}</text><text x="35" y="30">max {high:.3f}; min {low:.3f}</text></svg>'
    body=slide('Part 2 portfolio review',text('DRAFT — submission verification incomplete. This deck describes this campaign only.')+text(str(output))+text('Headline performance and investment claims require the full OOS acceptance checks.'))
    body+=slide('Investment process and methodology',text('Agent proposals are evaluated with recorded provenance. Supported strategies receive capital from prior-settled evidence; orders fill in full at supplied observed prices.')+text('Complete the investment thesis, selected signal rationale, split dates and the four data-screening answers from this campaign’s evidence and decision logs.'))
    body+=slide('Portfolio construction and risk',text(status['sector_compliance']['neutrality_definition'])+text('200–500 names; gross ≤200%; net ±50%; single name ≤5%; shorts ≥$500m at formation. Full-period daily realised volatility ceiling: 12%.')+text('No volatility targeting, trading cost or borrow cost model is applied.')+text(f"Daily realised volatility: {status['volatility']['annualized_realized_volatility']}; constraints pass: {status['constraints_pass']}"))
    body+=slide('Standardised performance table',table.to_html(escape=True,na_rep='Unavailable',border=0)+text('Monthly arithmetic moments use ×12 and √12. Daily beta and volatility audit use ×252 and √252. CAGR is separate.'))
    body+=slide('Portfolio history',chart(exposures.gross,'Gross weight / NAV')+chart(exposures.net,'Net weight / NAV')+chart(exposures.positions,'Position count'))
    sector_chart=''
    for code,rows in sectors.loc[sectors.sector!='unknown'].groupby('sector',sort=True):
        sector_chart+=chart(rows.net_weight,f'GICS {code}: signed net weight / NAV; limit ±0.20')
    body+=slide('Sector exposure history',sector_chart)
    body+=slide('Performance and holdings evidence',text('See daily_ledger.csv, calendar_year_returns.csv, average_weight_extremes.csv, return_contributors.csv and holdings_history.csv for auditable results.')+text('The holdings history includes market cap, dollar volume and position size in days of ADV, with source coverage disclosed.'))
    body+=slide('Earnings versus ex-earnings',text(json.dumps(status['earnings'],default=str))+text('Earnings windows are one observed trading session before through one after the dated announcement.'))
    body+=slide('Arcana historical verification — deferred',text('Account verification is deferred at the user’s request. No local chart is presented as an Arcana exhibit.')+'<ol>'+''.join('<li>'+html.escape(x)+'</li>' for x in ARCANA_EXHIBITS)+'</ol>')
    body+=slide('Stress, costs and agent evidence',text('Arcana’s >50 factor/macro/historical scenarios remain pending. Local monthly outcomes are context, not independent stress verification.')+text('Trading and borrow costs are not modelled. Cost sensitivity, agent counterfactual, decision log example and implementation discussion remain required.'))
    body+=slide('Readiness and limitations','<ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in status['pending_requirements'])+'</ul>')
    css='body{font:18px Arial;color:#102a43;background:#dbe4ed;margin:0}section{background:white;max-width:1100px;padding:48px;margin:24px auto;break-after:page}h1{font-size:34px;border-bottom:4px solid #087f8c;padding-bottom:12px}p,li{line-height:1.5}table{font-size:12px;width:100%;border-collapse:collapse}td,th{padding:5px;border-bottom:1px solid #ddd;text-align:left}svg{width:100%;max-height:190px}svg text{font-size:13px}@media print{body{background:white}section{margin:0;max-width:none;padding:20px}h1{font-size:26px}}'
    (output/'submission_deck.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Part 2 review — draft</title><style>'+css+'</style></head><body>'+body+'</body></html>',encoding='utf-8')
