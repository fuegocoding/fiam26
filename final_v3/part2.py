"""Part 2 mandate audits and explicit external-verification status."""
import json

import numpy as np
import pandas as pd

SECTORS = ('10','15','20','25','30','35','40','45','50','55','60')
SECTOR_LIMIT = .20
VOLATILITY_LIMIT = .12
NEUTRALITY = 'Hard sector net bounds +/-20% NAV and >=5 sectors; underlying selection count-matches sectors, then beta-tilts notionals. Exact dollar neutrality is not enforced.'


def sector_checks(positions):
    p = positions.copy()
    p['sector'] = p.sector.astype('string').str.replace(r'\.0$', '', regex=True).str[:2]
    p['sector'] = p.sector.where(p.sector.isin(SECTORS), 'unknown').fillna('unknown')
    p = p.loc[p.target_weight.abs() > 1e-10]
    p['long_weight'] = p.target_weight.clip(lower=0)
    p['short_weight'] = -p.target_weight.clip(upper=0)
    history = p.groupby(['return_month','sector']).agg(
        long_weight=('long_weight','sum'), short_weight=('short_weight','sum'),
        net_weight=('target_weight','sum'), positions=('target_weight','size'))
    months = pd.Index(positions.return_month.unique())
    history = history.reindex(pd.MultiIndex.from_product([months, [*SECTORS,'unknown']],
        names=['return_month','sector']), fill_value=0).reset_index()
    breadth = history.loc[history.sector.isin(SECTORS)].groupby('return_month').positions.apply(lambda x:int((x>0).sum()))
    checks = {
        'sector_breadth_at_least_five': bool(len(months) and breadth.ge(5).all()),
        'sector_net_at_most_20pct': bool(len(months) and history.net_weight.abs().le(SECTOR_LIMIT+1e-8).all()),
        'sector_classification_complete': bool(history.loc[history.sector=='unknown','positions'].sum()==0),
    }
    return history, checks


def enforce_book(frame, weights):
    p = frame.loc[frame.permno.isin(weights), ['permno','sector']].copy()
    p['target_weight'] = p.permno.map(weights)
    p['return_month'] = 0
    _, checks = sector_checks(p)
    if not all(checks.values()):
        raise ValueError('Part 2 infeasible portfolio: '+', '.join(k for k,v in checks.items() if not v))


def volatility_check(values):
    valid = values is not None and len(values)>1 and np.isfinite(values).all()
    vol = float(np.std(values,ddof=1)*np.sqrt(252)) if valid else None
    return {'volatility_within_12pct':bool(valid and vol<=VOLATILITY_LIMIT+1e-12),
            'annualised_realised_volatility_daily':vol,'volatility_ceiling':VOLATILITY_LIMIT,
            'volatility_basis':'daily net returns; sample std x sqrt(252)'}


def write_report(p, monthly, daily, out, stats):
    history, checks = sector_checks(p)
    history.to_csv(out/'sector_exposures.csv',index=False)
    complete_returns = daily is not None and not daily.missing_held_returns.any()
    volatility = volatility_check(daily.total if complete_returns else None)
    checks['volatility_within_12pct'] = volatility['volatility_within_12pct']
    expected = pd.period_range('2021-01','2026-08',freq='M')
    checks['full_oos_months'] = bool(monthly.index.equals(expected))
    checks['daily_returns_available'] = daily is not None and len(daily)>1
    checks['daily_held_returns_complete'] = bool(daily is not None and not daily.missing_held_returns.any())
    checks['sp500_coverage_complete'] = bool(daily is not None and daily.spx.notna().all())
    expected_days = None if daily is None else daily.attrs.get('expected_sessions')
    checks['full_oos_daily_sessions'] = bool(expected_days is not None
        and daily.index.equals(expected_days) and len(daily)
        and daily.index.min()==pd.Timestamp('2021-01-04')
        and daily.index.max()==pd.Timestamp('2026-08-31'))
    checks['daily_monthly_nav_reconciles'] = False
    if daily is not None:
        daily.to_csv(out/'daily_returns.csv',index_label='date')
        reconstructed = (1+daily.total).groupby(daily.index.to_period('M')).prod()-1
        reconciliation = monthly[['total']].join(reconstructed.rename('daily_compounded'))
        reconciliation['difference'] = reconciliation.daily_compounded-reconciliation.total
        reconciliation.to_csv(out/'daily_monthly_reconciliation.csv')
        checks['daily_monthly_nav_reconciles'] = bool(reconciliation.difference.notna().all() and reconciliation.difference.abs().le(1e-6).all())
        ex, market = daily.total-daily.rf, daily.spx-daily.rf
        rolling = ex.rolling(252).cov(market)/market.rolling(252).var()
        rolling.rename('beta_252_sessions').to_csv(out/'rolling_beta_daily.csv')
    status = {'checks':checks,**volatility,'neutrality_definition':NEUTRALITY,
        'submission_ready':False,'arcana_status':'deferred_by_user',
        'pending':['Arcana HISTORICAL full-history upload, summary and ten labelled exhibits',
                   'Arcana >50 scenarios and attribution reconciliation',
                   'Earnings versus ex-earnings analysis with verified event coverage',
                   'Final deck, data screening and agent counterfactual review'],
        'daily_accounting_note':'Daily returns charge monthly trading cost at first session and spread borrow cost across sessions; monthly reconciliation is mandatory before submission readiness.'}
    (out/'part2_status.json').write_text(json.dumps(status,indent=2)+'\n')
    stats.update(volatility)
    stats['neutrality_definition'] = NEUTRALITY
    stats['submission_ready'] = False
    standard_table(monthly,stats).to_csv(out/'standardised_performance.csv',index=False)
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
    fig,ax=plt.subplots(figsize=(10,4))
    for code, group in history.loc[history.sector!='unknown'].groupby('sector'):
        ax.plot(pd.PeriodIndex(group.return_month,freq='M').to_timestamp(),group.net_weight*100,label=code)
    ax.axhline(20,color='grey',ls='--'); ax.axhline(-20,color='grey',ls='--')
    ax.set_ylabel('Sector net exposure (% NAV)'); ax.legend(ncol=6)
    fig.tight_layout(); fig.savefig(out/'sector_exposures.png'); plt.close(fig)
    if daily is not None:
        fig,ax=plt.subplots(figsize=(10,4)); ax.plot(rolling.index,rolling)
        ax.set_ylabel('252-session beta to S&P 500'); ax.axhline(0,color='grey')
        fig.tight_layout(); fig.savefig(out/'rolling_beta_daily.png'); plt.close(fig)
    html = '<!doctype html><meta charset="utf-8"><h1>Part 2 review — draft</h1>'
    html += '<p>'+NEUTRALITY+'</p>'+standard_table(monthly,stats).to_html(index=False)
    html += '<img width="900" src="sector_exposures.png"><p>Arcana, stress scenarios, earnings attribution and final review remain pending. This is not a verified final deck.</p>'
    (out/'part2_review.html').write_text(html,encoding='utf-8')
    return checks


def standard_table(r,s):
    """Brief table with monthly arithmetic moments separated from CAGR."""
    def summary(series):
        wealth=(1+series).cumprod()
        return [float(wealth.iloc[-1]),float(wealth.iloc[-1]-1),float(series.mean()*12),
                float(wealth.iloc[-1]**(12/len(series))-1),float(series.std()*np.sqrt(12)),
                float((1-wealth/wealth.cummax().clip(lower=1)).max()),float(series.min())]
    a,b=summary(r.total),summary(r.benchmark)
    rows=[]
    for i,(name,units) in enumerate([('Ending endowment','US$ billion'),('Cumulative return','fraction'),('Annualised return','monthly arithmetic'),('CAGR','annual compound'),('Volatility','monthly std x sqrt(12)'),('Maximum drawdown','fraction'),('Worst month','signed return')]):
        rows.append((name,units,a[i],b[i]))
    fields=[('Alpha vs S&P 500','alpha_ann'),('Alpha standard error','alpha_ann_se'),
        ('Monthly beta','beta'),('Monthly beta standard error','beta_se'),('Daily beta','beta_daily'),
        ('Daily beta standard error','beta_daily_se'),('Sharpe','sharpe'),('Information ratio vs T-bill+4%','ir'),
        ('Tracking error','tracking_error'),('Worst day','max_one_day_loss'),('Average positions','positions_avg'),
        ('Minimum positions','positions_min'),('Maximum positions','positions_max'),('Average gross (% NAV)','gross_exposure_avg_pct'),
        ('Maximum gross (% NAV)','gross_exposure_max_pct'),('Average net (% NAV)','net_exposure_avg_pct'),
        ('Minimum net (% NAV)','net_exposure_min_pct'),('Maximum net (% NAV)','net_exposure_max_pct'),
        ('One-way turnover / rebalance','turnover_one_way_per_rebalance'),('Annualised one-way turnover','turnover_one_way_annualised'),
        ('Holding period (estimated trading days)','avg_holding_period_trading_days'),('Average position (US$ million)','position_size_avg_musd'),
        ('Short median market cap (US$ million)','short_median_market_cap_musd'),('Annual trading cost','cost_ann'),
        ('Annual borrow cost','borrow_ann'),('Information ratio net of costs','ir')]
    for name,key in fields: rows.append((name,'as labelled; returns are fractions',s.get(key,'Unavailable'),'n/a'))
    return pd.DataFrame(rows,columns=['Metric','Units','Strategy','Benchmark'])
