"""Measured replay attribution and separately generated daily fixture controls."""
from __future__ import annotations
import csv
import datetime as dt
import random
from decimal import Decimal
from pathlib import Path
from ..reporting.attribution import (HoldingsPeriod,SizingPeriod,PositionContribution,DecayCohort,
    holdings_factor_attribution,selection_vs_sizing,hit_rate_win_loss,alpha_decay_curve,regression_factor_attribution)
from ..reporting.daily import DailyPolicy,DailyInputs,DailyFill,DailyReturn,DailyRate,DailyMarket,reconcile_daily
from .common import atomic_write_json,read_json,SYNTHETIC_LABEL

def build_exhibits(run_dir,fixture,*,synthetic=False):
    root=Path(run_dir);monthly=read_json(root/'monthly.json');weights={}
    with (root/'holdings_by_period.csv').open() as handle:
        for row in csv.DictReader(handle):weights.setdefault(row['period'],{})[row['security_id']]=float(row['post_trade_weight'])
    periods=[];sizing=[];contributions=[];factor_returns={};exposures={}
    for row in monthly:
        label=row['period'];day=dt.date.fromisoformat(label);w=weights[label];returns=fixture.outcomes[day]
        periods.append(HoldingsPeriod(label,w,returns));sizing.append(SizingPeriod(label,w,returns))
        # Declared internal diagnostic factor: cross-sectional mean supplied return.
        factor_returns[label]={'cross_section_mean':sum(float(r) for r in returns.values() if r is not None)/sum(r is not None for r in returns.values())}
        for sid,weight in w.items():
            exposures[sid]={'cross_section_mean':1.}
            contributions.append(PositionContribution(sid,label,'long' if weight>0 else 'short',weight*float(returns[sid])))
    attr={'basis':'actual post-trade weights times supplied excess returns; excludes cash income and costs',
        'holdings':holdings_factor_attribution(periods,exposures,factor_returns,factor_set_label='internal_cross_section_mean'),
        'selection_sizing':selection_vs_sizing(sizing,seed=2608,n_draws=100),
        'hit_rate':hit_rate_win_loss(contributions,unit='NAV contribution')}
    # Regression needs a nonconstant diagnostic factor; undefined fits stay explicit.
    try:
        attr['regression']=regression_factor_attribution([m['period'] for m in monthly],
            [m['net_return']-m['rf_monthly'] for m in monthly],
            {'cross_section_mean':[factor_returns[m['period']]['cross_section_mean'] for m in monthly]},
            factor_set_label='internal_cross_section_mean')
    except ValueError as exc:attr['regression']={'status':'UNDEFINED','reason':str(exc)}
    attr['cost_sensitivity']={'method':'first-order return sensitivity, fixed realized holdings; no execution rerun',
        'scenarios':[{'cost_multiplier':m,'net_return_sum':sum(row['net_return']-(m-1)*row['cost_return'] for row in monthly)} for m in (1,2,3)],
        'borrow_return_sum':sum(row['borrow_return'] for row in monthly),
        'turnover_one_way':[row['turnover_one_way'] for row in monthly]}
    attr['stress']={'kind':'hypothetical fixed-holdings scenarios, not realized stress history',
        'scenarios':[{'period':p.label,'long_down20_short_up20':sum(-.2*abs(w) for w in p.weights.values()),
            'parallel_down20':sum(-.2*w for w in p.weights.values())} for p in periods]}
    attr['capacity']={'status':'BOUNDED_BY_FIXTURE_LIQUIDITY','assumptions':fixture.config.get('execution_policy'),
        'limitations':['Real borrow availability and impact beyond supplied liquidity are unvalidated.']}
    attr['earnings_split']={'status':'INCOMPLETE','reason':'No dated earnings-event calendar supplied.'}
    atomic_write_json(root/'attribution_report.json',attr)
    if synthetic:build_daily_fixture(root)
    else:atomic_write_json(root/'daily_report.json',{'status':'INCOMPLETE','reason':'Replay fixture has monthly outcomes only; supply dated daily observations.'})
    return attr


def build_daily_fixture(root):
    """Separate permitted daily test book. No daily path is inferred from monthly returns."""
    rng=random.Random(2610);start=dt.date(2016,1,4);days=[];day=start
    while len(days)<26:
        if day.weekday()<5:days.append(day)
        day+=dt.timedelta(days=1)
    names=[f'D{i:03}' for i in range(40)]
    signals={sid:(1 if i<20 else -1)*(1+(i%20)/20) for i,sid in enumerate(names)}
    returns={};market={}
    for day in days:
        market[day]=rng.gauss(0,.005)
        for sid in names:returns[sid,day]=.001*signals[sid]+.5*market[day]+rng.gauss(0,.005)
    selected=names[:4]+names[-4:]
    fills=[DailyFill(security_id=sid,on=days[0],quantity=Decimal('1250')*(1 if signals[sid]>0 else -1),
        price=Decimal(100),fee_usd=Decimal('25')) for sid in selected]
    report=reconcile_daily(DailyInputs(start_cash_usd=Decimal(1000000),fills=fills,
        returns=[DailyReturn(security_id=sid,on=day,total_return=Decimal(str(returns[sid,day]))) for sid in selected for day in days],
        risk_free=[DailyRate(on=day,tb3ms_annual_percent=3.) for day in days],
        market_returns=[DailyMarket(on=day,total_return=market[day]) for day in days]),
        DailyPolicy(trading_days=days,rolling_window=20))
    payload=report.as_dict();payload.update({'label':SYNTHETIC_LABEL,'book':'PERMITTED_DAILY_TEST_BASELINE',
        'relationship_to_monthly_replay':'independent generated observations; does not infer or reconcile a monthly daily path'})
    atomic_write_json(root/'daily_report.json',payload)
    cohorts=[DecayCohort(f'entry_{i}',days[i],signals) for i in (0,5,10,15)]
    decay=alpha_decay_curve(cohorts,returns,days,[(1,5),(6,10)],min_cohorts=4,min_names=30)
    decay['label']=SYNTHETIC_LABEL
    atomic_write_json(root/'alpha_decay.json',decay)
    atomic_write_json(root/'daily_observations.json',{'label':SYNTHETIC_LABEL,'calendar':[d.isoformat() for d in days],
        'returns':[{'security_id':sid,'date':day.isoformat(),'return':r} for (sid,day),r in sorted(returns.items())]})


def build_removal_controls(run_dir,fixture,*,certificate_policy=None):
    """Actual replay counterfactuals, with one admitted family removed at a time."""
    from dataclasses import replace
    from ..orchestration.replay import ControlledReplay
    from ..research.certificates import CertificatePolicy
    root=Path(run_dir);base=read_json(root/'run_summary.json');rows=[]
    families=sorted({f.strategy_id for f in fixture.forecasts})
    for i,family in enumerate(families):
        remaining=[f for f in fixture.forecasts if f.strategy_id!=family]
        if not remaining:
            rows.append({'removed':family,'status':'INCOMPLETE','reason':'No admitted family remains; cash book control not allocated.'});continue
        config={**fixture.config,'admitted_families':[f for f in families if f!=family]}
        fx=replace(fixture,config=config,forecasts=remaining)
        policy=certificate_policy or CertificatePolicy.production()
        result=ControlledReplay(fx,root/'removal_controls'/f'without_{i}',certificate_policy=policy).run()['summary']
        rows.append({'removed':family,'status':'COMPUTED','ending_nav':result['ending_nav'],
            'ending_nav_difference':float(result['ending_nav'])-float(base['ending_nav']),
            'evidence_artifact':f'removal_controls/without_{i}/run_summary.json'})
    result={'kind':'self-financing replay after removal of one family; same supplied market outcomes and execution policy',
        'baseline_ending_nav':base['ending_nav'],'rows':rows,
        'redundancy_note':'Synthetic admitted candidates share a planted signal; zero removal differences demonstrate redundancy, not independent alpha.',
        'limitations':['Return attribution to an identical strategy cannot identify causal contribution.']}
    atomic_write_json(root/'incremental_evidence.json',result)
    return result
