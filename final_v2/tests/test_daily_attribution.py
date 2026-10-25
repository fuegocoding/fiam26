"""Hand-calculated accounting and attribution identities, not snapshot tests."""
import datetime as dt
from decimal import Decimal as D
import pytest
from pod_platform.reporting.daily import (DailyPolicy,DailyInputs,DailyFill,DailyReturn,DailyRate,DailyCashEntry,reconcile_daily,compound_periods,daily_beta)
from pod_platform.reporting.attribution import (HoldingsPeriod,SizingPeriod,PositionContribution,DecayCohort,
    holdings_factor_attribution,regression_factor_attribution,selection_vs_sizing,hit_rate_win_loss,alpha_decay_curve)
from pod_platform.contracts.errors import PodPlatformError
DAYS=[dt.date(2026,8,n) for n in (3,4,5)]

def test_daily_close_fills_costs_cash_borrow_and_compounding():
    # $1000 starts in cash. Close buy 5@$100, fee $1 -> NAV999.
    # Next close stock110 => +50; borrow2 -> NAV1047.
    # Last stock99 => -55; sell all@$99, fee1 -> final991.
    report=reconcile_daily(DailyInputs(start_cash_usd=D(1000),
        fills=[DailyFill(security_id='A',on=DAYS[0],quantity=D(5),price=D(100),fee_usd=D(1)),
            DailyFill(security_id='A',on=DAYS[2],quantity=D(-5),price=D(99),fee_usd=D(1))],
        returns=[DailyReturn(security_id='A',on=DAYS[1],total_return=D('.1')),
            DailyReturn(security_id='A',on=DAYS[2],total_return=D('-.1'))],
        cash_entries=[DailyCashEntry(on=DAYS[1],kind='borrow',debit_usd=D(2),evidence_id='borrow_rate')],
        risk_free=[DailyRate(on=d,tb3ms_annual_percent=0) for d in DAYS]),DailyPolicy(trading_days=DAYS))
    assert report.nav_path==pytest.approx([999,1047,991])
    growth=1.
    for r in report.returns:growth*=1+r
    assert growth-1==pytest.approx(-.009)
    assert all(abs(float(r['nav_residual']))<1e-8 for r in report.rows)

def test_daily_held_missing_returns_fails_and_no_rate_fill():
    args=DailyInputs(start_cash_usd=D(500),initial_positions={'A':(D(5),D(100))},
        risk_free=[DailyRate(on=d,tb3ms_annual_percent=0) for d in DAYS])
    with pytest.raises(PodPlatformError,match='held name'):reconcile_daily(args,DailyPolicy(trading_days=DAYS))
    with pytest.raises(PodPlatformError,match='TB3MS'):
        reconcile_daily(DailyInputs(start_cash_usd=D(1000)),DailyPolicy(trading_days=DAYS))

def test_cash_income_with_short_proceeds_and_explicit_borrow():
    report=reconcile_daily(DailyInputs(start_cash_usd=D(1000),
        initial_positions={'A':(D(-5),D(100))},
        returns=[DailyReturn(security_id='A',on=d,total_return=D(0)) for d in DAYS],
        risk_free=[DailyRate(on=d,tb3ms_annual_percent=2.52) for d in DAYS],
        cash_entries=[DailyCashEntry(on=DAYS[1],kind='borrow',debit_usd=D(1),evidence_id='b')]),DailyPolicy(trading_days=DAYS))
    expected=D(1000)
    navs=[]
    for i in range(3):
        expected*=D('1.0001')
        if i==1:expected-=D(1)
        navs.append(float(expected-D(500)))
    assert report.nav_path==pytest.approx(navs)

def test_factor_and_idiosyncratic_reconcile_hand_calculation():
    # .6*.10 + (-.4)*(-.05)=.08. Factor=(.6*1-.4*2)*.03=-.006.
    result=holdings_factor_attribution([HoldingsPeriod('m',{'A':.6,'B':-.4},{'A':.1,'B':-.05})],
        {'A':{'market':1},'B':{'market':2}},{'m':{'market':.03}},factor_set_label='internal_market')
    row=result['periods'][0]
    assert row['total']==pytest.approx(.08)
    assert row['factor_total']==pytest.approx(-.006)
    assert row['idiosyncratic']==pytest.approx(.086)
    assert not result['standard_factor_definitions']

def test_regression_known_coefficients_and_missing_exposures():
    factors=[-.03,.02,.05,-.01,.04]
    y=[.001+1.5*x for x in factors]
    result=regression_factor_attribution(list('abcde'),y,{'market':factors},factor_set_label='diagnostic')
    assert result['alpha_per_period']==pytest.approx(.001)
    assert result['betas']['market']==pytest.approx(1.5)
    assert all(abs(r['idiosyncratic_residual'])<1e-10 for r in result['rows'])
    with pytest.raises(PodPlatformError):
        holdings_factor_attribution([HoldingsPeriod('m',{'A':1},{'A':.1})],{}, {'m':{'market':.03}},factor_set_label='internal')

def test_selection_sizing_and_win_loss_hand_calculation():
    result=selection_vs_sizing([SizingPeriod('m',{'A':.4,'B':.1,'C':-.5},{'A':.1,'B':-.1,'C':-.04})],seed=42,n_draws=100)
    row=result['periods'][0]
    assert row['actual']==pytest.approx(.05)
    assert row['selection']==pytest.approx(.02)
    assert row['sizing']==pytest.approx(.03)
    hit=hit_rate_win_loss([PositionContribution('A','m','long',.04),PositionContribution('B','m','long',-.01),
        PositionContribution('C','m','short',.02)],unit='NAV contribution')
    assert hit['all']['hit_rate']==pytest.approx(2/3)
    assert hit['all']['win_loss_ratio']==pytest.approx(3)

def test_alpha_decay_executable_entry_and_compounding():
    signals={'A':1,'B':-1,'C':2}
    returns={(s,DAYS[0]):9 for s in signals} # entry day never earned
    returns.update({('A',DAYS[1]):.1,('B',DAYS[1]):-.1,('C',DAYS[1]):.2,
        ('A',DAYS[2]):-.1,('B',DAYS[2]):.1,('C',DAYS[2]):-.2})
    result=alpha_decay_curve([DecayCohort('cohort',DAYS[0],signals)],returns,DAYS,[(1,1),(2,2),(1,2)],min_cohorts=1,min_names=3)
    assert result['status']=='COMPUTED'
    assert result['windows'][0]['mean_signed_payoff']==pytest.approx(.4/3)
    assert result['windows'][2]['mean_signed_payoff']==pytest.approx(-.04/3)
    incomplete=alpha_decay_curve([DecayCohort('cohort',DAYS[-1],signals)],returns,DAYS,[(1,2)],min_cohorts=1,min_names=3)
    assert incomplete['status']=='INSUFFICIENT'
    del returns[('A',DAYS[1])]
    with pytest.raises(PodPlatformError):
        alpha_decay_curve([DecayCohort('cohort',DAYS[0],signals)],returns,DAYS,[(1,2)],min_cohorts=1,min_names=3)

def test_earnings_split_uses_explicit_calendar_and_reconciles():
    from pod_platform.reporting.attribution import earnings_vs_ex_earnings
    contributions=[PositionContribution('A',d.isoformat(),'long',value) for d,value in zip(DAYS,[.01,-.02,.03])]
    result=earnings_vs_ex_earnings(contributions,{'A':[DAYS[1]]},DAYS,window=(0,0))
    assert result['earnings_contribution']==pytest.approx(-.02)
    assert result['ex_earnings_contribution']==pytest.approx(.04)
    assert result['total']==pytest.approx(.02)
    with pytest.raises(ValueError,match='calendar supplied'):
        earnings_vs_ex_earnings(contributions,{},DAYS)
