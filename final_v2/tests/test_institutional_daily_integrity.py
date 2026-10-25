"""Hand-accounted daily analytics and missing-data/event-calendar regressions."""
import numpy as np
import pandas as pd
import pytest

from pod_platform.accounting.evaluation import daily_alpha_decay, evaluate_daily_spine, earnings_window_decomposition


def data(returns=(.1, 0.), sid=1):
    return pd.DataFrame({'day':pd.bdate_range('2021-01-04', periods=len(returns)),
        'permno':[sid]*len(returns),'daily_return':returns})


def rates(pct=25.2):
    return pd.DataFrame({'series':['TB3MS','SP500','SP500'],
        'date':pd.to_datetime(['2021-01-01','2021-01-01','2021-01-04']),
        'value':[pct,100.,101.]})


def test_total_returns_do_not_earn_cash_interest_twice():
    out, summary=evaluate_daily_spine({pd.Period('2021-01'):{1:1.}},data(),rates())
    assert list(out['total'])==pytest.approx([.1,0.])
    assert out['wealth'].iloc[-1]==pytest.approx(1.1)
    assert summary['actual_dated_execution_ledger'] is False
    assert summary['missing_market_returns']==1  # No fabricated flat market observation.


def test_short_proceeds_earn_cash_rate_and_weights_drift_self_financing():
    out,_=evaluate_daily_spine({pd.Period('2021-01'):{1:-1.}},data((-.1,0.)),rates())
    # Start NAV1, short liability-1, cash2. Day1 asset PnL+.1, cash income+.002.
    # Day2 liability unchanged-.9, cash2.002 earns .002002.
    assert out['wealth'].iloc[0]==pytest.approx(1.102)
    assert out['wealth'].iloc[1]==pytest.approx(1.104002)


def test_monthly_modeled_costs_allocate_dollars_not_repeated_nav_fractions():
    month=pd.Period('2021-01')
    costs=pd.DataFrame({'cost':[.02],'borrow':[0.]},index=pd.PeriodIndex([month]))
    out,summary=evaluate_daily_spine({month:{1:.5}},data((0.,0.)),rates(0),costs)
    assert out['wealth'].iloc[-1]==pytest.approx(.98)
    assert 'modeled' in summary['monthly_cost_timing']


@pytest.mark.parametrize('case',['missing_return','missing_security','missing_month','missing_calendar_day','invalid_return'])
def test_missing_or_invalid_held_observations_never_disappear(case):
    frame=data();book={pd.Period('2021-01'):{1:.5}}
    if case=='missing_return':frame.loc[0,'daily_return']=np.nan
    elif case=='missing_security':book={pd.Period('2021-01'):{99:.5}}
    elif case=='missing_month':book={pd.Period('2021-02'):{1:.5}}
    elif case=='missing_calendar_day':frame.attrs['trading_calendar']=list(pd.bdate_range('2021-01-04',periods=3))
    else:frame.loc[0,'daily_return']=-1.01
    with pytest.raises(ValueError):evaluate_daily_spine(book,frame,rates())


def decay_inputs():
    dates=pd.bdate_range('2021-01-04',periods=3)
    frame=pd.DataFrame([{'permno':p,'day':day,'daily_return':(-1. if p>40 and day==dates[1] else 0.)}
        for day in dates for p in range(1,51)])
    preds=pd.DataFrame({'permno':range(1,51),'return_month':['2021-01-01']*50,'final_score':range(1,51)})
    return frame,preds


def test_decay_preserves_total_loss_and_respects_entry_delay():
    frame,preds=decay_inputs();out=daily_alpha_decay(frame,preds,horizons_days=(1,))
    assert out['quintile_spread_1d_cum_bps']==pytest.approx(-10000)
    assert out['quintile_spread_1d_cohorts']==1
    assert out['quintile_spread_1d_t'] is None


def test_decay_missing_outcome_excludes_whole_cohort_with_disclosure():
    frame,preds=decay_inputs();frame.loc[(frame['permno']==50)&(frame['day']==pd.Timestamp('2021-01-05')),'daily_return']=np.nan
    out=daily_alpha_decay(frame,preds,horizons_days=(1,))
    assert out['quintile_spread_1d_cohorts']==0
    assert out['quintile_spread_1d_missing_cohorts']==1
    assert out['quintile_spread_1d_cum_bps'] is None


def test_earnings_window_is_four_trading_days_not_calendar_days(monkeypatch):
    frame=pd.DataFrame({'day':pd.bdate_range('2021-01-07',periods=5),'permno':[1]*5,'daily_return':[0.]*5})
    ext=pd.DataFrame([
        {'table':'ibes_link','TICKER':'SYNTHETIC','PERMNO':1,'SCORE':1,'sdate':'2020-01-01','edate':'2022-01-01'},
        {'table':'ibes_surprise','TICKER':'SYNTHETIC','USFIRM':1,'MEASURE':'EPS','FISCALP':'QTR','anndats':'2021-01-08'}])
    monkeypatch.setattr(pd,'read_parquet',lambda *a,**kw:ext)
    out=earnings_window_decomposition({pd.Period('2021-01'):{1:.5}},frame,'unused')
    assert out['earnings_window_stock_day_fraction']==pytest.approx(.8)
    assert 'ex-post' in out['basis']
