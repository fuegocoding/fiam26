"""Local Part 2 exhibits from the same holdings and ledger as the table."""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .campaign_metrics import regression


def write_charts(output, daily, monthly, exposures, sectors, positions, average, contributors):
    """Return relative SVG paths; these are local analytics, never Arcana views."""
    paths = []

    def save(fig, name, title):
        fig.suptitle(title, fontsize=14)
        fig.tight_layout()
        path = 'exhibit_' + name + '.svg'
        fig.savefig(output / path, format='svg')
        plt.close(fig)
        paths.append(path)

    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                         'font.size': 9, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(10, 4))
    reference = (1 + monthly.cash + .04 / 12).cumprod()
    ax.plot(monthly.index, ((1 + monthly.total).cumprod() / reference - 1) * 100, label='Strategy / benchmark wealth - 1')
    ax.plot(monthly.index, ((1 + monthly.market).cumprod() / reference - 1) * 100, label='S&P 500 / benchmark wealth - 1')
    ax.axhline(0, color='grey', lw=.7)
    ax.set_ylabel('Cumulative relative wealth (%)'); ax.legend()
    save(fig, 'active_return', 'Cumulative active performance; monthly T-bill + 4% benchmark')

    rolling = []
    for end in range(252, len(daily) + 1):
        window = daily.iloc[end-252:end]
        row = regression(window.total-window.cash, window.market-window.cash)
        rolling.append({'date':daily.index[end-1], **row})
    pd.DataFrame(rolling, columns=['date','alpha','beta','beta_se','alpha_se','regression_observations']).to_csv(output/'rolling_beta.csv',index=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    if rolling:
        series = pd.DataFrame(rolling).set_index('date')
        ax.plot(series.index, series.beta.astype(float), label='Beta')
        ax.fill_between(series.index, (series.beta-1.96*series.beta_se).astype(float),
                        (series.beta+1.96*series.beta_se).astype(float), alpha=.2, label='95% HC1 interval')
        ax.legend()
    else:
        ax.text(.1,.5,'Fewer than 252 observed sessions; rolling beta unavailable',transform=ax.transAxes)
    ax.axhline(0,color='grey',lw=.7); ax.set_ylabel('Beta coefficient')
    save(fig, 'rolling_beta', '252-session rolling S&P 500 excess-return beta')

    yearly = pd.DataFrame({'Strategy (no modelled costs)':daily.total,
                           'T-bill + 4% (daily convention)':daily.cash+.04/252,
                           'S&P 500':daily.market}).groupby(daily.index.year).agg(lambda x: np.prod(1+x)-1)
    fig, ax = plt.subplots(figsize=(10, 4))
    (yearly*100).plot.bar(ax=ax); ax.set_ylabel('Calendar-year return (%)')
    save(fig, 'calendar_years', 'Calendar-year performance; trading and borrow costs not modelled')

    fig, axes = plt.subplots(3,1,figsize=(10,7),sharex=True)
    axes[0].plot(exposures.date,exposures.gross*100); axes[0].set_ylabel('Gross (% NAV)')
    axes[1].plot(exposures.date,exposures.net*100); axes[1].set_ylabel('Net (% NAV)')
    axes[2].plot(exposures.date,exposures.positions); axes[2].set_ylabel('Positions')
    save(fig, 'book', 'Actual daily portfolio exposures')

    fig, ax = plt.subplots(figsize=(10,4))
    for code, rows in sectors.loc[sectors.sector!='unknown'].groupby('sector'):
        ax.plot(pd.to_datetime(rows.date),rows.net_weight*100,label=str(code))
    ax.axhline(20,color='grey',ls='--'); ax.axhline(-20,color='grey',ls='--')
    ax.set_ylabel('Sector net (% NAV)'); ax.legend(ncol=6,fontsize=8)
    save(fig, 'sectors', 'GICS sector net exposure; hard cap +/-20% of NAV')

    def ranking(frame, column, name, title, unit):
        frame = frame.copy()
        labels = positions.drop_duplicates('permno').set_index('permno').ticker.to_dict()
        frame['label'] = [f"{labels.get(int(p), '')} ({int(p)})" for p in frame.permno]
        fig, ax = plt.subplots(figsize=(10,6))
        ax.barh(frame.label,frame[column]*100,color=np.where(frame[column]>=0,'#087f8c','#c45a40'))
        ax.set_xlabel(unit)
        save(fig,name,title)
    ranking(average.drop_duplicates('permno').sort_values('average_weight'),'average_weight','holdings',
            'Top and bottom holdings by average signed weight; all reporting sessions','Average signed weight (% NAV)')
    ranking(contributors.drop_duplicates('permno').sort_values('linked_return_contribution'),'linked_return_contribution','contributors',
            'Largest positive and negative linked return contributors','Contribution to cumulative return (percentage points)')

    shorts = positions.loc[positions.weight<0]
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for ax,column,label in zip(axes,['market_cap_millions','dollar_volume'],['Market cap (US$ million)','Dollar volume (US$)']):
        values=shorts[column].replace([np.inf,-np.inf],np.nan).dropna()
        values=values[values>0]
        if len(values):
            ax.hist(np.log10(values),bins=30)
        ax.set_xlabel('log10 '+label); ax.set_ylabel('Short position observations')
    axes[0].axvline(np.log10(500),color='#c45a40',ls='--',label='$500m floor'); axes[0].legend()
    save(fig,'short_book','Short-book market cap and dollar-volume distributions; daily holdings')

    fig, ax = plt.subplots(figsize=(10,4))
    for long,label in [(True,'Long'),(False,'Short')]:
        values=positions.loc[(positions.weight>0)==long,'position_days_adv'].dropna()
        values=values[(values>0)&np.isfinite(values)]
        if len(values): ax.hist(np.log10(values),bins=30,alpha=.5,label=label)
    ax.set_xlabel('log10 position size / average daily dollar volume (days)')
    ax.set_ylabel('Position observations')
    handles,_=ax.get_legend_handles_labels()
    if handles: ax.legend()
    save(fig,'days_adv','Position size in days of ADV; long and short separately')
    return paths
