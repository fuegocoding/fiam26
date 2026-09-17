"""Retrospective risk attribution: histories versus snapshots and calibrated EWMA.

This audit explains idea 39; it does not select another trading strategy.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from src.graph.walk_forward import split_masks, hac_mean
from src.time_series_search import ROOT, SNAPSHOT, UNIVERSES, atomic_json, ridge_solve


def run(base):
    destination = base / 'time_series_risk_audit'
    destination.mkdir(exist_ok=True)
    p = pd.read_parquet(base/'time_series_cost25/panel.parquet')
    audit = json.loads((base/'time_series_followups/model_39_audit.json').read_text())
    full = pd.read_parquet(base/'time_series_followups/model_39.parquet')
    state = list(dict.fromkeys(SNAPSHOT + ['fgap_p','fgap_disp','days_since_ann',
        'expected_report_next_month','rev3_next_p','exp_growth_p','sue_vol8_p']))
    snapshot = (p[state].groupby(p.eom).rank(pct=True)-.5).fillna(0).to_numpy(float)
    snapshot = np.column_stack([snapshot,p[state].isna().to_numpy(float)])
    ewma = np.log(p.ewma_vol.to_numpy(float)**2+.0001)
    # Raw log variance + missing flag; intercept learns the log-square/Jensen bias.
    calibrated = np.column_stack([np.nan_to_num(ewma,nan=0),~np.isfinite(ewma)])
    y = np.log(p.ret_exc_lead1m.to_numpy(float)**2+.0001)
    frames,fit_audits = [],[]
    for year in range(2021,2027):
        masks = split_masks(p.eom,1,year,label_dates=p.eom+pd.offsets.MonthEnd(1))
        train = masks['train'] & np.isfinite(y)
        val = masks['validation'] & np.isfinite(y)
        test_window = split_masks(p.eom,1,year)['test'] & (p.eom+pd.offsets.MonthEnd(1)).between('2021-01-31','2026-08-31')
        f = p.loc[test_window,['permno','eom','eligible','size_grp']].copy()
        f['date'] = f.eom+pd.offsets.MonthEnd(1)
        f['target'] = y[test_window]
        key = pd.MultiIndex.from_frame(f[['permno','eom']])
        f['trajectory'] = full[full.fold.eq(year) & full.window.eq('test')].set_index(['permno','eom']).prediction.reindex(key).to_numpy()
        f['ewma_uncalibrated'] = ewma[test_window]
        for name,x in [('snapshot',snapshot),('ewma_calibrated',calibrated)]:
            best = None
            for penalty in [.001,.01,.1,1.]:
                coef = ridge_solve(x[train],y[train],penalty)
                loss = np.mean((y[val]-coef[0]-x[val]@coef[1:])**2)
                if best is None or loss < best[0]: best = loss,penalty,coef
            f[name] = best[2][0]+x[test_window]@best[2][1:]
            fit_audits.append({'year':year,'model':name,'penalty':best[1],
                'train_rows':int(train.sum()),'validation_rows':int(val.sum()),
                'features':state if name=='snapshot' else ['log_ewma_variance','missing']})
        f['fold'] = year
        frames.append(f)
        print(f'Risk audit {year} fitted',flush=True)
    f = pd.concat(frames,ignore_index=True)
    f.to_parquet(destination/'predictions.parquet',index=False)
    atomic_json(destination/'fit_audit.json',fit_audits)
    monthly=[]
    for universe,sizes in {'all_formation':None,**UNIVERSES}.items():
        chosen = np.ones(len(f),bool) if sizes is None else f.eligible & f.size_grp.isin(sizes)
        for date,g in f[chosen].groupby('date'):
            # Compare every model on the same labeled sample, count unavailable outcomes.
            available = g[['trajectory','snapshot','ewma_calibrated','ewma_uncalibrated']].notna().all(axis=1)
            observed = available & g.target.notna()
            z=g[observed]
            for model in ['trajectory','snapshot','ewma_calibrated','ewma_uncalibrated']:
                monthly.append({'universe':universe,'date':date,'model':model,
                    'mse':float(((z.target-z[model])**2).mean()),
                    'rank_ic':float(z.target.corr(z[model],method='spearman')),
                    'rows':len(z),'unknown_outcomes':int((available & g.target.isna()).sum()),
                    'public_scores':int(available.sum())})
    monthly=pd.DataFrame(monthly)
    monthly.to_csv(destination/'monthly_metrics.csv',index=False)
    summaries=[]
    comparisons=[]
    for (universe,model),g in monthly.groupby(['universe','model']):
        ic=hac_mean(g.rank_ic,maxlags=6)
        measured = g[g.rows.gt(0) & g.mse.notna()]
        summaries.append({'universe':universe,'model':model,'rank_ic':ic['mean'],
            'ic_hac_t':ic['hac_tstat'],'mse':np.average(measured.mse,weights=measured.rows),
            'months':len(g),'rows':int(g.rows.sum()),
            'unknown_outcome_share':g.unknown_outcomes.sum()/g.public_scores.sum()})
    for universe,g in monthly.groupby('universe'):
        wide=g.pivot(index='date',columns='model',values=['mse','rank_ic'])
        for baseline in ['snapshot','ewma_calibrated','ewma_uncalibrated']:
            for measure in ['mse','rank_ic']:
                delta=(wide[measure][baseline]-wide[measure]['trajectory']) if measure=='mse' else (wide[measure]['trajectory']-wide[measure][baseline])
                stat=hac_mean(delta,maxlags=6)
                t=stat['hac_tstat']
                comparisons.append({'universe':universe,'baseline':baseline,'measure':measure,
                    'mean_improvement':stat['mean'],'hac_t':t,
                    'approximate_p':float(2*student_t.sf(abs(t),max(1,len(delta)-1))) if t is not None else 1.,
                    'positive_years':int((delta.groupby(delta.index.year).mean()>0).sum())})
    c=pd.DataFrame(comparisons).sort_values('approximate_p')
    c['holm_p']=np.minimum(1,np.maximum.accumulate(c.approximate_p.to_numpy()*np.arange(len(c),0,-1)))
    c.to_csv(destination/'paired_predictive_comparisons.csv',index=False)
    pd.DataFrame(summaries).to_csv(destination/'summary.csv',index=False)
    print(f'Risk attribution audit complete: {destination}',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-output',type=Path,default=ROOT/'output')
    args=ap.parse_args()
    run(args.base_output)

if __name__=='__main__': main()
