"""Matched predictive attribution for snapshot/trajectory pairs, including liquid universes."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from src.graph.walk_forward import hac_mean
from src.time_series_search import ROOT,UNIVERSES,rank_month


def run(base):
    out=base/'time_series_prediction_audit';out.mkdir(exist_ok=True)
    p=pd.read_parquet(base/'time_series_cost25/panel.parquet')
    p['date']=p.eom+pd.offsets.MonthEnd(1)
    p=p[p.date.between('2021-01-31','2026-08-31')].reset_index(drop=True)
    p['surprise_rank']=rank_month(p,p.target_surprise_p)
    p['log_risk']=np.log(p.ret_exc_lead1m**2+.0001)
    pairs=[('ridge_return','29','30','ret_exc_lead1m','time_series_cost25'),
           ('ridge_surprise','31','32','surprise_rank','time_series_cost25')]
    if (base/'time_series_nonlinear/model_N39T.parquet').exists():
        pairs += [('xgb_return','N29','N30','ret_exc_lead1m','time_series_nonlinear'),
                  ('xgb_surprise','N31','N32','surprise_rank','time_series_nonlinear'),
                  ('xgb_risk','N39S','N39T','log_risk','time_series_nonlinear')]
    if (base/'time_series_full_characteristics/model_F32.parquet').exists():
        pairs += [('full_xgb_return','F29','F30','ret_exc_lead1m','time_series_full_characteristics'),
                  ('full_xgb_surprise','F31','F32','surprise_rank','time_series_full_characteristics')]
    key=pd.MultiIndex.from_frame(p[['permno','eom']]);monthly=[]
    for name,snapshot,trajectory,target,folder in pairs:
        for model in [snapshot,trajectory]:
            f=pd.read_parquet(base/folder/f'model_{model}.parquet')
            f=f[f.window.eq('test')]
            if f.duplicated(['permno','eom']).any():raise ValueError('Duplicate test forecasts')
            p[model]=f.set_index(['permno','eom']).prediction.reindex(key).to_numpy()
        for universe,sizes in {'all_formation':None,**UNIVERSES}.items():
            eligible=np.ones(len(p),bool) if sizes is None else p.eligible & p.size_grp.isin(sizes)
            for date,g in p[eligible].groupby('date'):
                public=g[[snapshot,trajectory]].notna().all(axis=1)
                observed=public & g[target].notna()
                z=g[observed]
                if len(z)<20:continue
                for role,model in [('snapshot',snapshot),('trajectory',trajectory)]:
                    monthly.append({'pair':name,'universe':universe,'date':date,'role':role,'model':model,
                        'rank_ic':z[target].corr(z[model],method='spearman'),
                        'mse':((z[target]-z[model])**2).mean(),'rows':len(z),
                        'unknown_labels':int((public & g[target].isna()).sum()),'public_scores':int(public.sum())})
    m=pd.DataFrame(monthly);m.to_csv(out/'monthly_metrics.csv',index=False)
    summary=[];comparison=[]
    for (name,universe,role,model),g in m.groupby(['pair','universe','role','model']):
        s=hac_mean(g.rank_ic,maxlags=6)
        summary.append({'pair':name,'universe':universe,'role':role,'model':model,
            'mean_rank_ic':s['mean'],'rank_ic_hac_t':s['hac_tstat'],
            'mse':np.average(g.mse,weights=g.rows),'months':len(g),
            'unknown_label_share':g.unknown_labels.sum()/g.public_scores.sum()})
    for (name,universe),g in m.groupby(['pair','universe']):
        w=g.pivot(index='date',columns='role',values=['rank_ic','mse'])
        for measure in ['rank_ic','mse']:
            d=(w[measure].trajectory-w[measure].snapshot) if measure=='rank_ic' else (w[measure].snapshot-w[measure].trajectory)
            s=hac_mean(d,maxlags=6);t=s['hac_tstat']
            comparison.append({'pair':name,'universe':universe,'measure':measure,
                'mean_improvement':s['mean'],'hac_t':t,'positive_years':int((d.groupby(d.index.year).mean()>0).sum()),
                'approximate_p':float(2*student_t.sf(abs(t),max(1,len(d)-1))) if t is not None else 1.})
    c=pd.DataFrame(comparison).sort_values('approximate_p')
    c['holm_p']=np.minimum(1,np.maximum.accumulate(c.approximate_p.to_numpy()*np.arange(len(c),0,-1)))
    c.to_csv(out/'paired_predictive_comparisons.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    print(f'Prediction attribution complete: {out}',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--base-output',type=Path,default=ROOT/'output')
    args=ap.parse_args();run(args.base_output)

if __name__=='__main__':main()
