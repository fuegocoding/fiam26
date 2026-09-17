"""Independently reconcile saved research holdings with monthly returns and costs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.time_series_search import ROOT,atomic_json


def run(base):
    p=pd.read_parquet(base/'time_series_cost25/panel.parquet',columns=['permno','eom','ret_exc_lead1m','eligible'])
    lookup=p.set_index(['permno','eom'])
    result=[];errors=[]
    folders=[(f'time_series_cost{c}',c) for c in [10,25,50]]+[
        ('time_series_followups',None),('time_series_nonlinear',None),('time_series_full_characteristics',None)]
    for name,fixed_cost in folders:
        folder=base/name
        paths=sorted((folder/'results').glob('*.json')) if fixed_cost is not None else sorted(folder.glob('*_cost*.json'))
        for path in paths:
            r=json.loads(path.read_text());cost=fixed_cost if fixed_cost is not None else r['cost_bp']
            token=path.stem
            bf=folder/'books'/f'{token}.csv' if fixed_cost is not None else folder/f'{token}_book.csv'
            hf=folder/'books'/f'{token}_holdings.parquet' if fixed_cost is not None else folder/f'{token}_holdings.parquet'
            b=pd.read_csv(bf,parse_dates=['date']);h=pd.read_parquet(hf)
            h['eom']=h.date-pd.offsets.MonthEnd(1)
            keys=pd.MultiIndex.from_frame(h[['permno','eom']])
            h['outcome']=lookup.ret_exc_lead1m.reindex(keys).to_numpy()
            h['eligible']=lookup.eligible.reindex(keys).fillna(False).to_numpy()
            if not h.eligible.all():errors.append({'book':token,'folder':name,'error':'Held stock was ineligible at formation.'})
            grouped={d:g for d,g in h.groupby('date')};previous={};max_error=0.;invalid_months=0
            expected_dates=pd.date_range('2021-01-31','2026-08-31',freq='ME')
            if not pd.DatetimeIndex(b.date).equals(expected_dates):errors.append({'book':token,'folder':name,'error':'Wrong/duplicate test-month coverage.'})
            for row in b.itertuples():
                if not np.isfinite([row.cost,row.turnover,row.cash,row.gross_exposure,row.net_exposure]).all():
                    errors.append({'book':token,'folder':name,'date':str(row.date),'error':'Nonfinite saved accounting field.'})
                g=grouped.get(row.date)
                current={} if g is None else dict(zip(g.permno,g.weight))
                if g is not None and g.permno.duplicated().any():errors.append({'book':token,'folder':name,'error':'Duplicate saved holding.'})
                traded=sum(abs(current.get(k,0.)-previous.get(k,0.)) for k in current.keys()|previous.keys())
                estimated_cost=traded*cost/10000
                max_error=max(max_error,abs(row.cost-estimated_cost),abs(row.turnover-traded))
                long=sum(v>0 for v in current.values());short=sum(v<0 for v in current.values())
                gross=sum(abs(v) for v in current.values());net=sum(current.values())
                if long!=row.n_long or short!=row.n_short or not np.isclose(gross,row.gross_exposure,atol=1e-6) or not np.isclose(net,row.net_exposure,atol=1e-6):
                    errors.append({'book':token,'folder':name,'date':str(row.date),'error':'Saved holdings disagree with exposure/count columns.'})
                if gross>2.000001 or abs(net)>.500001:errors.append({'book':token,'folder':name,'error':'Exposure exceeds mandate.'})
                missing=0 if g is None else int(g.outcome.isna().sum())
                if missing!=row.missing_held_returns:errors.append({'book':token,'folder':name,'date':str(row.date),'error':'Unknown outcome count mismatch.'})
                if missing:
                    invalid_months+=1
                    if pd.notna(row.gross_excess) or pd.notna(row.net_excess):errors.append({'book':token,'folder':name,'error':'Unknown held return was assigned P&L.'})
                    previous=current.copy()
                else:
                    if not np.isfinite([row.gross_excess,row.net_excess]).all():
                        errors.append({'book':token,'folder':name,'date':str(row.date),'error':'Nonfinite P&L without an unknown held outcome.'})
                    pnl=0. if g is None else float((g.weight*g.outcome).sum())
                    max_error=max(max_error,abs(pnl-row.gross_excess),abs(pnl-estimated_cost-row.net_excess))
                    nav=1+row.cash+pnl-estimated_cost
                    previous={} if g is None else dict(zip(g.permno,g.weight*(1+g.outcome+row.cash)/nav))
            if max_error>2e-6:errors.append({'book':token,'folder':name,'error':'Independent P&L/total-return drift reconciliation failed.','maximum_error':max_error})
            result.append({'folder':name,'book':token,'cost_bp':cost,'months':len(b),
                'missing_return_months':invalid_months,'maximum_reconciliation_error':max_error,
                'cash_months':int((b.n_long+b.n_short).eq(0).sum())})
        print(f'Audited {len(paths)} books in {name}',flush=True)
    destination=base/'time_series_batch';destination.mkdir(exist_ok=True)
    pd.DataFrame(result).to_csv(destination/'book_reconciliation.csv',index=False)
    atomic_json(destination/'book_audit.json',{'books':len(result),'errors':errors,'passed':not errors,
        'checks':'68-month coverage; duplicate holdings; public eligibility; counts/exposures; missing outcomes; signed P&L; total-return drift; one-way costs.'})
    print(f'Book reconciliation: {len(result)} books, {len(errors)} errors',flush=True)
    if errors:raise RuntimeError('Saved book reconciliation failed; inspect book_audit.json.')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--base-output',type=Path,default=ROOT/'output')
    args=ap.parse_args();run(args.base_output)

if __name__=='__main__':main()
