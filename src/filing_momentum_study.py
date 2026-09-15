"""Reproducible monthly 8-K momentum diagnostic; does not change production signals."""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.graph.walk_forward import split_masks

OUT = Path('output/filing_momentum'); OUT.mkdir(parents=True, exist_ok=True)
SECTORS = {'10':'Energy','15':'Materials','20':'Industrials','25':'Consumer discretionary',
 '30':'Consumer staples','35':'Health care','40':'Financials','45':'Technology',
 '50':'Communication services','55':'Utilities','60':'Real estate'}
ITEMS = {'1.01':'Material agreement','2.02':'Earnings','2.03':'Debt obligation',
 '3.02':'Unregistered equity sale','5.02':'Management or compensation','7.01':'Reg FD',
 '8.01':'Other events','2.05':'Restructuring','4.02':'Financial non-reliance','3.01':'Listing notice'}

def stats(x):
    x = np.asarray(x, float); valid=np.isfinite(x); n=int(valid.sum())
    if n<6: return (n, np.nan, np.nan, np.nan)
    mean=x[valid].mean(); u=np.where(valid,x-mean,0); v=u@u/n
    for lag in range(1,min(6,len(x)-1)+1):
        v += 2*(1-lag/7)*(u[lag:]@u[:-lag])/n
    se=np.sqrt(max(v,0)/n); t=mean/se if se else np.nan
    return n, mean, t, math.erfc(abs(t)/math.sqrt(2))

def main():
    cols=['permno','eom','date','ret','ret_exc','ret_exc_lead1m','me','me_company','prc','dolvol','gics','size_grp']
    c=pd.read_parquet('data/raw/chars_final_with_names.parquet',columns=cols)
    c['eom']=pd.to_datetime(c.eom); c['date']=pd.to_datetime(c.date)
    assert not c.duplicated(['permno','eom']).any()
    c['month']=c.eom.dt.year*12+c.eom.dt.month
    c['sector']=c.gics.str[:2].map(SECTORS).fillna('Unknown')
    c['cap']=pd.cut(c.me_company,[-np.inf,50,300,2000,10000,200000,np.inf],
        labels=['Nano <$50m','Micro $50–300m','Small $0.3–2bn','Mid $2–10bn','Large $10–200bn','Mega >$200bn']).astype(str)
    c.loc[c.me_company.isna(),'cap']='Unknown'
    lookup=c.set_index(['permno','month'])
    for k in range(-3,4):
        idx=pd.MultiIndex.from_arrays([c.permno,c.month+k])
        c[f'r{k}']=lookup.ret.reindex(idx).to_numpy()
    c['pre3']=(1+c['r-3'])*(1+c['r-2'])*(1+c['r-1'])-1
    c['during']=c.ret
    # Validate the supplied lead target against the exact next calendar month.
    idx=pd.MultiIndex.from_arrays([c.permno,c.month+1])
    next_exc=lookup.ret_exc.reindex(idx).to_numpy()
    both=np.isfinite(next_exc)&c.ret_exc_lead1m.notna().to_numpy()
    mismatch=int((np.abs(next_exc[both]-c.ret_exc_lead1m.to_numpy()[both])>1e-8).sum())
    assert mismatch==0, f'Lead target mismatch: {mismatch}'
    f=pd.read_parquet('data/raw/8k_20150101_20260831_identified.parquet',
        columns=['document_id','permno','filing_date','items'])
    assert f.document_id.is_unique
    f['filing_date']=pd.to_datetime(f.filing_date)
    f['eom']=f.filing_date+pd.offsets.MonthEnd(0)
    # Last-trading-date cutoff: exclude filings after the observed snapshot date.
    f=f.merge(c[['permno','eom','date']],on=['permno','eom'],how='left',validate='many_to_one')
    linked=int(f.date.notna().sum()); late=int((f.filing_date>f.date).sum())
    f=f[f.filing_date.le(f.date)].copy()
    agg=f.groupby(['permno','eom']).size().rename('filings').reset_index()
    c=c.merge(agg,on=['permno','eom'],how='left',validate='one_to_one')
    c['filings']=c.filings.fillna(0); c['event']=c.filings.gt(0)
    for item in ITEMS:
        keys=f[f['items'].map(lambda a: item in a if a is not None else False)][['permno','eom']].drop_duplicates()
        c[f'item_{item}']=pd.MultiIndex.from_frame(c[['permno','eom']]).isin(pd.MultiIndex.from_frame(keys))
    c['target']=c.eom+pd.offsets.MonthEnd(1)
    c['period']=np.select([c.target.dt.year<=2018,c.target.dt.year<=2020],['train','validation'],default='test')
    c['liquid']=c.prc.abs().ge(5)&c.dolvol.ge(c.groupby('eom').dolvol.transform(lambda x:x.quantile(.2)))
    c['y']=c.ret_exc_lead1m
    # Equal-weight size/sector contemporaneous peer benchmark, including non-filers.
    c['peer_y']=c.y-c.groupby(['eom','cap','sector']).y.transform('mean')
    groups=[('all','All',pd.Series(True,index=c.index)),('screen','Price/liquidity screened',c.liquid)]
    groups += [('cap',v,c.cap.eq(v)&c.liquid) for v in sorted(c.cap.unique()) if v!='Unknown']
    groups += [('sector',v,c.sector.eq(v)&c.liquid) for v in sorted(c.sector.unique()) if v!='Unknown']
    groups += [('item',i+' '+name,c.liquid) for i,name in ITEMS.items()]
    monthly=[]; paths=[]; coverage=[]
    for kind,name,mask in groups:
        d=c[mask].copy()
        flag=f'item_{name.split()[0]}' if kind=='item' else 'event'
        coverage.append(dict(kind=kind,bucket=name,stock_months=len(d),event_months=int(d[flag].sum()),securities=int(d[d[flag]].permno.nunique())))
        for k in range(-3,4):
            z=d[d[flag]].groupby(['period','eom'])[f'r{k}'].mean()
            for period, zz in z.groupby(level=0):
                n,mean,t,p=stats(zz); paths.append(dict(kind=kind,bucket=name,period=period,offset=k,months=n,mean=mean,t=t))
        for signal in ['pre3','during']:
            # Identical breakpoints for event and control stocks in each month/group.
            d['rank']=d.groupby('eom')[signal].rank(pct=True)
            d['side']=np.select([d['rank'].le(.3),d['rank'].ge(.7)],[-1,1],0)
            for eom,z in d.groupby('eom'):
                row=dict(kind=kind,bucket=name,signal=signal,eom=eom,target=eom+pd.offsets.MonthEnd(1))
                for event,label in [(True,'event'),(False,'control')]:
                    a=z[z[flag].eq(event)&z.y.notna()]
                    lo=a[a.side.eq(-1)]; hi=a[a.side.eq(1)]
                    row[label+'_n']=min(len(lo),len(hi))
                    complete=a[[signal,'peer_y']].dropna()
                    row['ic_'+label]=complete.rank().corr().iloc[0,1] if len(complete)>=10 else np.nan
                    for y,prefix in [('y',''),('peer_y','adjusted_')]:
                        row[prefix+label]=(hi[y].mean()-lo[y].mean()) if min(len(lo),len(hi))>=10 else np.nan
                    if event:
                        row['long']=hi.y.mean() if len(hi)>=10 else np.nan
                        row['short_stock_return']=lo.y.mean() if len(lo)>=10 else np.nan
                row['increment']=row['event']-row['control']
                row['adjusted_increment']=row['adjusted_event']-row['adjusted_control']
                row['ic_increment']=row['ic_event']-row['ic_control']
                monthly.append(row)
    m=pd.DataFrame(monthly)
    m=m[m.target.le(c.loc[c.y.notna(),'target'].max())]
    summaries=[]; folds=[]
    for keys,z in m.groupby(['kind','bucket','signal']):
        z=z.sort_values('eom')
        for period,mask in [('train',z.target.dt.year.le(2018)),('validation',z.target.dt.year.between(2019,2020)),('test',z.target.dt.year.ge(2021))]:
            a=z[mask]
            for metric in ['event','control','increment','adjusted_event','adjusted_increment','long','short_stock_return','ic_event','ic_increment']:
                n,mean,t,p=stats(a[metric]); summaries.append(dict(zip(['kind','bucket','signal'],keys),period=period,metric=metric,months=n,mean=mean,hac_t=t,p=p))
        # Annual selection uses only earlier training and rolling two-year validation.
        for year in range(2021,2027):
            masks=split_masks(z.eom,1,year)
            tr=stats(z.loc[masks['train'],'adjusted_increment']); va=stats(z.loc[masks['validation'],'adjusted_increment'])
            keep=tr[0]>=24 and abs(tr[2])>=2 and va[0]>=12 and np.sign(tr[1])==np.sign(va[1])
            test=z.loc[masks['test']]
            folds.append(dict(zip(['kind','bucket','signal'],keys),year=year,selected=bool(keep),direction=float(np.sign(tr[1])),train_t=tr[2],validation_mean=va[1],test_event_mean=test.event.mean(),test_increment_mean=test.adjusted_increment.mean()))
    s=pd.DataFrame(summaries)
    s['q']=np.nan
    for _,z in s.groupby(['period','metric']):
        good=z.p.dropna().sort_values(); n=len(good)
        if n: s.loc[good.index,'q']=np.minimum.accumulate((good.to_numpy()*n/np.arange(1,n+1))[::-1])[::-1].clip(0,1)
    s.to_csv(OUT/'summary.csv',index=False); m.to_csv(OUT/'monthly.csv',index=False)
    pd.DataFrame(paths).to_csv(OUT/'event_paths.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    fold=pd.DataFrame(folds)
    fold.to_csv(OUT/'walk_forward.csv',index=False)
    w=m[m.target.dt.year.ge(2021)].copy(); w['year']=w.target.dt.year
    w=w.merge(fold,on=['kind','bucket','signal','year'],validate='many_to_one')
    w['selected_increment']=np.where(w.selected,w.direction*w.adjusted_increment,0)
    wf=[]
    for keys,z in w.groupby(['kind','bucket','signal']):
        n,mean,t,p=stats(z.selected_increment)
        wf.append(dict(zip(['kind','bucket','signal'],keys),selected_years=int(z[z.selected].year.nunique()),months=n,mean=mean,hac_t=t))
    pd.DataFrame(wf).to_csv(OUT/'walk_forward_summary.csv',index=False)
    audit=dict(panel_rows=len(c),filings_total=373139,filings_linked=linked,filings_after_snapshot_excluded=late,
        eligible_filings=len(f),event_stock_months=int(c.event.sum()),lead_pairs_checked=int(both.sum()),lead_mismatches=mismatch,
        min_target=str(c.target[c.y.notna()].min()),max_target=str(c.target[c.y.notna()].max()),
        gics_missing=int(c.sector.eq('Unknown').sum()))
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    p=pd.DataFrame(paths)
    fig,ax=plt.subplots(figsize=(10,5))
    for name,z in p[(p.kind=='cap')&(p.period=='test')].groupby('bucket'):
        ax.plot(z.offset,100*z['mean'],marker='o',label=name)
    ax.axvline(0,color='black',ls='--',alpha=.4); ax.axhline(0,color='gray',lw=.7)
    ax.set(xlabel='Month relative to filing month (0 includes before and after filing)',ylabel='Mean monthly raw return (%)',title='8-K stock-month returns by company size, 2021–2026\nPrice/liquidity screened; descriptive, not causal')
    ax.legend(fontsize=8,ncol=2); fig.tight_layout(); fig.savefig(OUT/'event_paths.png',dpi=150)
    print(json.dumps(audit,indent=2)); print(s[(s.period=='test')&(s.metric=='adjusted_increment')].to_string(index=False))

if __name__=='__main__': main()
