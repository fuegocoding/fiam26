"""Batched full-body pattern audit and cached FinBERT conditional momentum study.

Text flags are transparent pattern matches, not semantic LLM classifications.
"""
import argparse
import json
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from src.filing_momentum_study import SECTORS, stats
from src.graph.walk_forward import split_masks

OUT=Path('output/filing_momentum')
PATTERNS={
 'guidance_up':r'(?:rais\w*|increas\w*)\s+(?:(?:its|the|full.year|annual|quarterly|financial|earnings|revenue|202\d|201\d)\s+){0,5}(?:guidance|outlook)|(?:guidance|outlook)\s+(?:was |is |has been )?(?:raised|increased)',
 'guidance_down':r'(?:lower\w*|reduc\w*|withdraw\w*|suspend\w*)\s+(?:(?:its|the|full.year|annual|quarterly|financial|earnings|revenue|202\d|201\d)\s+){0,5}(?:guidance|outlook)|(?:guidance|outlook)\s+(?:was |is |has been )?(?:lowered|reduced|withdrawn|suspended)',
 'buyback':r'(?:authoriz\w*|approv\w*|announc\w*)[^.;]{0,120}(?:share|stock) repurchase|(?:share|stock) repurchase (?:program|plan)',
 'dividend_increase':r'(?:increas\w*|rais\w*)[^.;]{0,70}(?:quarterly |cash |regular )?dividend',
 'dividend_cut':r'(?:reduced|cut|suspended|eliminated) (?:its |the |their )?(?:quarterly |cash |regular |common stock ){0,3}dividend',
 'distress':r'substantial doubt[^.;]{0,130}going concern|(?:has |have )?filed (?:a |for |voluntary |petitions? |under |relief ){1,6}(?:chapter 11|bankruptcy)|(?:company|registrant) (?:has |had )?failed to (?:pay|make)[^.;]{0,65}(?:interest|principal)',
 'non_reliance':r'(?:should|can|could) no longer be relied (?:upon|on)',
 'restructuring':r'(?:approv\w*|announc\w*|implement\w*)[^.;]{0,100}restructuring|reduction in (?:its |the )?workforce',
 'acquisition':r'(?:entered into|executed)[^.;]{0,100}(?:merger agreement|agreement and plan of merger)|completed (?:the |its |an )?acquisition',
 'capital_financing':r'(?:entered into|announced|completed|closed)[^.;]{0,120}(?:public offering|private placement|registered direct offering)',
}
RX={k:re.compile(r'\b(?:'+v+r')\b',re.I) for k,v in PATTERNS.items()}

def extract(limit):
    cache=OUT/('content_features.parquet' if not limit else 'content_pilot.parquet')
    if cache.exists(): return pd.read_parquet(cache)
    rows=[]; examples={k:[] for k in RX}; start=time.time(); chars=0; truncated=0
    for batch in pq.ParquetFile('data/raw/8k_20150101_20260831_identified.parquet').iter_batches(
            batch_size=512,columns=['document_id','permno','filing_date','text']):
        for a in batch.to_pylist():
            body=a.pop('text') or ''; chars+=len(body); truncated+=int(len(body)>12000)
            for label,rx in RX.items():
                match=rx.search(body); a[label]=bool(match)
                if match and len(examples[label])<8:
                    examples[label].append(dict(document_id=a['document_id'],excerpt=body[max(0,match.start()-100):match.end()+160]))
            rows.append(a)
            if limit and len(rows)>=limit: break
        if len(rows)%10240==0: print(f'Scanned {len(rows):,} complete filing bodies in {time.time()-start:.0f}s',flush=True)
        if limit and len(rows)>=limit: break
    f=pd.DataFrame(rows); f.to_parquet(cache,index=False)
    stem='content_pilot' if limit else 'content'
    (OUT/f'{stem}_examples.json').write_text(json.dumps(examples,indent=2),encoding='utf-8')
    audit=dict(filings=len(rows),characters=chars,over_finbert_12000_character_cap=truncated,seconds=time.time()-start,counts={k:int(f[k].sum()) for k in RX})
    (OUT/f'{stem}_audit.json').write_text(json.dumps(audit,indent=2)); print(audit,flush=True)
    return f

def main(limit):
    OUT.mkdir(exist_ok=True,parents=True); f=extract(limit)
    if limit:return
    scores=pd.read_parquet('output/finbert_scores.parquet',columns=['document_id','sentiment','n_chunks'])
    assert scores.document_id.is_unique and f.document_id.is_unique
    f=f.merge(scores,on='document_id',validate='one_to_one',how='left')
    f.loc[f.n_chunks.fillna(0).le(0),'sentiment']=np.nan
    c=pd.read_parquet('data/raw/chars_final_with_names.parquet',columns=['permno','eom','date','ret','ret_exc_lead1m','me_company','prc','dolvol','gics'])
    c['eom']=pd.to_datetime(c.eom); c['date']=pd.to_datetime(c.date); c['month']=c.eom.dt.year*12+c.eom.dt.month
    c['cap']=pd.cut(c.me_company,[-np.inf,50,300,2000,10000,200000,np.inf],labels=['Nano','Micro','Small','Mid','Large','Mega']).astype(str)
    c['sector']=c.gics.str[:2].map(SECTORS).fillna('Unknown'); c['target']=c.eom+pd.offsets.MonthEnd(1)
    look=c.set_index(['permno','month']).ret
    for k in [-3,-2,-1,1,2,3]:
        c[f'r{k}']=look.reindex(pd.MultiIndex.from_arrays([c.permno,c.month+k])).to_numpy()
    c['pre3']=(1+c['r-3'])*(1+c['r-2'])*(1+c['r-1'])-1; c['during']=c.ret
    f['filing_date']=pd.to_datetime(f.filing_date); f['eom']=f.filing_date+pd.offsets.MonthEnd(0)
    f=f.merge(c[['permno','eom','date']],on=['permno','eom'],how='left',validate='many_to_one')
    f=f[f.filing_date.le(f.date)]
    agg=f.groupby(['permno','eom']).agg({**{k:'max' for k in RX},'sentiment':'mean'})
    c=c.merge(agg,on=['permno','eom'],how='inner',validate='one_to_one')
    # Screen thresholds come from the full panel, not filing stocks only.
    allc=pd.read_parquet('data/raw/chars_final_with_names.parquet',columns=['eom','dolvol'])
    allc['eom']=pd.to_datetime(allc.eom); floor=allc.groupby('eom').dolvol.quantile(.2)
    c=c[c.prc.abs().ge(5)&c.dolvol.ge(c.eom.map(floor))].copy()
    c['y']=c.ret_exc_lead1m
    c['peer_y']=c.y-c.groupby(['eom','cap','sector']).y.transform('mean')
    groups=[('all','All filing stocks',pd.Series(True,index=c.index))]
    groups += [('cap',v,c.cap.eq(v)) for v in ['Nano','Micro','Small','Mid','Large','Mega']]
    groups += [('sector',v,c.sector.eq(v)) for v in sorted(SECTORS.values())]
    groups += [('text',v,c[v]) for v in RX]
    records=[]; coverage=[]; paths=[]
    for kind,name,mask in groups:
        d=c[mask].copy(); coverage.append(dict(kind=kind,bucket=name,stock_months=len(d),securities=d.permno.nunique()))
        for eom,z in d.groupby('eom'):
            # Rank IC remains informative when portfolios cannot meet minimum breadth.
            for signal in ['sentiment','pre3','during']:
                a=z[[signal,'y','peer_y']].dropna()
                ic=a[[signal,'peer_y']].rank().corr().iloc[0,1] if len(a)>=10 else np.nan
                q=a[signal].rank(pct=True); hi=a[q.ge(.7)]; lo=a[q.le(.3)]
                valid=min(len(hi),len(lo))>=10
                records.append(dict(kind=kind,bucket=name,eom=eom,target=eom+pd.offsets.MonthEnd(1),signal=signal,ic=ic,
                    before_ic=z[['sentiment','pre3']].rank().corr().iloc[0,1] if signal=='sentiment' and z[['sentiment','pre3']].dropna().shape[0]>=10 else np.nan,
                    during_ic=z[['sentiment','during']].rank().corr().iloc[0,1] if signal=='sentiment' and z[['sentiment','during']].dropna().shape[0]>=10 else np.nan,
                    after_raw_ic=a[[signal,'y']].rank().corr().iloc[0,1] if len(a)>=10 else np.nan,
                    spread=hi.y.mean()-lo.y.mean() if valid else np.nan,
                    adjusted_spread=hi.peer_y.mean()-lo.peer_y.mean() if valid else np.nan,n=len(a)))
            for signal in ['pre3','during']:
                # Does matching the signs of content tone and momentum improve the trade?
                a=z[[signal,'sentiment','y','peer_y']].dropna()
                rank=a[signal].rank(pct=True)
                for agree in [True,False]:
                    hi=a[rank.ge(.7)&(a.sentiment.gt(0) if agree else a.sentiment.lt(0))]
                    lo=a[rank.le(.3)&(a.sentiment.lt(0) if agree else a.sentiment.gt(0))]
                    valid=min(len(hi),len(lo))>=10
                    records.append(dict(kind=kind,bucket=name,eom=eom,target=eom+pd.offsets.MonthEnd(1),signal=signal+('_tone_agrees' if agree else '_tone_opposes'),ic=np.nan,
                        spread=hi.y.mean()-lo.y.mean() if valid else np.nan,
                        adjusted_spread=hi.peer_y.mean()-lo.peer_y.mean() if valid else np.nan,n=len(hi)+len(lo)))
            for k,col in [(-3,'r-3'),(-2,'r-2'),(-1,'r-1'),(0,'ret'),(1,'r1'),(2,'r2'),(3,'r3')]:
                paths.append(dict(kind=kind,bucket=name,eom=eom,offset=k,mean=z[col].mean(),n=z[col].notna().sum()))
    m=pd.DataFrame(records)
    m=m[m.target.le(c.loc[c.y.notna(),'target'].max())]
    s=[]; folds=[]
    for keys,z in m.groupby(['kind','bucket','signal']):
        calendar=pd.date_range(c.eom.min(),m.eom.max(),freq='ME')
        z=z.set_index('eom').reindex(calendar).rename_axis('eom').reset_index()
        z['target']=z.eom+pd.offsets.MonthEnd(1)
        for period,mask in [('train',z.target.dt.year.le(2018)),('validation',z.target.dt.year.between(2019,2020)),('test',z.target.dt.year.ge(2021))]:
            for metric in ['spread','adjusted_spread','ic','before_ic','during_ic','after_raw_ic']:
                n,mean,t,p=stats(z.loc[mask,metric]); s.append(dict(zip(['kind','bucket','signal'],keys),period=period,metric=metric,months=n,mean=mean,hac_t=t,p=p))
        for year in range(2021,2027):
            masks=split_masks(z.eom,1,year); tr=stats(z.loc[masks['train'],'adjusted_spread']); va=stats(z.loc[masks['validation'],'adjusted_spread'])
            keep=tr[0]>=24 and abs(tr[2])>=2 and va[0]>=12 and np.sign(tr[1])==np.sign(va[1])
            n,mean,t,p=stats(z.loc[masks['test'],'spread'])
            folds.append(dict(zip(['kind','bucket','signal'],keys),year=year,selected=bool(keep),direction=float(np.sign(tr[1])),test_months=n,test_signed_mean=mean*np.sign(tr[1]) if keep else 0))
    s=pd.DataFrame(s); s['q']=np.nan
    for _,z in s.groupby(['period','metric']):
        p=z.p.dropna().sort_values(); n=len(p)
        if n:s.loc[p.index,'q']=np.minimum.accumulate((p.to_numpy()*n/np.arange(1,n+1))[::-1])[::-1].clip(0,1)
    m.to_csv(OUT/'content_monthly.csv',index=False); s.to_csv(OUT/'content_summary.csv',index=False)
    pd.DataFrame(folds).to_csv(OUT/'content_walk_forward.csv',index=False)
    pd.DataFrame(paths).to_csv(OUT/'content_paths.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'content_coverage.csv',index=False)
    print(s[(s.period=='test')&(s.bucket=='All filing stocks')].to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=0);main(p.parse_args().limit)
