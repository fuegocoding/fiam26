"""Fixed nonlinear ablations of the existing return, surprise, and risk hypotheses.

Three tree depths are selected by dated validation MSE, never test performance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from src.graph import expectations as ibes
from src.graph.walk_forward import split_masks
from src.time_series_search import (ROOT,SNAPSHOT,UNIVERSES,Market,atomic_json,
    rank_month,temporal_features,simulate,select_validation,account_selected_book,metrics)

MODELS={
 'N29':('return_snapshot','ret_exc_lead1m',False),
 'N30':('return_trajectory','ret_exc_lead1m',True),
 'N31':('surprise_snapshot','target_surprise_p',False),
 'N32':('surprise_trajectory','target_surprise_p',True),
 'N39S':('risk_snapshot','next_log_risk',False),
 'N39T':('risk_trajectory','next_log_risk',True),
}


def forecasts(p,idea,output):
    name,target,history=MODELS[idea]
    extra=['fgap_p','fgap_disp','days_since_ann','expected_report_next_month',
           'rev3_next_p','exp_growth_p','sue_vol8_p'] if target=='next_log_risk' else []
    features=list(dict.fromkeys(SNAPSHOT+extra+(temporal_features(p) if history else [])))
    # Same ranks and missing indicators as the linear ablation; XGBoost can split ranks nonlinearly.
    x=(p[features].groupby(p.eom).rank(pct=True)-.5).fillna(0).to_numpy(np.float32)
    x=np.column_stack([x,p[features].isna().to_numpy(np.float32)])
    y=p[target].to_numpy(float)
    frames,audits=[],[]
    for year in range(2021,2027):
        path=output/f'{idea}_{year}_predictions.parquet'
        audit_path=output/f'{idea}_{year}_audit.json'
        if path.exists() and audit_path.exists():
            frames.append(pd.read_parquet(path)); audits.append(json.loads(audit_path.read_text()));continue
        label_dates=p.target_ann if target=='target_surprise_p' else p.eom+pd.offsets.MonthEnd(1)
        masks=split_masks(p.eom,1,year,label_dates=label_dates)
        train=masks['train'] & np.isfinite(y); val=masks['validation'] & np.isfinite(y)
        train_y=y[train];val_y=y[val]
        if target=='target_surprise_p':
            train_y=rank_month(p.loc[train],train_y).to_numpy();val_y=rank_month(p.loc[val],val_y).to_numpy()
        dtrain=xgb.DMatrix(x[train],label=train_y,nthread=4)
        dval=xgb.DMatrix(x[val],label=val_y,nthread=4)
        best=None; candidates=[]
        for depth in [2,4,6]:
            params={'objective':'reg:squarederror','eval_metric':'rmse','tree_method':'hist',
                    'max_depth':depth,'eta':.05,'min_child_weight':100,'subsample':.8,
                    'colsample_bytree':.8,'reg_lambda':10.,'seed':2026,'nthread':4}
            model=xgb.train(params,dtrain,num_boost_round=600,evals=[(dval,'validation')],
                            early_stopping_rounds=40,verbose_eval=False)
            rounds=int(model.best_iteration)+1
            pred=model.predict(dval,iteration_range=(0,rounds))
            loss=float(np.mean((val_y-pred)**2))
            candidates.append({'depth':depth,'rounds':rounds,'validation_mse':loss})
            if best is None or loss < best[0]: best=loss,depth,rounds,model
            logging.info('%s fold %s depth %s rounds %s validation MSE %.6g',idea,year,depth,rounds,loss)
        windows=split_masks(p.eom,1,year)
        keep=windows['validation'] | windows['test']
        f=p.loc[keep,['permno','eom']].copy()
        f['prediction']=best[3].predict(xgb.DMatrix(x[keep],nthread=4),iteration_range=(0,best[2]))
        f['fold']=year;f['window']=np.where(windows['test'][keep],'test','validation')
        test=windows['test'] & (p.eom+pd.offsets.MonthEnd(1)).between('2021-01-31','2026-08-31') & np.isfinite(y)
        testpred=best[3].predict(xgb.DMatrix(x[test],nthread=4),iteration_range=(0,best[2]))
        test_y=y[test]
        if target=='target_surprise_p': test_y=rank_month(p.loc[test],test_y).to_numpy()
        audit={'fold':year,'target':target,'features':features,'candidates':candidates,
            'depth':best[1],'rounds':best[2],'train_rows':int(train.sum()),'validation_rows':int(val.sum()),
            'test_rows':int(test.sum()),'test_mse':float(np.mean((test_y-testpred)**2)),
            'test_rank_ic':ibes.monthly_rank_ic(p.loc[test,'eom'],test_y,testpred)}
        best[3].save_model(output/f'{idea}_{year}_model.json')
        f.to_parquet(path,index=False);atomic_json(audit_path,audit)
        frames.append(f);audits.append(audit)
    result=pd.concat(frames,ignore_index=True)
    result.to_parquet(output/f'model_{idea}.parquet',index=False)
    atomic_json(output/f'model_{idea}_audit.json',audits)
    return result


def portfolios(p,models,destination):
    m=Market(p);m.wide_me=m.wide(p.me)
    original=m.eligible.copy();oldvol=m.vol.copy();rows=[]
    keys=pd.MultiIndex.from_frame(p[['permno','eom']])
    for idea,model in models.items():
        for cost in [10,25,50]:
            for universe,sizes in UNIVERSES.items():
                m.eligible=original & m.wide(p.size_grp.isin(sizes).astype(float),fill=0).astype(bool)
                for mode in ['raw','factor']:
                    token=f'{idea}_cost{cost}_{universe}_{mode}'
                    result_file=destination/f'{token}.json'
                    if result_file.exists():
                        summary=json.loads(result_file.read_text())['summary']
                    else:
                        chosen_books,chosen_weights,choices=[],[],[]
                        for year in range(2021,2027):
                            pred=model[model.fold.eq(year)].set_index(['permno','eom']).prediction.reindex(keys).to_numpy()
                            if idea.startswith('N39'):
                                m.vol=m.wide(np.exp(np.clip(pred,-12,4)/2));score=p.ret_12_1.to_numpy();simulator='33'
                            else: score=pred;simulator=idea
                            books={h:simulate(m,m.wide(score),h,mode,simulator,cost) for h in [1,3,6]}
                            hold,val=select_validation(books,year)
                            template=next(iter(books.values()))
                            selected=(books[hold] if hold is not None else template).loc[template.date.dt.year.eq(year)].copy()
                            weights=books[hold].attrs['weights'][selected.index] if hold is not None else np.zeros((len(selected),len(m.ids)),np.float32)
                            selected.attrs={}
                            if hold is None:
                                for col in ['gross_excess','net_excess','cost','turnover','n_long','n_short','gross_exposure',
                                    'net_exposure','missing_held_returns','long_pnl','short_pnl','max_position','exante_beta']:selected[col]=0
                                selected['short_me']=np.nan;selected['turnover_uncertain']=False
                            chosen_books.append(selected);chosen_weights.append(weights)
                            choices.append({'year':year,'chosen_hold':hold,'validation':val})
                        m.vol=oldvol
                        book=pd.concat(chosen_books,ignore_index=True);weights=np.concatenate(chosen_weights)
                        book=account_selected_book(m,book,weights,cost)
                        book.to_csv(destination/f'{token}_book.csv',index=False)
                        mi,si=np.nonzero(weights)
                        pd.DataFrame({'date':book.date.to_numpy()[mi],'permno':m.ids[si],'weight':weights[mi,si]}).to_parquet(destination/f'{token}_holdings.parquet',index=False)
                        summary=metrics(book)
                        atomic_json(result_file,{'experiment':idea,'name':MODELS[idea][0],'cost_bp':cost,
                            'universe':universe,'mode':mode,'summary':summary,'choices':choices})
                    rows.append({'experiment':idea,'name':MODELS[idea][0],'cost_bp':cost,'universe':universe,'mode':mode,
                        'timesfm_cache_diagnostic':False,**summary})
                    pd.DataFrame(rows).sort_values('information_ratio',ascending=False).to_csv(destination/'leaderboard.csv',index=False)
                    logging.info('Nonlinear portfolio %s complete',token)


def run(base,predict_only=False):
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    output=base/'time_series_nonlinear';output.mkdir(exist_ok=True)
    fingerprint=hashlib.sha256((Path(__file__).read_text()+(base/'time_series_cost25/manifest.json').read_text()).encode()).hexdigest()
    manifest=output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['fingerprint']!=fingerprint:raise ValueError('Nonlinear code/source changed; use fresh output.')
    atomic_json(manifest,{'fingerprint':fingerprint,'models':MODELS,'depths':[2,4,6],
        'max_rounds':600,'early_stop':40,'seed':2026,'xgboost_version':xgb.__version__,
        'purpose':'Retrospective nonlinear ablations of frozen hypotheses; no new untouched holdout.'})
    p=pd.read_parquet(base/'time_series_cost25/panel.parquet')
    p['next_log_risk']=np.log(p.ret_exc_lead1m**2+.0001)
    models={idea:forecasts(p,idea,output) for idea in MODELS}
    if not predict_only:portfolios(p,models,output)
    print(f'Nonlinear ablations complete: {output}',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-output',type=Path,default=ROOT/'output');ap.add_argument('--predict-only',action='store_true')
    args=ap.parse_args();run(args.base_output,args.predict_only)

if __name__=='__main__':main()
