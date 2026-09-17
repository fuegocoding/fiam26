"""Full 147-characteristic controls for nonlinear snapshot/trajectory ablations."""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import src.time_series_nonlinear as nonlinear
from src.time_series_search import ROOT,atomic_json

MODELS={
 'F29':('full_return_snapshot','ret_exc_lead1m',False),
 'F30':('full_return_trajectory','ret_exc_lead1m',True),
 'F31':('full_surprise_snapshot','target_surprise_p',False),
 'F32':('full_surprise_trajectory','target_surprise_p',True),
}


def run(base,predict_only=False):
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    output=base/'time_series_full_characteristics';output.mkdir(exist_ok=True)
    factor_path=ROOT/'data/raw/factor_char_list.csv'
    raw_path=ROOT/'data/raw/chars_final_with_names.parquet'
    fingerprint=hashlib.sha256((Path(__file__).read_text()+Path(nonlinear.__file__).read_text()
        +(base/'time_series_cost25/manifest.json').read_text()+factor_path.read_text()).encode()).hexdigest()
    manifest=output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['fingerprint']!=fingerprint:raise ValueError('Full-characteristic source/code changed; use fresh output.')
    factors=pd.read_csv(factor_path).variable.tolist()
    if len(factors)!=147:raise ValueError('Expected challenge baseline of 147 characteristics.')
    p=pd.read_parquet(base/'time_series_cost25/panel.parquet')
    missing=[c for c in factors if c not in p]
    raw=pq.read_table(raw_path,columns=['permno','eom']+missing).to_pandas()
    raw.eom=pd.to_datetime(raw.eom).astype('datetime64[ns]')
    p=p.merge(raw,on=['permno','eom'],how='left',validate='one_to_one')
    # Keep the exact compact contemporaneous Street/graph controls in both arms.
    nonlinear.SNAPSHOT=list(dict.fromkeys(factors+nonlinear.SNAPSHOT))
    nonlinear.MODELS.clear();nonlinear.MODELS.update(MODELS)
    atomic_json(manifest,{'fingerprint':fingerprint,'models':MODELS,'base_features':nonlinear.SNAPSHOT,
        'factor_count':len(factors),'source_size':raw_path.stat().st_size,'source_mtime_ns':raw_path.stat().st_mtime_ns,
        'depths':[2,4,6],'max_rounds':600,'early_stop':40,'seed':2026,
        'purpose':'Retrospective attribution against all supplied baseline characteristics plus compact Street/graph state.'})
    models={idea:nonlinear.forecasts(p,idea,output) for idea in MODELS}
    if not predict_only:nonlinear.portfolios(p,models,output)
    print(f'Full-characteristic ablations complete: {output}',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--base-output',type=Path,default=ROOT/'output')
    ap.add_argument('--predict-only',action='store_true');args=ap.parse_args();run(args.base_output,args.predict_only)

if __name__=='__main__':main()
