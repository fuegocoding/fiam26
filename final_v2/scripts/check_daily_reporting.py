#!/usr/bin/env python3
"""Validate operator supplied daily CSV coverage; optionally reconcile explicit inputs."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pod_platform.reporting.supplied_daily import read_supplied_daily,inputs_from_json,write_json
from pod_platform.reporting.daily import reconcile_daily
def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--file',required=True); p.add_argument('--security-id',action='append',required=True); p.add_argument('--start',required=True); p.add_argument('--end',required=True); p.add_argument('--out',required=True); p.add_argument('--inputs-json'); p.add_argument('--descriptive',action='store_true'); p.add_argument('--overwrite',action='store_true'); a=p.parse_args()
 source=read_supplied_daily(a.file,a.security_id,a.start,a.end,descriptive=a.descriptive); result=source.metadata()
 if a.inputs_json:
  inputs,policy=inputs_from_json(json.loads(Path(a.inputs_json).read_text()),source); result.update(status='daily_reconciled',performance_metrics=reconcile_daily(inputs,policy).as_dict())
 write_json(a.out,result,overwrite=a.overwrite); print(json.dumps({k:result[k] for k in ('status','observations','trading_days','missing_return_count','sha256')},sort_keys=True)); return 0
if __name__=='__main__':
 try: raise SystemExit(main())
 except Exception as exc: print(f'daily reporting check failed: {exc}',file=sys.stderr); raise SystemExit(2)
