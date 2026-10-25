#!/usr/bin/env python3
"""One-year bounded contemporary research trial. Never backdates model/admission receipts."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pod_platform.application.year_trial import YearResearchTrial,CappedResearchProvider


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state-dir',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--dataset-id',required=True)
    p.add_argument('--papers-db',type=Path,required=True)
    p.add_argument('--prepared-papers',type=Path,required=True)
    p.add_argument('--source-discovery',type=Path)
    p.add_argument('--daily-csv',type=Path)
    p.add_argument('--year',type=int,default=2025)
    p.add_argument('--max-calls',type=int,default=40)
    p.add_argument('--online',action='store_true')
    p.add_argument('--budget-dir',type=Path,help='Reuse the original global budget on an explicitly amended run')
    p.add_argument('--active-months',type=int,nargs='+')
    p.add_argument('--predecessor')
    p.add_argument('--min-call-interval',type=float,default=45)
    a=p.parse_args()
    if not a.online:p.error('actual Groq inference requires --online')
    trial=YearResearchTrial(a.state_dir,manifest_path=a.manifest,dataset_id=a.dataset_id,papers_path=a.papers_db,
        year=a.year,max_calls=a.max_calls,active_months=a.active_months,predecessor=a.predecessor,
        provider=CappedResearchProvider(a.budget_dir or a.state_dir,max_calls=a.max_calls,min_interval_seconds=a.min_call_interval))
    result=trial.run(a.prepared_papers,daily_csv=a.daily_csv,source_discovery=a.source_discovery)
    print(json.dumps({k:result[k] for k in ('year','mode','calendar_ticks','submitted_comparisons','papers_retrieved','model_budget','jobs','trading_status','strategy_return')},default=str))
    return 0


if __name__=='__main__':raise SystemExit(main())
