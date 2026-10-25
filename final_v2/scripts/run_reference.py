#!/usr/bin/env python3
"""Offline reference workflow; fixture evidence never authorizes an investment book."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pod_platform.application.reference import ReferenceWorkflow

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=['demo','research','replay'])
    parser.add_argument('--state-dir',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--run-id',default='reference_v1')
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--dataset-id')
    parser.add_argument('--research-asof')
    parser.add_argument('--data',action='append',default=[],type=Path)
    parser.add_argument('--fixture',type=Path)
    parser.add_argument('--label',choices=['test','formation'],default='test')
    args=parser.parse_args(argv)
    try:
        workflow=ReferenceWorkflow(args.state_dir,args.output_dir,run_id=args.run_id)
        if args.mode=='demo':result=workflow.run_demo()
        elif args.mode=='research':
            if not all((args.manifest,args.dataset_id,args.research_asof)):raise ValueError('research requires --manifest --dataset-id --research-asof')
            result=workflow.run_research(args.manifest,args.dataset_id,args.research_asof,args.data)
        else:
            if not args.fixture:raise ValueError('replay requires --fixture')
            result=workflow.run_replay(args.fixture,args.label)
        print(json.dumps(result,sort_keys=True));return 0
    except Exception as exc:
        print(json.dumps({'status':'FAILED_CLOSED','error':str(exc)[:1000]}),file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
