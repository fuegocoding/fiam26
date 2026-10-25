#!/usr/bin/env python3
"""Summarize preserved one-year trial runs without inference or source-row export."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pod_platform.agents.budgets import PersistentCallBudget


def events(path,kind):
    if not path.exists():return []
    with sqlite3.connect(path) as conn:
        return [{'namespace':ns,**json.loads(body)} for ns,body in conn.execute('SELECT namespace,payload FROM runtime_events WHERE kind=? ORDER BY seq',(kind,))]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.state_root
    budget=PersistentCallBudget(root/'live/global_llm_budget.sqlite',budget_id='year_trial_2025',pod='all_pods',cycle='whole_year',max_calls=40,max_tokens=300000)
    runs=[];drafts=[];critiques=[];outcomes=[]
    for run in sorted(root.glob('live*')):
        db=run/'research/runtime.sqlite';queue=run/'research/queue.sqlite'
        if not db.exists() or not queue.exists():continue
        with sqlite3.connect(queue) as conn:
            jobs=[{'namespace':ns,'kind':kind,'state':state,'error_type':json.loads(error).get('type') if error else None}
                  for ns,kind,state,error in conn.execute('SELECT namespace,kind,state,error_json FROM jobs')]
        tick_files=sorted((run/'ticks').glob('*.json'))
        runs.append({'run':run.name,'completed_ticks':len(tick_files),'jobs':jobs,
                     'complete_summary':(run/'summary.json').exists()})
        for memory in (run/'research/memory').glob('*.sqlite'):
            # Read schema-independent via the existing namespace store.
            from pod_platform.research.memory import PodMemory
            store=PodMemory(run/'research/memory',memory.stem)
            for entry in store.list(kind='draft_proposal'):
                proposal=entry.content['proposal']
                drafts.append({'run':run.name,'namespace':memory.stem,'title':proposal['title'],
                    'mechanism':proposal['mechanism'],'limitations':proposal['limitations'],
                    'planned_variants':proposal['experiment_plan']['variants']})
        critiques.extend({'run':run.name,**c} for c in events(db,'critique'))
        outcomes.extend({'run':run.name,**c} for c in events(db,'outcome'))
    recovery=root/'events_recovery/summary.json'
    if recovery.exists():
        extra=json.loads(recovery.read_text())
        drafts.append({'run':'events_recovery','namespace':'pod_events','title':extra['proposal']['title'],
            'mechanism':extra['proposal']['mechanism'],'limitations':extra['proposal']['limitations'],
            'planned_variants':extra['proposal']['experiment_plan']['variants']})
        critiques.append({'run':'events_recovery','namespace':'pod_events','value':extra['critique']})
    controls_path=root/'design_controls/summary.json'
    controls=json.loads(controls_path.read_text()) if controls_path.exists() else None
    quarterly=json.loads((root/'quarterly_discovery.json').read_text())
    paper_rows=json.loads((root/'prepared-papers.json').read_text())
    fixture=json.loads((root/'fixture_v2/output/complete_summary.json').read_text())
    market=json.loads((root/'market_2025_profile.json').read_text())
    result={'schema_version':'year_trial_report.v1','year':2025,
        'interpretation':'Contemporary research plus a separate synthetic twelve-period systems acceptance test; no valid historical LLM strategy return.',
        'live_runs':runs,'global_model_budget':budget.snapshot(),'research_proposals':drafts,'critiques':critiques,'candidate_outcomes':outcomes,
        'papers':[{'pod':r['pod'],'title':r['selected']['title'],'paper_id':r['selected']['paper_id'],
                  'version':r['selected']['version_key'],'url':r['selected']['urls'].get('abstract'),
                  'host_applicability':r.get('host_applicability')} for r in paper_rows if r.get('selected')],
        'quarterly_searches':len(quarterly),'successful_quarterly_searches':sum(x['status']=='ok' for x in quarterly),
        'new_paper_versions_discovered':sum(x.get('new_versions',0) for x in quarterly),
        'alternative_data_candidates':sum(len(x['result']['data']) for x in json.loads((root/'source-discovery.json').read_text())),
        'actual_2025_market':{'rows':sum(x['rows'] for x in market.values()),'trading_days':sum(x['trading_days'] for x in market.values()),
                             'missing_returns_retained':sum(x['missing_returns'] for x in market.values())},
        'synthetic_acceptance':{'periods':fixture['book']['periods'],'publication_batches':fixture['publication_count'],
                                'feedback_items':fixture['feedback_items'],'fixture_only':True},
        'operator_design_controls':controls,'real_strategy_return':None,'real_orders':0,'admitted_production_strategies':0,
        'limitations':['All failed API attempts are retained and charged within the single40-call account.',
            'Public paper excerpts and bounded method selectors only were sent to Groq; price data/outcomes stayed local.',
            'Quarterly metadata discoveries need full-text/method review and were not silently converted to trading strategies.',
            'Synthetic performance is not investment evidence. Real deployment requires valid source/model/candidate admission.']}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'calls_used':budget.calls_used,'drafts':len(drafts),'critiques':len(critiques),'outcomes':len(outcomes),'report':str(args.output)}))


if __name__=='__main__':main()
