"""Bounded contemporary research over a past operational calendar.

Provider/registration receipts retain today's time. The study calendar drives
weekly/monthly research allocation only; it never grants historical model or
portfolio eligibility. Real and synthetic economic results remain separate.
"""
from __future__ import annotations

import calendar
import dataclasses
import datetime as dt
import json
import re
import time
from pathlib import Path

from .common import UTC, atomic_write_json, digest, file_sha256, now_utc, read_json
from .research_loop import AutonomousResearchLoop, RoleProviders, SCORES
from ..agents.budgets import PersistentCallBudget
from ..agents.provider import OpenAICompatibleProvider, LLMRequest, GenerationParams, urllib_transport
from ..papers import PaperService, PaperStore
from ..papers.watches import paper_evidence
from ..research.builtins.catalog import DEFAULT_RECIPE, RECIPES
from ..research.builtins.executors import EXECUTOR_IDS
from ..research.jobs import open_sqlite
from ..pods.charters import POD_SPECS, pod_spec
from ..pods.schema import ResearchProposal, normalise_evidence


@dataclasses.dataclass
class StudyClock:
    now: dt.datetime

    def __call__(self):
        return self.now


def study_paper_evidence(card,prepared):
    evidence=paper_evidence(card)
    selected=set(prepared.get('reviewed_claim_ids',[]))
    if selected:
        evidence['claims']=[c.text for c in card.claims if c.claim_id in selected]
    return evidence


def calendar_ticks(year):
    """Every Monday, first/final day of each month, and final day of the year."""
    if not 2021 <= year <= 2025:
        raise ValueError('choose a complete supported year in 2021..2025')
    start, end = dt.date(year,1,1),dt.date(year,12,31)
    days={start,end}
    day=start
    while day<=end:
        if day.weekday()==0 or day.day==1 or day.day==calendar.monthrange(year,day.month)[1]:
            days.add(day)
        day+=dt.timedelta(days=1)
    return [dt.datetime.combine(d,dt.time(21),UTC) for d in sorted(days)]


class CappedResearchProvider:
    """One shared durable cap across all pods/roles, reserved before every HTTP call."""
    is_mock=False

    def __init__(self,state_dir,*,max_calls=40,provider=None,min_interval_seconds=0):
        if not 1<=max_calls<=40:
            raise ValueError('this trial is authorized for at most 40 calls')
        self.provider=provider or OpenAICompatibleProvider('https://api.groq.com/openai/v1',
            'openai/gpt-oss-20b','GROQ_API_KEY',json_mode=True,transport=self._wire_transport)
        self.is_mock=bool(getattr(self.provider,'is_mock',False))
        self.model=self.provider.model
        self.name='year-trial-groq-strict-schema-v2'
        self.budget=PersistentCallBudget(Path(state_dir)/'global_llm_budget.sqlite',budget_id='year_trial_2025',
            max_calls=max_calls,max_tokens=300000,pod='all_pods',cycle='whole_year')
        self.public_papers=None
        if not 0<=min_interval_seconds<=60:
            raise ValueError('provider call interval must be bounded at 0..60 seconds')
        self.min_interval_seconds=min_interval_seconds
        self.rate_path=Path(state_dir)/'provider_rate.sqlite'
        with open_sqlite(self.rate_path) as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS rate_clock (id INTEGER PRIMARY KEY, last_at REAL NOT NULL)')

    @staticmethod
    def strict_schema(schema,params=None):
        """Close objects for Groq's documented constrained-decoding API.

        The proposal's parameter dictionary has exactly the host-reviewed keys;
        all original semantic/length/pattern validation still runs locally.
        """
        import copy
        value=copy.deepcopy(schema)
        defs=value.get('$defs',{})
        if 'ExperimentVariant' in defs:
            defs['ExperimentVariant']['properties']['params']={'type':'object',
                'properties':{k:{'enum':[v],'type':'number' if isinstance(v,(int,float)) else 'string'}
                              for k,v in (params or {}).items()},'additionalProperties':False}
        keep={'type','properties','items','anyOf','$ref','$defs','enum','additionalProperties','required','description','title'}
        def clean(node):
            if isinstance(node,list):return [clean(x) for x in node]
            if not isinstance(node,dict):return node
            out={k:({name:clean(v) for name,v in val.items()} if k in ('properties','$defs') else clean(val))
                 for k,val in node.items() if k in keep}
            if out.get('type')=='object':
                out['additionalProperties']=False
                properties=out.get('properties',{})
                if properties:
                    out['required']=list(properties)
                else:
                    # Groq rejects required:[] when the object is empty.
                    # Keep an exactly empty closed object for unreviewed plans.
                    out.pop('required',None)
            return out
        return clean(value)

    def _wire_transport(self,url,headers,body,timeout,max_bytes):
        payload=json.loads(body)
        payload['reasoning_effort']='low'
        payload['response_format']={'type':'json_schema','json_schema':{
            'name':'pod_research','strict':True,'schema':self._current_schema}}
        return urllib_transport(url,headers,json.dumps(payload).encode(),timeout,max_bytes)

    def _pace(self):
        if not self.min_interval_seconds:return
        while True:
            with open_sqlite(self.rate_path,immediate=True) as conn:
                row=conn.execute('SELECT last_at FROM rate_clock WHERE id=1').fetchone()
                wait=max(0,(row[0] if row else 0)+self.min_interval_seconds-time.time())
                if wait<=0:
                    conn.execute('INSERT INTO rate_clock VALUES (1,?) ON CONFLICT(id) DO UPDATE SET last_at=excluded.last_at',(time.time(),))
                    return
            time.sleep(min(wait,45))

    def validate_payload(self,request):
        """Allow public paper prompt views and bounded comparison metadata only.

        Competition rows, file paths, outcomes and experiment reports are never
        model inputs. This validation runs before budget reservation or HTTP.
        """
        payload=json.loads(request.user)
        if re.search(r'(/home/|/tmp/|\.parquet\b|gsk_[A-Za-z0-9]{20,}|sk-ant-)',request.user):
            raise ValueError('non-public content is forbidden in model payloads')
        if 'untrusted_evidence' in payload:
            if set(payload)-{'task','allowed_executors','max_variants','untrusted_evidence','untrusted_prior_work'}:
                raise ValueError('unexpected model input fields')
            for item in payload['untrusted_evidence']:
                if item['kind']=='paper_card':
                    if self.public_papers is None or item not in self.public_papers:
                        raise ValueError('paper evidence is not on the public-source allowlist')
                elif item['kind']=='dataset_card':
                    if not item['text'].startswith('description: '):
                        raise ValueError('unexpected dataset metadata envelope')
                    metadata=json.loads(item['text'][len('description: '):])
                    if set(metadata)!={'reviewed_comparison','baseline','limitations'}:
                        raise ValueError('only public method selector metadata is allowed')
                    params=metadata['reviewed_comparison']
                    if params is not None and set(params)!={'dataset_id','recipe_id','method_id','feature_set','ridge_lambda'}:
                        raise ValueError('row-level or unreviewed comparison data is forbidden')
                else:
                    raise ValueError('only public paper and bounded method metadata are allowed')
        elif set(payload)=={'proposal','evidence_ids'}:
            ResearchProposal.model_validate(payload['proposal'])
        else:
            raise ValueError('unknown outbound research payload')

    def complete(self,request):
        self.validate_payload(request)
        payload=json.loads(request.user)
        params=None
        for evidence in payload.get('untrusted_evidence',[]):
            if evidence['kind']=='dataset_card':
                params=json.loads(evidence['text'][len('description: '):])['reviewed_comparison']
        self._current_schema=self.strict_schema(request.response_schema or {'type':'object'},params)
        self._pace()
        system=request.system
        if request.response_schema:
            system+='\nReturn exactly one JSON object matching this schema: '+json.dumps(request.response_schema,sort_keys=True)
        # No tool execution, gold labels or protected outcomes are added.
        request=dataclasses.replace(request,system=system)
        reserve=self.budget.reserve((len(request.system)+len(request.user))//4+request.params.max_tokens)
        try:
            response=self.provider.complete(request)
        except BaseException:
            self.budget.settle(reserve,None)
            raise
        actual=None if response.prompt_tokens is None or response.completion_tokens is None else response.prompt_tokens+response.completion_tokens
        self.budget.settle(reserve,actual)
        return dataclasses.replace(response,provider=self.name)


def profile_daily_year(csv_path,year):
    """Inspect actual supplied market availability; no strategy returns are invented."""
    import pandas as pd
    import numpy as np
    counts={m:{'rows':0,'names':set(),'dates':set(),'missing_returns':0,'invalid_returns':0} for m in range(1,13)}
    for chunk in pd.read_csv(csv_path,usecols=['permno','date','daily_return'],chunksize=500000,
                             dtype={'permno':'int64','date':'string','daily_return':'float64'}):
        # Official CSV uses YYYYMMDD; ISO dates are also accepted, without
        # slicing compact dates as if they contained hyphens.
        compact=chunk['date'].str.replace('-','',regex=False)
        if not compact.str.fullmatch(r'\d{8}').all():
            raise ValueError('daily source dates must be YYYYMMDD or YYYY-MM-DD')
        parsed=pd.to_datetime(compact,format='%Y%m%d',errors='raise')
        chunk['date']=parsed.dt.strftime('%Y-%m-%d')
        part=chunk[parsed.dt.year==year]
        for month,group in part.groupby(part['date'].str.slice(5,7)):
            item=counts[int(month)];item['rows']+=len(group)
            item['names'].update(int(v) for v in group['permno'].unique())
            item['dates'].update(str(v) for v in group['date'].unique())
            item['missing_returns']+=int(group['daily_return'].isna().sum())
            observed=group['daily_return'].dropna()
            item['invalid_returns']+=int((~np.isfinite(observed)|(observed < -1)).sum())
    return {f'{year}-{m:02d}':{**item,'names':len(item['names']),'trading_days':len(item['dates']),
             'first_date':min(item['dates'],default=None),'last_date':max(item['dates'],default=None),'dates':sorted(item['dates'])}
            for m,item in counts.items()}


class YearResearchTrial:
    """One persistent state, four private pods, twelve predeclared comparison slots."""

    def __init__(self,state_dir,*,manifest_path,dataset_id,papers_path,year=2025,provider=None,max_calls=40,active_months=None,predecessor=None):
        self.state=Path(state_dir).resolve();self.state.mkdir(parents=True,exist_ok=True)
        self.year=year;self.ticks=calendar_ticks(year);self.study_clock=StudyClock(self.ticks[0])
        self.active_months=tuple(active_months or range(1,13))
        if len(set(self.active_months))!=len(self.active_months) or any(m not in range(1,13) for m in self.active_months):
            raise ValueError('active_months must be distinct calendar months')
        profile_path=self.state/'study_profile.json'
        started=read_json(profile_path)['research_started_at'] if profile_path.exists() else now_utc().isoformat()
        self.started=dt.datetime.fromisoformat(started)
        self.provider=provider or CappedResearchProvider(self.state,max_calls=max_calls,min_interval_seconds=45)
        self.papers=PaperService(PaperStore(papers_path),clock=now_utc)
        self.manifest_path=Path(manifest_path).resolve();self.dataset_id=dataset_id
        self.loop=AutonomousResearchLoop(self.state/'research',providers=RoleProviders(self.provider,self.provider,self.provider),
            manifest_path=self.manifest_path,clock=lambda:self.started,scheduler_clock=self.study_clock,
            run_id=f'year_{year}_contemporary',papers=self.papers,max_role_calls=4,max_role_tokens=60000)
        profile={'year':year,'research_started_at':started,'manifest_sha256':file_sha256(self.manifest_path),
            'dataset_id':dataset_id,'mode':'contemporary_research_over_past_calendar',
            'model':self.provider.model,'max_calls':max_calls,'slots':12,
            'code_hash':self.loop.profile['code'],'historical_model_eligibility':False}
        profile['active_months']=list(self.active_months)
        profile['predecessor']=predecessor
        if profile_path.exists() and read_json(profile_path)!=profile:
            raise ValueError('year study profile/code changed; preserve this run and use a new study state')
        atomic_write_json(profile_path,profile)
        self.profile=profile

    def _review(self,pod,card,slot,prepared):
        recipe=RECIPES[DEFAULT_RECIPE[pod.charter.value]]
        # A tiny registered grid, selected before any year outcome is read.
        features='extended' if slot==1 else 'core'
        ridge=.25 if slot==2 else .05
        if prepared.get('host_applicability','').startswith('unresolved'):
            return None
        claim_ids=prepared.get('reviewed_claim_ids') or [c.claim_id for c in card.claims if c.kind in ('economic_claim','mechanism')]
        kinds={c.kind for c in card.claims if c.claim_id in claim_ids}
        if not {'economic_claim','mechanism'}<=kinds:
            return None
        review={'namespace':pod.namespace,'paper_id':card.paper_id,'version_key':card.version_key,
            'label':'MECHANISM_TRANSFER','claim_ids':claim_ids[:20],
            'deviations':['Uses monthly operator panel proxies rather than the paper original data and timing.',
                          'Only a related mechanism comparison; this is not a replication of the paper results.'],
            'comparison_plan':'Compare the preregistered core or extended panel recipe with its declared baseline on design only.',
            'baseline':recipe.baseline_description,'recipe_id':recipe.recipe_id,'recipe_digest':recipe.digest(),
            'dataset_id':self.dataset_id,'method_id':'ridge','feature_set':features,'ridge_lambda':ridge,
            'economic_effect':recipe.summary,'declared_sign':'positive',
            'falsification':['Reject if out-of-training belief improvement or measured net bridge fails the registered criteria.'],
            'selected_by':'implementation','reviewed_by':'bounded_year_trial_host','reviewed_at':self.started.isoformat()}
        self.loop.review_method(pod.namespace,review)
        return review

    def run(self,prepared_papers,*,daily_csv=None,source_discovery=None):
        prepared={r['pod']:r for r in read_json(prepared_papers)}
        if isinstance(self.provider,CappedResearchProvider):
            allowed=[]
            for item in prepared.values():
                if item.get('selected'):
                    card=self.papers.get(item['selected']['paper_id'],as_of=self.ticks[0])
                    if card.source!='arxiv' or not card.urls.get('pdf','').startswith('https://arxiv.org/'):
                        raise ValueError('live trial paper must have a version-pinned public arXiv source')
                    allowed += [v.prompt_view() for v in normalise_evidence([study_paper_evidence(card,item)],'paper_card')]
            self.provider.public_papers=allowed
        pods=[pod_spec(p) for p in ('fundamentals','events','valuation','dynamics')]
        events=[]
        for at in self.ticks:
            self.study_clock.now=at
            name=at.strftime('%Y%m%d');path=self.state/'ticks'/f'{name}.json'
            if path.exists():
                events.append(read_json(path));continue
            submitted=[]
            if at.day==1 and at.month in self.active_months:
                index=at.month-1;pod=pods[index%4];slot=index//4
                item=prepared.get(pod.pod_id,{})
                selected=item.get('selected')
                if selected:
                    try:
                        # Search receipts are public now, but only version-pinned
                        # paper text available at this study tick is passed along.
                        card=self.papers.get(selected['paper_id'],as_of=at)
                        rv=self._review(pod,card,slot,item)
                        evidence=study_paper_evidence(card,item)
                        params=None if rv is None else {k:rv[k] for k in ('dataset_id','recipe_id','method_id','feature_set','ridge_lambda')}
                        dataset={'dataset_id':self.dataset_id,'title':'Operator-reviewed panel comparison',
                            'available_at':'2019-01-01T00:00:00+00:00',
                            'description':json.dumps({'reviewed_comparison':params,'baseline':RECIPES[DEFAULT_RECIPE[pod.charter.value]].baseline_description,
                                'limitations':['Monthly proxy panel does not contain all variables in the paper.','No target or source approval is conferred by literature.']})}
                        question=('Evaluate the related paper and this separately reviewed monthly-panel comparison. '
                            'Use one design variant with only the exact reviewed_comparison scalar selector if appropriate; '
                            'otherwise explain missing evidence without inventing methods. This is mechanism transfer, not replication. '
                            'Do not access the 2025 outcomes. Comparison slot '+str(slot+1)+'.')
                        trigger=self.loop.add_evidence(pod.namespace,subject=f'year_{self.year}_slot_{index}',question=question,
                            paper_cards=[evidence],dataset_cards=[dataset],kind='paper',line_id=f'year_slot_{index}')
                        submitted.append({'pod':pod.pod_id,'paper_id':card.paper_id,'paper_title':card.title,
                            'version':card.version_key,'reviewed_comparison':params,'trigger_id':trigger['trigger_id']})
                    except Exception as exc:
                        submitted.append({'pod':pod.pod_id,'status':'paper_review_blocked','reason':type(exc).__name__})
                else:
                    submitted.append({'pod':pod.pod_id,'status':'no_eligible_fulltext_paper'})
            # Monthly matured diagnostics are actual research status, not an
            # invented IC or trade return. Without cards these are review items.
            if at.day==calendar.monthrange(at.year,at.month)[1]:
                cohort=f'{at.year-1}-12' if at.month==1 else f'{at.year}-{at.month-1:02d}'
                for pod in pods:
                    self.loop.scheduler.add_trigger(pod.namespace,'coverage',f'year_coverage_{at:%Y%m}',
                        'Review measured source coverage and outstanding research gates before the next comparison.',SCORES,
                        cohort_month=cohort,line_id=f'coverage_{at:%Y%m}')
            result=self.loop.tick(max_jobs=8,max_iterations=3)
            event={'study_at':at.isoformat(),'executed_at':now_utc().isoformat(),'submitted':submitted,
                'jobs_this_tick':result['jobs_this_tick'],'worker_outcomes':result['worker_outcomes'],
                'candidate_outcomes':result['outcomes'],'publications':len(result['publications'])}
            atomic_write_json(path,event);events.append(event)
        return self.report(events,prepared,daily_csv,source_discovery)

    def report(self,events,prepared,daily_csv,source_discovery):
        designs=[]
        for adaptation in self.loop._events('adaptation'):
            r=self.loop.builtin.report(adaptation['registration_id'],adaptation['variant'])
            if r:
                designs.append({'namespace':adaptation['namespace'],'adaptation_id':adaptation['adaptation_id'],
                    'recipe':adaptation['params']['recipe_id'],'params':adaptation['params'],
                    'status':r.get('status'),'metrics':r.get('metrics'),'report_hash':r.get('report_hash')})
        routes=self.loop._events('route')
        proposals=[]
        for pod in POD_SPECS.values():
            for entry in self.loop.memory(pod.namespace).list(kind='draft_proposal'):
                proposals.append({'namespace':pod.namespace,'entry_id':entry.entry_id,'proposal':entry.content['proposal']})
        jobs=self.loop.queue.list(limit=1000)
        budget=self.provider.budget.snapshot() if hasattr(self.provider,'budget') else {'provider':'injected_test'}
        summary={'schema_version':'year_research_trial.v1','year':self.year,'mode':self.profile['mode'],
            'historical_model_eligibility':False,'research_started_at':self.profile['research_started_at'],
            'calendar_ticks':len(events),'weekly_cycles':{},'monthly_cycles':{},
            'submitted_comparisons':sum(len(e['submitted']) for e in events),'papers_retrieved':sum(bool(x.get('selected')) for x in prepared.values()),
            'planned_slots':12,'active_months':list(self.active_months),'predecessor':self.profile['predecessor'],
            'model':self.provider.model,'model_budget':budget,'jobs':{'total':len(jobs),'succeeded':sum(j.state=='succeeded' for j in jobs),'failed':sum(j.state=='failed' for j in jobs)},
            'routes':routes,'research_proposals':proposals,'critiques':self.loop._events('critique'),
            'papers':[{'pod':k,'paper_id':v.get('selected',{}).get('paper_id') if v.get('selected') else None,
                'title':v.get('selected',{}).get('title') if v.get('selected') else None,
                'host_applicability':v.get('host_applicability')} for k,v in prepared.items()],
            'measured_designs':designs,'outcomes':self.loop._events('outcome'),
            'source_discovery':read_json(source_discovery) if source_discovery else [],
            'admitted_production_strategies':0,'orders':0,'trading_status':'BLOCKED_NO_PRODUCTION_ADMISSION',
            'strategy_return':None,'investment_validation':'UNPROVEN',
            'limitations':['Contemporary Groq used with user authorization; exact Qwen roster not used.',
                'Twelve bounded reviewed comparisons; unsupported new methods remain review work.',
                '2025 market outcomes were not fed to proposal models or used to relax admission criteria.',
                'No valid historical/production admission means no equity strategy return is reported.']}
        for pod in POD_SPECS.values():
            cycles=self.loop.scheduler.status(pod.namespace)['cycles']
            summary['weekly_cycles'][pod.pod_id]=len([c for c in cycles if c.startswith('w:')])
            summary['monthly_cycles'][pod.pod_id]=len([c for c in cycles if c.startswith('m:')])
        if daily_csv:
            profile_path=self.state/'market_2025_profile.json'
            if not profile_path.exists():atomic_write_json(profile_path,profile_daily_year(daily_csv,self.year))
            summary['real_market_availability']=read_json(profile_path)
        atomic_write_json(self.state/'summary.json',summary)
        return summary
