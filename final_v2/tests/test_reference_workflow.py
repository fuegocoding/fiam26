"""End-to-end software evidence; synthetic fixtures make no investment claims."""
import datetime as dt
import importlib
import json
from pathlib import Path
import pytest
from pod_platform.application.reference import ReferenceWorkflow
from pod_platform.application.common import read_json, atomic_write_json, UTC
from pod_platform.application.judge import FactRef,FactResolver,verify_commitment,verify_judge_package,score_prospective
from pod_platform.application.decision_store import DecisionStore
from pod_platform.application.publication import PublicationService,certificate_scope
from pod_platform.application.namespaces import RoleContext,RoleGateway,WorkspaceStore
from pod_platform.application.methods import default_registry
from pod_platform.contracts.errors import PodPlatformError
from pod_platform.research.promotion import AdmissionCertificate
from pod_platform.services.app import build_server

@pytest.fixture(scope='module')
def demo(tmp_path_factory):
    root=tmp_path_factory.mktemp('reference')
    workflow=ReferenceWorkflow(root/'state',root/'out')
    summary=workflow.run_demo()
    return root,summary

def test_full_demo_real_artifacts_rejected_never_trades(demo):
    root,summary=demo;out=root/'out'
    assert summary['software_status']=='PASS'
    assert summary['investment_validation']=='UNPROVEN'
    assert summary['prospective_status']=='INCOMPLETE'
    assert not summary['llm_called'] and not summary['certificates_authorize_competition_book']
    candidates=read_json(out/'candidates.json')
    assert len(candidates)==4
    assert len({c['charter'] for c in candidates})==4
    rejected=[c for c in candidates if c['outcome']=='REJECTED']
    assert rejected and all(c['metrics']['check_mean_ic']<0 for c in rejected)
    forecasts=read_json(out/'fixture/forecasts.json')
    assert not {c['candidate_id'] for c in rejected}&{f['strategy_id'] for f in forecasts}
    assert all(c['criteria']['coverage_met'] for c in candidates)
    assert all(c['trial_counts']['check_window_access']>=1 for c in candidates)
    assert read_json(out/'book/twin_exhibit.json')['agent_kind']=='DETERMINISTIC_STAND_IN_NOT_LLM'
    assert read_json(out/'book/twin_exhibit.json')['failures']
    assert verify_judge_package(out)['facts_verified']>=8
    assert verify_commitment(out)['artifacts_verified']>20

def test_restart_does_not_repeat_trials(demo):
    root,summary=demo
    before=(root/'out/candidates.json').read_bytes()
    again=ReferenceWorkflow(root/'state',root/'out').run_demo()
    assert again==summary
    assert (root/'out/candidates.json').read_bytes()==before
    assert {r['stage'] for r in ReferenceWorkflow(root/'state',root/'out').runner.records() if r['status']=='resumed'}=={'challenge','publication','book','acceptance'}

def test_tampered_artifact_or_commitment_fails_closed(tmp_path,demo):
    import shutil
    root,_=demo;shutil.copytree(root/'out',tmp_path/'out')
    out=tmp_path/'out'
    payload=read_json(out/'book/performance_report.json');payload['injected']=True
    atomic_write_json(out/'book/performance_report.json',payload)
    with pytest.raises(ValueError,match='changed'):verify_commitment(out)
    with pytest.raises(ValueError,match='hash mismatch'):verify_judge_package(out)
    resolver=FactResolver(out)
    with pytest.raises(ValueError,match='outside'):resolver.resolve(FactRef('../private','x'))

def test_resume_artifact_hash_tamper(tmp_path):
    workflow=ReferenceWorkflow(tmp_path/'state',tmp_path/'out')
    workflow.run_demo()
    (tmp_path/'out/candidates.json').write_text('[]')
    with pytest.raises(PodPlatformError):workflow.run_demo()

def test_test_certificate_cannot_enter_formation_replay(tmp_path,demo):
    root,_=demo
    cert=AdmissionCertificate.model_validate(read_json(root/'out/fixture/certificates.json')[0])
    assert certificate_scope(cert)=='fixture'
    with pytest.raises(PodPlatformError):
        ReferenceWorkflow(tmp_path/'state',tmp_path/'out').run_replay(root/'out/fixture','formation')
    assert not (tmp_path/'out/book/holdings_export.csv').exists()

def test_replay_test_mode_is_resumable(tmp_path,demo):
    root,_=demo
    flow=ReferenceWorkflow(tmp_path/'state',tmp_path/'out')
    result=flow.run_replay(root/'out/fixture','test')
    assert result['mode']=='replay'
    assert result['judge_status']=='INCOMPLETE' # no invented research rejection in operator-only replay
    assert flow.run_replay(root/'out/fixture','test')==result

def test_stale_publications_are_excluded(demo):
    root,_=demo
    service=PublicationService(root/'state/publications.sqlite')
    context=RoleContext('allocation','allocation','read',dt.datetime(2026,8,31,tzinfo=UTC))
    result=service.active_for_decision(context,context.research_asof)
    assert not result.active and result.stale_counts

def test_dossier_arbitrary_date_actual_holdings(demo):
    root,_=demo;store=DecisionStore(root/'state/decision_store.sqlite')
    rows=store.positions_on(dt.date(2015,3,1),'reference_v1')
    assert rows
    evidence=store.explain(rows[0]['security_id'],dt.date(2015,3,1),'reference_v1')
    assert evidence['held_on_date'] and evidence['governing_formation_date']=='2015-02-19'
    assert evidence['forecast_ids'] and evidence['risk_review_ids'] and evidence['order_ids']
    exited=store.explain(rows[0]['security_id'],dt.date(2016,1,1),'reference_v1')
    assert not exited['held_on_date'] and exited['last_held_period']

def call(server,name,args):
    server.initialized=True
    response=server.dispatch({'jsonrpc':'2.0','id':1,'method':'tools/call','params':{'name':name,'arguments':args}})
    return response['result']

def test_mcp_fixed_roles_bounded_args_and_date_lookup(demo):
    root,_=demo
    reporting=build_server(root/'state',role='reporting')
    assert call(reporting,'runs.list',{})['isError'] is False
    data=json.loads(call(reporting,'dossier.lookup',{'run_id':'reference_v1','security_id':'S000','on_date':'2015-03-01'})['content'][0]['text'])
    assert data['held_on_date']
    assert call(reporting,'runs.show',{'run_id':'../escape'})['isError']
    assert call(reporting,'reports.show',{'run_id':'reference_v1','path':'/etc/passwd'})['isError']
    assert call(reporting,'notes.append',{'content':{'role':'integrity'},'idempotency_key':'n1','namespace':'other'})['isError']
    execution=build_server(root/'state',role='execution')
    assert call(execution,'reports.show',{'run_id':'reference_v1'})['isError']
    assert call(execution,'publications.list',{})['isError']
    strategy=build_server(root/'state','fundamentals')
    try:
        assert call(strategy,'dossier.lookup',{'run_id':'reference_v1','security_id':'S000','on_date':'2015-03-01'})['isError']
        assert json.loads(call(strategy,'platform.status',{})['content'][0]['text'])['role']=='strategy'
    finally:strategy.close()

def test_namespace_isolation_and_methods_import(tmp_path):
    context=RoleContext('strategy','alpha','run',dt.datetime(2026,1,1,tzinfo=UTC))
    store=WorkspaceStore(tmp_path)
    store.write_json(context,'alpha','notes.json',{'note':'private'})
    with pytest.raises(PodPlatformError):store.read_json(context,'beta','notes.json')
    with pytest.raises(PodPlatformError):RoleGateway().authorize(context,'certificate.issue')
    for card in default_registry().list():
        module,name=card.reference_implementation.split(':')
        assert getattr(importlib.import_module(module),name) is not None

def test_research_invokes_campaign_without_admission(tmp_path):
    from pod_platform.research.builtins import write_fixture_dataset
    manifest=write_fixture_dataset(tmp_path/'dataset')
    from pod_platform.research.builtins.manifest import load_manifest
    dataset_id=next(iter(load_manifest(manifest).datasets))
    flow=ReferenceWorkflow(tmp_path/'state',tmp_path/'out')
    result=flow.run_research(manifest,dataset_id,'2026-08-31T00:00:00+00:00')
    assert result['mode']=='research' and result['judge_status']=='INCOMPLETE'
    decisions=read_json(tmp_path/'out/candidates.json')
    assert len(decisions)==4 and all(c['certificate_id'] is None and c['outcome']!='ADMITTED' for c in decisions)
    assert read_json(tmp_path/'out/publications.json')==[]
    assert flow.run_research(manifest,dataset_id,'2026-08-31T00:00:00+00:00')==result

def test_prospective_rejects_retrospective_dates(demo):
    root,_=demo
    assert score_prospective(root/'out')['status']=='INCOMPLETE'

def test_failure_diagnostics_queue_and_prior_retrieval_persist(tmp_path):
    from pod_platform.application.diagnostics import ResearchDiagnostics,FailureKind
    from pod_platform.research.memory import PodMemory
    diag=ResearchDiagnostics(PodMemory(tmp_path,'pod_fundamentals'))
    failures=diag.classify({'target_ic_improvement':-.1})
    assert failures[0].kind==FailureKind.TARGET_FORECAST_WRONG
    diag.record('idea',failures,attempted='registered comparison',outcome='negative measured improvement',retry_justified=False)
    item=diag.queue_item_for(failures[0],subject_id='idea',decision_deadline=dt.datetime(2026,8,31,tzinfo=UTC))
    diag.enqueue(item)
    reopened=ResearchDiagnostics(PodMemory(tmp_path,'pod_fundamentals'))
    assert reopened.prior_attempts('idea')[0]['retry_justified'] is False
    assert reopened.list_queue()[0].item_id==item.item_id
    reopened.pause_line(item.item_id,'No new discriminating evidence')
    assert reopened.list_queue()==[]
    assert reopened.classify({})[0].kind==FailureKind.UNCLASSIFIED

def test_demo_reports_and_controls_have_actual_calculations(demo):
    root,_=demo;book=root/'out/book'
    attribution=read_json(book/'attribution_report.json')
    for period in attribution['holdings']['periods']:
        assert period['factor_total']+period['idiosyncratic']==pytest.approx(period['total'])
    for period in attribution['selection_sizing']['periods']:
        assert period['selection']+period['sizing']==pytest.approx(period['actual'])
    daily=read_json(book/'daily_report.json')
    assert daily['book']=='PERMITTED_DAILY_TEST_BASELINE'
    assert daily['summary']['exact_nav']
    assert daily['summary']['daily_beta']['status']=='computed'
    assert read_json(book/'alpha_decay.json')['status']=='COMPUTED'
    controls=read_json(book/'incremental_evidence.json')
    assert len(controls['rows'])==3
    assert all(r['status']=='COMPUTED' and r['ending_nav_difference']==pytest.approx(0) for r in controls['rows'])

def test_full_paired_self_financing_nav_includes_all_costs_and_periods(demo):
    root,_=demo;twin=read_json(root/'out/book/twin_exhibit.json')
    assert twin['schema_version']=='paired_exhibit.v2' and twin['status']=='COMPLETED'
    assert twin['paired_nav_difference']==pytest.approx(0)
    assert len(twin['periods'])==4
    for arm in ('fixed','agent'):
        nav=1e6
        for row in twin['arms'][arm]['monthly']:
            nav*=1+row['net_return']
            assert nav==pytest.approx(float(row['nav_end']))
            assert row['borrow_return']>0 and row['cost_return']>0
            assert all(abs(float(v))<1e-6 for v in row['reconciliation'].values())
        assert nav==pytest.approx(twin['arms'][arm]['ending_nav'])
    acceptance=read_json(root/'out/acceptance.json')
    assert acceptance['checks']['paired_nav']['valid']

def test_scripted_agent_full_path_can_harm_and_includes_all_losses(tmp_path,demo):
    from pod_platform.application.twin_exhibit import FixtureExceptionAgent
    root,_=demo
    # The same fixture contains a failed long. Retaining it at full size has
    # different costs, borrow, drift and later rebalancing from the half-size twin.
    agent=FixtureExceptionAgent(lambda ctx:'retain',identity='retain_all_test')
    result=ReferenceWorkflow(tmp_path/'state',tmp_path/'out',exception_agent=agent).run_replay(root/'out/fixture','test')
    twin=read_json(tmp_path/'out/book/twin_exhibit.json')
    assert twin['agent_kind']=='FIXTURE_STAND_IN_NOT_LLM'
    assert not twin['llm_skill_evidence'] and not result['llm_called']
    assert len(twin['periods'])==4 and agent.calls>0
    assert abs(twin['paired_nav_difference'])>1
    assert twin['arms']['agent']['monthly'][0]['trading_fees']!=twin['arms']['fixed']['monthly'][0]['trading_fees']
    assert twin['failures'] # losing positions are never omitted

def test_crashed_controlled_book_quarantines_only_owned_output(tmp_path,monkeypatch):
    import sqlite3
    from pod_platform.orchestration.replay import ControlledReplay
    original=ControlledReplay.run
    def crash(self):
        self.writer.write_json('failure_injected.json',{'evidence':'do not erase'})
        raise SystemExit('injected crash after output creation')
    monkeypatch.setattr(ControlledReplay,'run',crash)
    flow=ReferenceWorkflow(tmp_path/'state',tmp_path/'out')
    with pytest.raises(SystemExit):flow.run_demo()
    with sqlite3.connect(tmp_path/'state/demo_registry.sqlite') as db:
        charged=db.execute('SELECT COUNT(*) FROM trial_counts').fetchone()[0]
    original_evidence=(tmp_path/'out/book/failure_injected.json').read_bytes()
    (tmp_path/'out/operator_note.txt').write_text('unowned evidence')
    monkeypatch.setattr(ControlledReplay,'run',original)
    ReferenceWorkflow(tmp_path/'state',tmp_path/'out').run_demo()
    attempts=tmp_path/'state/application/runs/reference_v1/attempts/book'
    assert read_json(attempts/'attempt_000001/terminal.json')['status']=='failed'
    assert (attempts/'attempt_000002/quarantine/book/failure_injected.json').read_bytes()==original_evidence
    assert read_json(attempts/'attempt_000002/quarantine_manifest.json')['outputs'][0]['sha256']
    assert (tmp_path/'out/operator_note.txt').read_text()=='unowned evidence'
    with sqlite3.connect(tmp_path/'state/demo_registry.sqlite') as db:
        assert db.execute('SELECT COUNT(*) FROM trial_counts').fetchone()[0]==charged

def test_hard_process_exit_recovers_without_terminal_record(tmp_path):
    import subprocess,sys,textwrap
    from pod_platform.application.stages import StageRunner
    output=tmp_path/'out';output.mkdir()
    source=textwrap.dedent(f'''
        import os
        from pathlib import Path
        from pod_platform.application.stages import StageRunner
        root=Path({str(output)!r})
        def crash():
            (root/'book').mkdir()
            (root/'book/partial.json').write_text('{{"charged":true}}')
            os._exit(73)
        StageRunner({str(tmp_path/'state')!r},'hard').run('book',crash,inputs={{'data':'fixed'}},
            source_hashes={{'code':'stable'}},artifacts_root=root,owned_outputs=('book',))
    ''')
    assert subprocess.run([sys.executable,'-c',source],cwd=Path(__file__).resolve().parents[1]).returncode==73
    runner=StageRunner(tmp_path/'state','hard')
    def retry():
        (output/'book').mkdir();atomic_write_json(output/'book/final.json',{'ok':True})
        from pod_platform.application.common import file_sha256
        return {'artifacts':{'book/final.json':file_sha256(output/'book/final.json')}}
    runner.run('book',retry,inputs={'data':'fixed'},source_hashes={'code':'stable'},artifacts_root=output,owned_outputs=('book',))
    attempts=runner.dir/'attempts/book'
    assert read_json(attempts/'attempt_000001/terminal.json')['status']=='interrupted'
    assert read_json(attempts/'attempt_000002/quarantine/book/partial.json')['charged']
    with pytest.raises(PodPlatformError):
        runner.run('book',retry,inputs={'data':'different'},source_hashes={'code':'stable'},artifacts_root=output,owned_outputs=('book',))

def test_failed_first_attempt_binds_inputs_before_checkpoint(tmp_path):
    from pod_platform.application.stages import StageRunner
    runner=StageRunner(tmp_path,'bound')
    def crash():raise RuntimeError('injected')
    with pytest.raises(RuntimeError):runner.run('stage',crash,inputs={'x':1},source_hashes={'code':'v1'})
    with pytest.raises(PodPlatformError):runner.run('stage',lambda:{},inputs={'x':2},source_hashes={'code':'v1'})
    with pytest.raises(PodPlatformError):runner.run('stage',lambda:{},inputs={'x':1},source_hashes={'code':'v2'})

def test_challenge_retry_keeps_scored_trials_and_settles_results(tmp_path,monkeypatch):
    import sqlite3
    from pod_platform.application.recovery import RecoverableRegisteredEvaluator
    original=RecoverableRegisteredEvaluator.evaluate;fault={'fired':False}
    def crash_after_scoring(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        if not fault['fired']:
            fault['fired']=True
            raise SystemExit('crash after first scored result was persisted')
        return result
    monkeypatch.setattr(RecoverableRegisteredEvaluator,'evaluate',crash_after_scoring)
    with pytest.raises(SystemExit):ReferenceWorkflow(tmp_path/'state',tmp_path/'out').run_demo()
    with sqlite3.connect(tmp_path/'state/demo_registry.sqlite') as db:
        before=db.execute('SELECT COUNT(*) FROM trial_counts').fetchone()[0]
    monkeypatch.setattr(RecoverableRegisteredEvaluator,'evaluate',original)
    result=ReferenceWorkflow(tmp_path/'state',tmp_path/'out').run_demo()
    assert result['software_status']=='PASS'
    with sqlite3.connect(tmp_path/'state/demo_registry.sqlite') as db:
        assert db.execute('SELECT COUNT(*) FROM trial_counts').fetchone()[0]>before
        assert db.execute("SELECT COUNT(*) FROM trial_counts WHERE kind='scored_variant'").fetchone()[0]==16
    with sqlite3.connect(tmp_path/'state/demo_evaluation.sqlite') as db:
        assert db.execute('SELECT COUNT(*) FROM comparison_trials').fetchone()[0]==16

def make_registered_exception_agent(tmp_path,*,adopt=True,released=None,context_limit=8192):
    from pod_platform.application.twin_exhibit import ProviderExceptionAgent
    from pod_platform.agents.provider import DeterministicMockProvider
    from pod_platform.agents.durable_cache import DurableExtractionCache
    from pod_platform.agents.budgets import PersistentCallBudget
    from pod_platform.agents.registry import ModelRegistry,ModelCheckpoint,CutoffEvidence,AdoptionRecord,TemporalPolicy
    at=dt.datetime(2014,1,1,tzinfo=UTC)
    provider=DeterministicMockProvider(lambda req:'{"action":"retain"}',model='fixture-mock-exception')
    registry=ModelRegistry()
    checkpoint=ModelCheckpoint(owner='integrity',created_at=at,checkpoint_id='mock_checkpoint',repository='fixture',
        exact_model_id=provider.model,checkpoint_revision='fixed-test-revision',weight_hash='f'*64,
        release_evidence_url='https://example.invalid/FIXTURE_NOT_REAL_CHECKPOINT',
        cutoff_evidence=CutoffEvidence(checkpoint_released_at=released or at,evidence_note='test-only exact checkpoint fixture'),
        tokenizer_revision='fixture',chat_template_revision='fixture',quantization='none',serving_engine='deterministic_mock',
        context_limit_verified=context_limit,reasoning_config='none',benchmark_artifact_id='fixture_benchmark',admitted=True).sealed()
    registry.register(checkpoint)
    if adopt:
        registry.adopt(AdoptionRecord(owner='integrity',created_at=at,adoption_id='mock_adoption',checkpoint_id=checkpoint.checkpoint_id,
            role='lead',temporal_policy=TemporalPolicy.RELEASE_GATED,effective_start=at,shadow_comparison_id='fixture_fixed_task_comparison').sealed())
    budget=PersistentCallBudget(tmp_path/'budget.sqlite',budget_id='exception-test',max_calls=100,max_tokens=200000)
    agent=ProviderExceptionAgent(provider,DurableExtractionCache(tmp_path/'cache.sqlite'),budget,registry=registry)
    return agent

def test_adopted_mock_provider_full_pair_is_cached_and_never_llm_skill(tmp_path,demo):
    from pod_platform.application.twin_exhibit import build_twin_exhibit
    from pod_platform.orchestration.replay import load_fixture
    import shutil
    root,_=demo
    agent=make_registered_exception_agent(tmp_path/'agent')
    run=tmp_path/'book'
    shutil.copytree(root/'out/book',run,ignore=shutil.ignore_patterns('paired_replays','twin_exhibit.json'))
    twin=build_twin_exhibit(run,root/'out/fixture',agent=agent,certificate_policy='test')
    assert twin['status']=='COMPLETED' and agent.calls==32
    assert all(r['provider_is_mock'] for r in twin['provider_records'])
    assert not twin['llm_skill_evidence']
    assert all(r['authorization']['label']=='historical_release_gated' for r in twin['provider_records'])
    before=agent.budget.calls_used
    second=tmp_path/'book2'
    shutil.copytree(root/'out/book',second,ignore=shutil.ignore_patterns('paired_replays','twin_exhibit.json'))
    reopened=make_registered_exception_agent(tmp_path/'agent')
    again=build_twin_exhibit(second,root/'out/fixture',agent=reopened,certificate_policy='test')
    assert again['paired_nav_difference']==pytest.approx(twin['paired_nav_difference'])
    assert reopened.calls==0 and reopened.budget.calls_used==before
    assert all(r['from_cache'] for r in again['provider_records'])

def test_agent_missing_adoption_future_release_or_oversize_context_denied(tmp_path):
    from pod_platform.application.twin_exhibit import ProviderExceptionAgent
    context={'decision_at':'2015-02-19T21:00:00+00:00','information_cutoff':'2015-02-18T00:00:00+00:00',
        'period':'2015-02-19','security_id':'S000','weight':'.125','exception':'name concentration','policy':'fixed','paired_input_hash':'f'*64}
    missing=make_registered_exception_agent(tmp_path/'missing',adopt=False)
    with pytest.raises(PodPlatformError):missing.decide(context)
    future=make_registered_exception_agent(tmp_path/'future',released=dt.datetime(2020,1,1,tzinfo=UTC))
    with pytest.raises(PodPlatformError):future.decide(context)
    narrow=make_registered_exception_agent(tmp_path/'narrow',context_limit=100)
    with pytest.raises(ValueError,match='context'):narrow.decide(context)
    ready=make_registered_exception_agent(tmp_path/'ready')
    with pytest.raises(ValueError,match='unexpected'):ready.decide({**context,'realized_return':.3})
    assert ready.calls==missing.calls==future.calls==narrow.calls==0

def test_interrupted_comparison_settles_same_charged_row(tmp_path):
    import sqlite3
    from pod_platform.application.recovery import RecoverableRegisteredEvaluator
    from pod_platform.research.evaluation import ComparisonPlan,ForecastOutcome
    from pod_platform.application.synthetic import build_world
    world=build_world();path=tmp_path/'evaluations.sqlite'
    evaluator=RecoverableRegisteredEvaluator(path)
    plan=ComparisonPlan(experiment_id='retry',owner='fixture',hypothesis_id='h',registered_at=world['registered_at'],
        target_id='target',source_ids=['synthetic_signal'],variant_budget=1,min_dates=2)
    evaluator.register(plan)
    kwargs=dict(owner='fixture',evaluation_asof=world['evaluation_asof'],source_cards=[world['card']])
    expected=evaluator.evaluate('retry','one',world['design'],**kwargs)
    # Fault injection models a process exit immediately after reservation but
    # before result settlement. The consumed row remains present throughout.
    with evaluator.connection:evaluator.connection.execute("UPDATE comparison_trials SET result=NULL WHERE experiment_id='retry'")
    actual=evaluator.evaluate('retry','one',world['design'],**kwargs)
    assert actual==expected
    assert evaluator.connection.execute('SELECT COUNT(*) FROM comparison_trials').fetchone()[0]==1
    with evaluator.connection:evaluator.connection.execute("UPDATE comparison_trials SET result=NULL WHERE experiment_id='retry'")
    altered=[o.model_copy(update={'prediction':-o.prediction}) for o in world['design']]
    with pytest.raises(ValueError,match='input hash changed'):evaluator.evaluate('retry','one',altered,**kwargs)
    assert evaluator.connection.execute('SELECT COUNT(*) FROM comparison_trials').fetchone()[0]==1
    evaluator.close()
