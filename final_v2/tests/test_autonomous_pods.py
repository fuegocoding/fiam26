"""Real registered research/challenge/publication integration; fixtures prove mechanics only."""
import json
from pathlib import Path
import pytest

from pod_platform.application.research_loop import build_fixture_loop, RoleProviders, NAMESPACES
from pod_platform.services.app import build_server


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp('autonomous')
    loop = build_fixture_loop(root)
    result = loop.tick()
    return root, loop, result


def test_four_private_pods_measure_admissions_rejections_and_feedback(world):
    root, loop, result = world
    outcomes = {o['namespace']: o for o in result['outcomes']}
    assert set(outcomes) == set(NAMESPACES)
    assert outcomes['pod_fundamentals']['outcome'] == 'ADMITTED'
    assert outcomes['pod_dynamics']['outcome'] == 'ADMITTED'
    assert outcomes['pod_valuation']['outcome'] == 'SHADOW'
    assert outcomes['pod_events']['outcome'] == 'DESIGN_EXECUTION_FAILED'
    assert not outcomes['pod_events']['check_opened']
    assert len(result['publications']) == 4
    assert all(p['scope'] == 'fixture' for p in result['publications'])
    assert all(d['status'] == 'not_configured' for d in result['delivery_receipts'])
    for namespace in NAMESPACES:
        assert len(loop.memory(namespace).list(kind='draft_proposal')) >= 1
        budgets = loop.status(namespace)['pods'][namespace]['budgets']
        assert all(v['calls_used'] == 1 for v in budgets.values())
    for p in result['publications']:
        chain = json.loads(Path(p['lineage_path']).read_text())
        assert [item['stage'] for item in chain] == ['source','method','hypothesis','experiment','result']
        forecasts = json.loads(Path(p['forecast_path']).read_text())
        assert forecasts and all(f['return_forecast']['horizon_trading_days'] == 21 for f in forecasts)
        assert all(f['target']['unit'] == 'centered_rank' for f in forecasts)
        # Native fundamental target and executable return bridge retain distinct horizons.
        if p['namespace'] == 'pod_fundamentals':
            assert forecasts[0]['target']['expected_resolution_at'] > forecasts[0]['expires_at']


def test_identical_resume_keeps_calls_trials_check_access_and_publications(world):
    root, old, initial = world
    before = {ns: old.status(ns)['pods'][ns]['budgets'] for ns in NAMESPACES}
    accesses = {o['candidate_id']: old.bridge.run_state(o['candidate_id']) for o in initial['outcomes'] if 'candidate_id' in o}
    loop = build_fixture_loop(root)
    result = loop.tick()
    assert result['publications'] == initial['publications']
    assert result['jobs_this_tick'] == 0
    assert {ns: loop.status(ns)['pods'][ns]['budgets'] for ns in NAMESPACES} == before
    assert {cid: loop.bridge.run_state(cid) for cid in accesses} == accesses


def test_callback_failure_retries_without_model_or_check_reexecution(world):
    root, loop, initial = world
    def broken(_):
        raise RuntimeError('offline consumer')
    loop.on_publication = broken
    with pytest.raises(RuntimeError):
        loop.tick()
    delivered = []
    loop.on_publication = lambda p: delivered.append(p) or {'consumer':'test','status':'received'}
    result = loop.tick()
    assert len(delivered) == 4 and all('decision_path' not in p for p in delivered)
    assert not any('check_mean_ic' in json.dumps(p) for p in delivered)
    assert all(d['status'] == 'delivered' for d in result['delivery_receipts'])
    loop.tick()
    assert len(delivered) == 4


def test_fixed_role_mcp_uses_same_runtime_and_cannot_choose_namespace(world):
    root, loop, _ = world
    server = build_server(root, 'valuation', clock=loop.clock, discovery=loop.discovery, research_runtime=loop)
    server.dispatch({'jsonrpc':'2.0','id':1,'method':'initialize','params':{}})
    def call(name,args):
        return server.dispatch({'jsonrpc':'2.0','id':2,'method':'tools/call','params':{'name':name,'arguments':args}})
    result = call('research.cycle_status',{})['result']
    assert not result['isError']
    view = json.loads(result['content'][0]['text'])
    assert set(view['pods']) == {'pod_valuation'}
    assert call('research.cycle_status',{'namespace':'pod_fundamentals'})['result']['isError']
    assert 'integrity.evaluate_protected' not in server.tools
    assert 'research.tick' not in server.tools
    assert 'sources.discover' in server.tools and 'catalogue.search' in server.tools
    server.close()


def test_future_evidence_unknown_namespace_and_invalid_backend_fail_before_calls(world):
    _, loop, _ = world
    with pytest.raises(ValueError):
        loop.add_evidence('pod_unknown',subject='x',question='a question')
    with pytest.raises(ValueError):
        loop.add_evidence('pod_events',subject='future',question='a question',paper_cards=[
            {'paper_id':'future','title':'Future paper','abstract':'unknown','available_at':'2099-01-01T00:00:00Z'}])
    with pytest.raises(ValueError,match='BLOCKED_BACKEND'):
        RoleProviders(None,None,None).identities()
    with pytest.raises(ValueError):
        loop.tick(max_jobs=0)


def test_failed_model_calls_are_charged_and_never_create_experiments(tmp_path):
    from pod_platform.agents.provider import ProviderError
    loop = build_fixture_loop(tmp_path)
    def failure(request):
        raise ProviderError('fixture unavailable')
    loop.providers.lead._responder = failure
    result = loop.tick()
    assert not result['publications'] and not result['outcomes']
    assert result['worker_outcomes'] and all(o['state']=='failed' for o in result['worker_outcomes'])
    for ns in NAMESPACES:
        assert loop.status(ns)['pods'][ns]['budgets']['lead']['calls_used'] >= 1
        assert not loop.memory(ns).evaluated_results()


def test_publication_reads_current_features_without_labels(world, monkeypatch):
    import pod_platform.research.challenge_bridge as module
    root,loop,result=world
    publication=result['publications'][0]
    frozen=loop.bridge.frozen(publication['candidate_id'])
    calls=[]
    original=module.load_panel
    def spy(dataset,columns,**kwargs):
        calls.append(set(columns))
        return original(dataset,columns,**kwargs)
    monkeypatch.setattr(module,'load_panel',spy)
    # All published inputs follow the protected historical check; they are
    # feature-only current snapshots, never check-label-dependent forecasts.
    for p in result['publications']:
        forecasts=json.loads(Path(p['forecast_path']).read_text())
        assert all(f['information_cutoff'].startswith(('2026-06','2026-07')) for f in forecasts)
    from pod_platform.application.integrity import IntegrityService, RegisteredEvaluator
    from pod_platform.application.namespaces import RoleContext
    import datetime as dt
    service=IntegrityService(registry=loop.registry,ledger=loop.ledger,
        evaluator=RegisteredEvaluator(root/'integrity_evaluations.sqlite'),policy=loop.policy,
        source_cards=[loop.bridge.integrity_source_card(loop.builtin,frozen)],
        evidence_seal=loop.authority.evidence_seal(),allow_unbound_fixture=False)
    ctx=RoleContext('integrity','integrity',loop.run_id,loop.clock(),('design','check'))
    auth=loop.authority.issue(ctx,frozen,check_start=loop.check_start,check_end=loop.check_end,
        issued_at=loop.clock(),expires_at=loop.clock()+dt.timedelta(days=1),max_family_accesses=1)
    module.ChallengeBridge.publish_forecasts(loop.bridge,RoleContext('strategy',frozen.namespace,loop.run_id,loop.clock()),
        loop.builtin,service,loop.publication,loop.artifacts,frozen.candidate_id,auth,
        published_at=dt.datetime(2026,7,1,tzinfo=dt.timezone.utc),information_month=2026*12+5,
        feature_window=(dt.date(2026,6,30),dt.date(2026,6,30)))
    assert calls and all('ret_exc_lead1m' not in c for c in calls)


def test_crash_after_sealed_check_recovers_original_certificate_without_second_look(tmp_path):
    loop=build_fixture_loop(tmp_path)
    def crash(*args,**kwargs):
        raise RuntimeError('crash between challenge and publication')
    loop.bridge.publish_forecasts=crash
    with pytest.raises(RuntimeError,match='between challenge'):
        loop.tick()
    resumed=build_fixture_loop(tmp_path)
    result=resumed.tick()
    assert len(result['publications'])==4
    for p in result['publications']:
        frozen=resumed.bridge.frozen(p['candidate_id'])
        assert resumed.bridge.family_accesses(frozen.mechanism_family)==1
    assert all(resumed.status(ns)['pods'][ns]['budgets']['lead']['calls_used']==1 for ns in NAMESPACES)


def test_live_clock_resume_preserves_registration_clock_and_never_future_dates_publication(world,tmp_path):
    import datetime as dt
    from pod_platform.application.research_loop import AutonomousResearchLoop
    root,fixture,_=world
    start=fixture.clock()
    kwargs={'providers':fixture.providers,'manifest_path':root/'fixture_data/manifest.json'}
    loop=AutonomousResearchLoop(tmp_path,clock=lambda:start,**kwargs)
    assert loop.publication_dates==(start,)
    resumed=AutonomousResearchLoop(tmp_path,clock=lambda:start+dt.timedelta(days=1),**kwargs)
    assert resumed.started_at==start and resumed.publication_dates==(start,)
    assert resumed.profile==loop.profile
