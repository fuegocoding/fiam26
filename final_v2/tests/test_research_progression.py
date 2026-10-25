"""Cheap research access never grants full design/check/trading authority."""
import datetime as dt
import json
import sqlite3
import csv
import hashlib

import pytest

from pod_platform.agents.provider import DeterministicMockProvider
from pod_platform.application.research_loop import AutonomousResearchLoop, build_fixture_loop
from pod_platform.research.progression import ProgressionPolicy, ResearchProgression
from pod_platform.research.builtins.catalog import RECIPES
from pod_platform.services.app import build_server

NOW = dt.datetime(2026, 10, 25, tzinfo=dt.timezone.utc)


def test_persistent_budget_reuse_negative_scope_and_policy_binding(tmp_path):
    path = tmp_path / 'stages.sqlite'
    policy = ProgressionPolicy(max_probes_per_pod=2, max_total_cases=3)
    stages = ResearchProgression(path, code_digest='a'*64, clock=lambda: NOW, policy=policy)
    first = stages.reserve('pod_fundamentals', stage='component_probe', fingerprint='sales', mechanism_key='sales', cases=2)
    stages.complete(first['attempt_id'], {'status': 'negative'}, negative=True)
    assert stages.reserve('pod_fundamentals', stage='component_probe', fingerprint='sales', mechanism_key='sales', cases=2)['status'] == 'reused'
    assert stages.reserve('pod_fundamentals', stage='component_probe', fingerprint='retune_sales', mechanism_key='sales')['status'] == 'stopped'
    # A different component is still testable after the sales component fails.
    other = stages.reserve('pod_fundamentals', stage='component_probe', fingerprint='income', mechanism_key='income')
    assert other['status'] == 'reserved'
    assert stages.reserve('pod_dynamics', stage='component_probe', fingerprint='momentum', mechanism_key='momentum')['reason'] == 'total_case_budget_exhausted'
    restarted = ResearchProgression(path, code_digest='a'*64, clock=lambda: NOW, policy=policy)
    assert restarted.snapshot()['cases_consumed'] == 3
    with pytest.raises(ValueError, match='immutable'):
        ResearchProgression(path, code_digest='b'*64, clock=lambda: NOW, policy=policy)
    with pytest.raises(ValueError, match='immutable'):
        ResearchProgression(path, code_digest='a'*64, clock=lambda: NOW)
    with pytest.raises(ValueError, match='strict authority'):
        stages.reserve('pod_fundamentals', stage='trade', fingerprint='trade', mechanism_key='trade')
    with sqlite3.connect(path) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute('DELETE FROM progression_events')


def test_failed_measurement_does_not_disprove_the_mechanism(tmp_path):
    stages = ResearchProgression(tmp_path/'stages.sqlite', code_digest='a'*64, clock=lambda: NOW)
    r = stages.reserve('pod_fundamentals', stage='component_probe', fingerprint='bad_data', mechanism_key='sales')
    stages.complete(r['attempt_id'], {'status': 'failed'}, negative=False)
    assert stages.reserve('pod_fundamentals', stage='component_probe', fingerprint='new_data', mechanism_key='sales')['status'] == 'reserved'


def grant(loop, namespace='pod_fundamentals', features=None):
    recipe = RECIPES['fund_earnings_innovation_v1']
    return dict(namespace=namespace, grant_id='native_sales', recipe_id=recipe.recipe_id,
        recipe_digest=recipe.digest(), feature_names=features or ['sale_gr1_rank'],
        dataset_id='fixture_fundamentals', reviewed_by='fixture_host', reviewed_at=loop.clock().isoformat(),
        code_digest=loop.profile['code'])


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    loop = build_fixture_loop(tmp_path_factory.mktemp('components_runtime'))
    loop.review_component('pod_fundamentals', grant(loop))
    loop.submit_hypothesis('pod_fundamentals', hypothesis())
    return loop


def hypothesis():
    return dict(grant_id='native_sales', hypothesis='Sales growth may predict later earnings innovations.',
                falsification='Reject if this fixed transform has no advantage over earnings persistence.',
                competing_explanation='The effect could reflect existing earnings persistence or noise.',
                unavailable_controls=['No independent holdout or sector controls for this diagnostic.'])


def test_own_reasoning_needs_no_paper_and_rewording_reuses_measurement(world):
    loop = world
    loop.review_component('pod_fundamentals', grant(loop))
    first = loop.submit_hypothesis('pod_fundamentals', hypothesis())
    assert first['status'] == 'component_probe_completed'
    assert first['probe_report']['status'] == 'positive_signal'
    assert first['lineage']['origin'] == 'own_reasoning'
    second = loop.submit_hypothesis('pod_fundamentals', {**hypothesis(), 'hypothesis': 'Different wording: growing sales may predict earnings.'})
    assert second['measurement_reused']
    assert second['probe_id'] == first['probe_id']
    assert len(loop.progression.snapshot()['attempts']) == 1
    assert not loop._events('adaptation') and not loop._events('outcome') and not loop._events('publication')
    assert not first['admitted'] and not first['check_opened']
    assert loop.submit_hypothesis('pod_fundamentals', hypothesis()) == first
    with pytest.raises(ValueError, match='trusted host grant'):
        loop.submit_hypothesis('pod_valuation', hypothesis())


def test_single_paper_excerpt_reuses_test_and_keeps_new_provenance(world):
    loop = world
    review = loop._events('review', 'pod_fundamentals')[0]
    card = loop.papers.get(review['paper_id'])
    request = {'grant_id': 'native_sales', 'component': dict(namespace='pod_fundamentals',
        component_kind='mechanism', component_name='Selected leading indicator mechanism',
        paper_id=card.paper_id, version_key=card.version_key, claim_ids=[card.claims[0].claim_id],
        source='full_text', unavailable_controls=['Synthetic excerpt is not investment evidence.'])}
    result = loop.submit_component('pod_fundamentals', request)
    assert result['measurement_reused']
    assert result['lineage']['component']['replication_status'] == 'not_replication'
    assert len(result['lineage']['component']['claims']) == 1
    assert len(loop.progression.snapshot()['attempts']) == 1
    assert not result['admitted'] and not result['check_opened']
    bad = {**request, 'component': {**request['component'], 'namespace': 'pod_events'}}
    with pytest.raises(ValueError, match='namespace'):
        loop.submit_component('pod_fundamentals', bad)


def test_fixed_role_mcp_component_tools_cannot_self_grant_or_override_scope(world):
    loop = world
    # A runtime with online paper access must not silently enable an offline MCP.
    loop.papers.online = True
    loop.papers._fetch = lambda *_a, **_k: pytest.fail('offline MCP must not fetch')
    server = build_server(loop.state_dir, 'fundamentals', clock=loop.clock,
        discovery=loop.discovery, research_runtime=loop)
    server.dispatch({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {}})
    def call(name, args):
        return server.dispatch({'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call',
            'params': {'name': name, 'arguments': args}})['result']
    assert not call('research.hypothesis_submit', hypothesis())['isError']
    assert call('papers.get', {'paper_id': 'arxiv:2001.99999v1'})['isError']
    caps = json.loads(call('research.component_capabilities', {})['content'][0]['text'])
    assert {g['namespace'] for g in caps['grants']} == {'pod_fundamentals'}
    for extra in ({'namespace': 'pod_events'}, {'code': 'exec(...)'}, {'partition': 'check'}, {'admitted': True}):
        assert call('research.hypothesis_submit', {**hypothesis(), **extra})['isError']
    review = loop._events('review', 'pod_fundamentals')[0]
    card = loop.papers.get(review['paper_id'])
    # MCP paper reads, quote selections and component submission use the same cache.
    assert not call('papers.get', {'paper_id': card.paper_id})['isError']
    assert not call('papers.component_quote', {'paper_id': card.paper_id, 'version_key': card.version_key,
        'quote': card.claims[0].quote, 'source': 'full_text', 'kind': 'statistical_design'})['isError']
    component = dict(component_kind='method', component_name='One exact cached excerpt',
        paper_id=card.paper_id, version_key=card.version_key, source='full_text', claim_ids=[card.claims[0].claim_id])
    assert not call('research.component_submit', {'grant_id': 'native_sales', 'component': component})['isError']
    for extra in ({'namespace': 'pod_events'}, {'as_of': '2020-01-01T00:00:00Z'}, {'formation_valid': True}):
        assert call('research.component_submit', {'grant_id': 'native_sales', 'component': {**component, **extra}})['isError']
    assert 'research.review_component' not in server.tools
    assert 'research.tick' not in server.tools
    server.close()
    assert loop.papers.get(review['paper_id']).paper_id == card.paper_id


@pytest.mark.parametrize('disposition,category,expected', [
    ('review', 'testable', 'component_probe_completed'),
    ('reject', 'testable', 'component_probe_completed'),
    ('review', 'incomplete_evidence', 'review_required'),
    ('reject', 'unsupported_source', 'review_required'),
    ('proceed', 'unknown_method', 'review_required'),
])
def test_challenger_can_allow_falsification_without_full_strategy(tmp_path, disposition, category, expected):
    loop = build_fixture_loop(tmp_path)
    def challenger(request):
        payload = json.loads(request.user)
        return json.dumps(dict(disposition=disposition, reason_category=category,
            evidence_refs=payload['evidence_ids'], competing_explanation='Noise or the fixed baseline may explain the result.',
            falsification_test='Reject if the preregistered bounded measurement fails the baseline.'))
    for runner in loop.challengers.values():
        runner.provider = DeterministicMockProvider(challenger, 'fixture-review', clock=loop.clock)
    loop.tick()
    routes = loop._events('route')
    assert len(routes) == 4
    assert all(r['status'] == expected for r in routes)
    assert not loop._events('adaptation') and not loop._events('outcome') and not loop._events('publication')
    if expected == 'component_probe_completed':
        assert all(not r['admitted'] and not r['check_opened'] for r in routes)
        assert len(loop.progression.snapshot()['attempts']) == 4


@pytest.mark.parametrize('disposition', ['review', 'proceed'])
def test_saved_output_replay_runs_offline_and_is_repeatable(tmp_path, disposition):
    from pod_platform.application.component_replay import replay_components
    source = build_fixture_loop(tmp_path/'captured')
    def critic(request):
        return json.dumps(dict(disposition=disposition, reason_category='testable',
            evidence_refs=json.loads(request.user)['evidence_ids'],
            competing_explanation='Noise or the native persistence baseline explains this.',
            falsification_test='Compare the fixed feature and native target against the baseline.'))
    for runner in source.challengers.values():
        runner.provider = DeterministicMockProvider(critic, 'captured-fixture-critic', clock=source.clock)
    source.tick()
    args = dict(from_research=source.state_dir, state_dir=tmp_path/'replay',
        manifest_path=source.builtin.manifest.path, papers_db=source.papers.store.path)
    first = replay_components(**args)
    assert first['new_model_api_calls'] == first['admissions'] == first['orders'] == 0
    assert len(first['comparisons']) == 4
    assert all(c['route']['status'] == 'component_probe_completed' for c in first['comparisons'])
    assert replay_components(**args) == first


def test_proceed_cannot_disguise_unimplemented_features_as_a_full_design(tmp_path):
    loop = build_fixture_loop(tmp_path)
    original = loop.providers.lead._responder
    def unmatched(request):
        proposal = json.loads(original(request))
        proposal['features'][0]['name'] = 'missing_authentic_sentiment'
        proposal['features'][0]['definition'] = 'Requires unavailable social sentiment measurements.'
        return json.dumps(proposal)
    for runner in loop.runners.values():
        runner.extractor.provider = DeterministicMockProvider(unmatched, 'fixture-unmatched-lead', clock=loop.clock)
    loop.tick()
    routes = loop._events('route')
    assert all(r['status'] == 'component_probe_completed' for r in routes)
    assert all(r['lineage']['untested_core_dependencies'] for r in routes)
    assert not loop._events('adaptation') and not loop._events('publication')


def test_runtime_can_grant_research_only_dataset_without_financial_metadata(tmp_path):
    source = build_fixture_loop(tmp_path/'base')
    raw = json.loads(source.builtin.manifest.path.read_text())
    spec = raw['datasets']['fixture_fundamentals']
    path = tmp_path/'minimal.csv'
    original = source.builtin.manifest.dataset('fixture_fundamentals').path
    with original.open(newline='') as src, path.open('w', newline='') as dest:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dest, fieldnames=['permno','eom','niq_su','sale_gr1'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(reader)
    for name in ('label','evaluation','sector_column','market_cap_column','exposure_columns','column_aliases'):
        spec.pop(name, None)
    spec['columns'] = {k:v for k,v in spec['columns'].items() if k in ('niq_su','sale_gr1')}
    spec['path'] = str(path)
    spec['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    spec['source_card'].update(formation_eligible=False, status='SAMPLED')
    raw['datasets'] = {'fixture_fundamentals': spec}
    manifest = tmp_path/'component_manifest.json'
    manifest.write_text(json.dumps(raw))
    loop = AutonomousResearchLoop(tmp_path/'runtime', providers=source.providers,
        manifest_path=source.builtin.manifest.path, component_manifest_path=manifest, clock=source.clock,
        run_id='minimal_data_runtime')
    loop.review_component('pod_fundamentals', grant(loop))
    result = loop.submit_hypothesis('pod_fundamentals', hypothesis())
    assert result['probe_report']['status'] == 'positive_signal'
    assert not result['admitted'] and not result['check_opened']
