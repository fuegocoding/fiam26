"""No model, network, weight download or billable call is used in these tests."""
import datetime as dt
import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest

from pod_platform.agents.model_trials import ModelTrial, load_tasks, public_task, trial_lock
from pod_platform.agents.provider import (
    DeterministicMockProvider, GenerationParams, LLMRequest, ProviderError,
)
from pod_platform.agents.strategy_models import MODEL_IDS, ServingProfile, StrategyModelProvider, load_profiles

ROOT = Path(__file__).resolve().parents[1]
NOW = dt.datetime(2026, 10, 25, 16, tzinfo=dt.timezone.utc)


def clock():
    return NOW


def profiles():
    return load_profiles(ROOT/'configs/models/serving.example.json')


def request():
    return LLMRequest(system='JSON only', user='synthetic evidence', params=GenerationParams(max_tokens=100), role='extractor')


def completion(model, **message):
    return 200, json.dumps({'model': model, 'choices': [{'message': {'content': '{"ok":true}', **message}, 'finish_reason':'stop'}],
        'usage': {'prompt_tokens':10, 'completion_tokens':5}}).encode()


def test_profiles_pin_exact_spec_models_and_cache_identity():
    ps = profiles()
    assert {r:p.repository for r,p in ps.items()} == {r:MODEL_IDS[r] for r in ps}
    assert all(p.revision and p.weight_hashes for p in ps.values())
    altered = ps['extractor'].model_copy(update={'generation':{'top_p':.7}})
    assert StrategyModelProvider(altered).name != StrategyModelProvider(ps['extractor']).name
    bad = ps['lead'].model_dump();bad['repository'] = MODEL_IDS['lead_upgrade']
    with pytest.raises(ValueError, match='exact repository'):
        ServingProfile.model_validate(bad)


@pytest.mark.parametrize('change', [
    {'base_url':'http://remote.example/v1'},
    {'base_url':'https://secret@remote.example/v1'},
    {'generation':{'tools':[]}},
    {'generation':{'num_ctx':1024}},
    {'generation':{'top_p':float('nan')}},
])
def test_profile_rejects_unsafe_or_unreviewed_configuration(change):
    raw = profiles()['extractor'].model_dump();raw.update(change)
    with pytest.raises(ValueError):
        ServingProfile.model_validate(raw)


@pytest.mark.parametrize('role', ['lead','challenger','extractor'])
def test_request_uses_native_role_controls_and_no_tools(role):
    p = profiles()[role];bodies=[]
    def transport(url, headers, body, timeout, maximum):
        assert headers['User-Agent']=='fiam-pod-platform/0.1'
        bodies.append(json.loads(body));return completion(p.served_model)
    provider = StrategyModelProvider(p, transport=transport)
    assert provider.complete(request()).served_model == p.served_model
    body = bodies[0]
    assert body['model'] == MODEL_IDS[role] and 'tools' not in body
    if role=='extractor':assert body['chat_template_kwargs']=={'enable_thinking':False}
    if role=='challenger':assert body['reasoning_effort']=='medium'
    if role=='lead':assert 'chat_template_kwargs' not in body


@pytest.mark.parametrize('response', ['different_model','reasoning','tools'])
def test_no_silent_substitution_reasoning_or_tool_calls(response):
    p = profiles()['extractor']
    body = completion('other' if response=='different_model' else p.served_model,
        **({'reasoning_content':'thinking'} if response=='reasoning' else {'tool_calls':[{}]} if response=='tools' else {}))
    with pytest.raises(ProviderError):
        StrategyModelProvider(p, transport=lambda *a:body).complete(request())


def ollama_profile():
    raw=profiles()['extractor'].model_dump()
    raw.update(protocol='ollama',base_url='http://127.0.0.1:11434',served_model='qwen3:8b',ollama_digest='a'*64,
        quantization='Q4_K_M',generation={'num_ctx':1024,'num_gpu':0})
    return ServingProfile.model_validate(raw)


def test_ollama_digest_checks_native_non_thinking_and_final_text():
    p=ollama_profile();seen=[];lookups=[]
    def tags():
        lookups.append(1);return {'models':[{'name':p.served_model,'digest':p.ollama_digest}]}
    def transport(url, headers, body, timeout, maximum):
        body=json.loads(body);seen.append((url,body))
        if url.endswith('/api/show'):return 200,json.dumps({'details':{'family':'qwen3'}}).encode()
        return 200,json.dumps({'model':p.served_model,'message':{'content':'{"ok":true}'},'done':True,'done_reason':'stop',
            'prompt_eval_count':10,'eval_count':5}).encode()
    response=StrategyModelProvider(p,transport=transport,metadata_get=tags).complete(request())
    assert response.text=='{"ok":true}' and len(lookups)==2
    assert seen[-1][1]['think'] is False and seen[-1][1]['keep_alive']==0
    assert seen[-1][1]['options']['num_predict']==100
    assert response.total_tokens is None


def test_ollama_changed_tag_never_calls_model():
    calls=[];p=ollama_profile()
    provider=StrategyModelProvider(p,transport=lambda *a:calls.append(a),metadata_get=lambda:{'models':[]})
    with pytest.raises(ProviderError,match='digest'):
        provider.complete(request())
    assert calls==[]


def test_explicit_response_schema_is_hashed_and_sent_to_ollama():
    import dataclasses
    p=ollama_profile();bodies=[]
    schema={'type':'object','properties':{'answer':{'type':'string'}},'required':['answer']}
    typed=dataclasses.replace(request(),response_schema=schema)
    assert typed.request_hash != request().request_hash
    def transport(url,headers,body,timeout,maximum):
        bodies.append(json.loads(body))
        if url.endswith('/api/show'):return 200,json.dumps({'details':{'family':'qwen3'}}).encode()
        return 200,json.dumps({'model':p.served_model,'message':{'content':'{"answer":"ok"}'},
            'done':True,'done_reason':'stop','prompt_eval_count':10,'eval_count':5}).encode()
    provider=StrategyModelProvider(p,transport=transport,metadata_get=lambda:{'models':[{'name':p.served_model,'digest':p.ollama_digest}]})
    provider.complete(typed)
    assert bodies[-1]['format']==schema


def task_data():
    return load_tasks(ROOT/'configs/models/benchmark_v1.json')


def test_fifty_task_mix_and_gold_not_in_public_payload():
    tasks=task_data()
    assert len(tasks)==50 and len({t['id'] for t in tasks})==50
    for task in tasks:
        public=public_task(task)
        assert set(public)=={'id','category','question','evidence','choices'}
        assert 'expected' not in public and 'gold_quote' not in public and 'critical' not in public


def workflow_mocks(ps, seen=None):
    seen = [] if seen is None else seen
    def reply(req):
        payload=json.loads(req.user);seen.append((req.role,payload))
        if req.role=='extractor':
            d={'available_at':'2026-10-01T18:00:00Z','measure':'store visits grew 8%',
               'source_quote':'store visits grew 8%','missing_inputs':['costs']}
        elif req.role=='lead':
            assert 'extractor' in payload['prior_role_outputs']
            d={'mechanism':'volume may mask weaker spending','experiment':'compare volume-only against volume-plus-spend',
               'baseline':'volume-only','failure_rule':'no paired improvement','source_quote':'spend per visit fell 6%',
               'missing_inputs':['costs']}
        else:
            assert 'lead' in payload['prior_role_outputs']
            d={'disconfirming_tests':['control for store coverage'],'unsupported_claims':['tradable alpha'],
               'source_quote':'No publication revision history','abstain_from_admission':True}
        return json.dumps(d)
    return {role:DeterministicMockProvider(reply,model=p.served_model,clock=clock) for role,p in ps.items()}


def packet():
    return json.loads((ROOT/'configs/models/trial_packet.example.json').read_text())


def test_three_real_call_path_stages_reuse_cache_and_preserve_budget(tmp_path):
    ps=profiles();seen=[];providers=workflow_mocks(ps,seen)
    trial=ModelTrial(tmp_path,'workflow',ps,providers=providers,clock=clock)
    first=trial.workflow(packet(),stop_after='extractor')
    assert first['status']=='STAGE_COMPLETED' and first['budget']['calls_used']==1
    complete=trial.workflow(packet())
    assert complete['status']=='COMPLETED_RESEARCH_DRAFT'
    assert [r for r,_ in seen]==['extractor','lead','challenger']
    assert complete['provider_is_mock'] and not complete['adoption_allowed']
    assert complete['budget']['calls_used']==3 and complete['roles'][0]['from_cache']
    again=ModelTrial(tmp_path,'workflow',ps,providers=providers,clock=clock).workflow(packet())
    assert all(r['from_cache'] for r in again['roles']) and again['budget']['calls_used']==3
    assert len(list((tmp_path/'workflow/attempts').glob('*.json')))>=7


def test_changed_source_bound_trial_is_rejected_without_new_calls(tmp_path):
    ps=profiles();providers=workflow_mocks(ps)
    trial=ModelTrial(tmp_path,'workflow',ps,providers=providers,clock=clock)
    trial.workflow(packet(),stop_after='extractor')
    with pytest.raises(ValueError,match='binding changed'):
        trial.workflow({'evidence':'different input'})
    assert trial.budget.calls_used==1


def test_missing_role_blocks_before_any_call(tmp_path):
    ps=profiles();del ps['lead']
    trial=ModelTrial(tmp_path,'workflow',ps,providers=workflow_mocks(ps),clock=clock)
    with pytest.raises(ValueError,match='requires profiles'):
        trial.workflow(packet())
    assert trial.budget.calls_used==0


def test_malformed_or_hallucinated_extraction_fails_and_spends_budget(tmp_path):
    ps=profiles();providers=workflow_mocks(ps)
    providers['extractor']=DeterministicMockProvider(lambda req: json.dumps({'available_at':'invented',
        'measure':None,'source_quote':'store visits grew 8%','missing_inputs':[]}),clock=clock)
    trial=ModelTrial(tmp_path,'workflow',ps,providers=providers,clock=clock)
    result=trial.workflow(packet())
    assert result['status']=='FAILED_CLOSED' and result['failure_role']=='extractor'
    assert result['budget']['calls_used']==1 and providers['lead'].calls==0


def test_full_repeated_benchmark_does_not_admit_mock_or_leak_gold(tmp_path):
    tasks=task_data();lookup={t['id']:t for t in tasks};seen=[]
    def reply(req):
        payload=json.loads(req.user);seen.append(payload)
        assert set(payload)=={'id','category','question','evidence','choices'}
        t=lookup[payload['id']]
        return json.dumps({'answer':t['expected'],'abstain':t['expected']=='ABSTAIN','evidence_quote':t['gold_quote']})
    ps=profiles();p=DeterministicMockProvider(reply,clock=clock)
    trial=ModelTrial(tmp_path,'benchmark',ps,providers={'extractor':p},clock=clock,max_calls=100,max_tokens=2000000)
    report=trial.benchmark(tasks,'extractor',limit=50,repeats=2)
    assert len(seen)==100 and report['accuracy']==1 and report['full_fifty_task_repeated_benchmark']
    assert report['repeated_answer_stable'] and report['critical_errors']==0
    assert not report['quality_gate_passed'] and not report['adoption_allowed']
    again=trial.benchmark(tasks,'extractor',limit=50,repeats=2)
    assert p.calls==100 and again['budget']['calls_used']==100


def test_benchmark_sends_public_choices_as_enum_without_gold(tmp_path):
    ps=profiles();seen=[];task=task_data()[0]
    def reply(req):
        seen.append(req.response_schema)
        return json.dumps({'answer':'USE','evidence_quote':task['gold_quote'],'abstain':False})
    provider=DeterministicMockProvider(reply,clock=clock)
    trial=ModelTrial(tmp_path,'choice-schema',ps,providers={'extractor':provider},clock=clock)
    result=trial.benchmark(task_data(),'extractor',limit=1)
    assert seen[0]['properties']['answer']['enum']==task['choices']
    assert result['rows'][0]['valid'] and not result['rows'][0]['correct']


def test_concurrent_trials_are_serialized_and_crash_lock_is_released(tmp_path):
    with trial_lock(tmp_path):
        with pytest.raises(sqlite3.OperationalError):
            with trial_lock(tmp_path):pass
    with trial_lock(tmp_path):pass


def test_budget_denial_stops_benchmark_without_provider_calls(tmp_path):
    ps=profiles();provider=DeterministicMockProvider(lambda request:'{}',clock=clock)
    trial=ModelTrial(tmp_path,'exhausted',ps,providers={'extractor':provider},clock=clock,max_calls=0)
    result=trial.benchmark(task_data(),'extractor',limit=3,repeats=2)
    assert result['status']=='BENCHMARK_STOPPED' and result['unrun_calls']==5
    assert result['budget']['calls_used']==0 and provider.calls==0
    assert not result['full_fifty_task_repeated_benchmark'] and not result['quality_gate_passed']


def test_provider_failure_is_recorded_once_not_retried(tmp_path):
    ps=profiles();seen=[]
    def fail(req):
        seen.append(req);raise ProviderError('authentication unavailable')
    trial=ModelTrial(tmp_path,'provider-error',ps,providers={'extractor':DeterministicMockProvider(fail,clock=clock)},clock=clock)
    report=trial.benchmark(task_data(),'extractor',limit=3)
    assert len(seen)==1 and report['status']=='BENCHMARK_STOPPED' and report['unrun_calls']==2
    assert report['budget']['calls_used']==1


def test_cli_plan_is_offline_and_inference_requires_explicit_flag(capsys):
    spec=importlib.util.spec_from_file_location('strategy_cli',ROOT/'scripts/try_strategy_models.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    assert cli.main(['plan'])==0
    report=json.loads(capsys.readouterr().out)
    assert report['status']=='PREPARED_NO_MODEL_CALLS' and report['benchmark_tasks']==50
    assert not report['historical_routing_enabled']
    with pytest.raises(SystemExit) as exc:
        cli.main(['workflow'])
    assert exc.value.code==2
