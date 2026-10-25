"""A bounded year calendar never implies historical model or production eligibility."""
import datetime as dt
import json
from pathlib import Path
import pytest
from pod_platform.application.year_trial import calendar_ticks,CappedResearchProvider,profile_daily_year
from pod_platform.agents.provider import DeterministicMockProvider,LLMRequest,GenerationParams,ProviderError


def test_year_calendar_has_every_week_month_start_end_and_no_after_year():
    ticks=calendar_ticks(2025)
    dates={x.date() for x in ticks}
    assert ticks==sorted(set(ticks))
    assert all(x.year==2025 for x in ticks)
    for month in range(1,13):
        assert dt.date(2025,month,1) in dates
    assert dt.date(2025,12,31) in dates
    assert len([d for d in dates if d.weekday()==0])==52
    with pytest.raises(ValueError):calendar_ticks(2026)


def test_shared_cap_includes_failures_and_cannot_reset_on_reopen(tmp_path):
    def fail(_):raise ProviderError('offline fixture')
    underlying=DeterministicMockProvider(fail)
    provider=CappedResearchProvider(tmp_path,max_calls=2,provider=underlying)
    request=LLMRequest('system','{"untrusted_evidence":[]}',GenerationParams(max_tokens=20))
    for _ in range(2):
        with pytest.raises(ProviderError):provider.complete(request)
    assert provider.budget.calls_used==2 and underlying.calls==2
    reopened=CappedResearchProvider(tmp_path,max_calls=2,provider=underlying)
    with pytest.raises(Exception):reopened.complete(request)
    assert underlying.calls==2
    with pytest.raises(ValueError):CappedResearchProvider(tmp_path,max_calls=41,provider=underlying)


def test_schema_prompt_only_adds_public_schema_and_preserves_request_generation(tmp_path):
    seen=[]
    def respond(request):seen.append(request);return '{"ok":true}'
    provider=CappedResearchProvider(tmp_path,max_calls=1,provider=DeterministicMockProvider(respond))
    params=GenerationParams(max_tokens=77)
    provider.complete(LLMRequest('system','{"untrusted_evidence":[]}',params,response_schema={'type':'object'}))
    assert 'schema' in seen[0].system and seen[0].params==params
    assert provider.budget.calls_used==1


def test_real_daily_year_profile_preserves_missing_and_total_loss(tmp_path):
    p=tmp_path/'daily.csv'
    p.write_text('permno,date,daily_return\n1,2024-12-31,0.1\n1,2025-01-02,-1\n2,2025-01-02,\n1,2025-02-03,0.2\n')
    result=profile_daily_year(p,2025)
    assert result['2025-01']['rows']==2 and result['2025-01']['names']==2
    assert result['2025-01']['missing_returns']==1 and result['2025-01']['invalid_returns']==0
    assert result['2025-02']['trading_days']==1 and result['2025-03']['rows']==0


def test_private_file_or_outcome_payload_never_reaches_transport(tmp_path):
    underlying=DeterministicMockProvider(lambda request:'{}')
    provider=CappedResearchProvider(tmp_path,max_calls=1,provider=underlying)
    for value in [{'daily_return':[.1,.2]},{'untrusted_evidence':[{'kind':'source_card','text':'private'}]},{'untrusted_evidence':[],'path':'/home/user/data.csv'}]:
        with pytest.raises(ValueError):provider.complete(LLMRequest('system',json.dumps(value),GenerationParams()))
    assert underlying.calls==0 and provider.budget.calls_used==0


def test_native_groq_schema_closes_objects_and_preserves_reviewed_parameter_values():
    from pod_platform.pods.schema import ResearchProposal
    selector={'dataset_id':'public_panel','recipe_id':'reviewed_v1','method_id':'ridge','feature_set':'core','ridge_lambda':.05}
    schema=CappedResearchProvider.strict_schema(ResearchProposal.model_json_schema(),selector)
    variant=schema['$defs']['ExperimentVariant']['properties']['params']
    assert variant['additionalProperties'] is False
    assert set(variant['required'])==set(selector)
    assert variant['properties']['ridge_lambda']['enum']==[.05]
    assert set(schema['required'])==set(schema['properties'])


def test_compact_official_dates_are_normalized_before_month_aggregation(tmp_path):
    p=tmp_path/'daily.csv'
    p.write_text('permno,date,daily_return\n1,20250102,0.1\n1,20250203,-1\n')
    result=profile_daily_year(p,2025)
    assert result['2025-01']['rows']==1 and result['2025-02']['rows']==1
    assert result['2025-01']['first_date']=='2025-01-02'


def test_unreviewed_empty_parameter_object_does_not_send_invalid_required_array():
    from pod_platform.pods.schema import ResearchProposal
    schema=CappedResearchProvider.strict_schema(ResearchProposal.model_json_schema(),None)
    params=schema['$defs']['ExperimentVariant']['properties']['params']
    assert params['properties']=={} and params['additionalProperties'] is False
    assert 'required' not in params
