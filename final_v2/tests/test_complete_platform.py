"""Source-to-trade acceptance: exact validated publications drive actual fills/NAV."""
import csv
import json
from pathlib import Path
import pytest

from pod_platform.application.system_workflow import CompletePlatformWorkflow


def read(root,name):
    return json.loads((root/name).read_text())


@pytest.fixture(scope='module')
def complete(tmp_path_factory):
    root=tmp_path_factory.mktemp('complete')
    workflow=CompletePlatformWorkflow(root/'state',root/'out')
    summary=workflow.run_demo()
    return root,workflow,summary


def test_full_four_pod_chain_drives_actual_coordinated_fills_and_accounting(complete):
    root,workflow,summary=complete
    out=root/'out'
    assert summary['publication_count']==4 and summary['book']['periods']==2
    assert summary['feedback_items']==2
    assert not summary['certificates_authorize_competition_book'] and not summary['llm_skill_evidence']
    manifest=read(out,'research_manifest.json')
    assert len(manifest['research']['outcomes'])==4
    admitted={p['candidate_id'] for p in manifest['publications']}
    fixture=read(out,'fixture/forecasts.json')
    assert {f['strategy_id'] for f in fixture} == admitted
    rejected={o.get('candidate_id') for o in manifest['research']['outcomes'] if o['outcome']!='ADMITTED'}
    assert not ({f['strategy_id'] for f in fixture}&rejected)
    receipts=read(out,'book/coordination_receipts.json')
    assert len(receipts)==2
    assert all(r['ex_ante_vol']<=.10+1e-10 for r in receipts)
    periods=read(out,'book/periods.json')
    fills=[trade for period in periods for trade in period['trades']]
    held=list(csv.DictReader((out/'book/holdings_by_period.csv').open()))
    assert fills and held and any(r['security_id']=='HEDGE' for r in fills)
    assert all(len([r for r in held if r['period']==p and r['security_id']=='HEDGE'])==1 for p in {r['period'] for r in held})
    for receipt in receipts:
        assert len(receipt['hedge_netting']['original_requests'])==3
        assert len(receipt['hedge_netting']['plan']['lines'])==1
        assert len(receipt['hedge_netting']['plan']['lines'][0]['requesters'])==3
        assert receipt['post_fill']['cancelled_remainders']
    assert float(summary['book']['ending_nav'])>0
    assert read(out,'dossiers.json')
    feedback=read(out,'feedback.json')
    assert all(f['scheduler_trigger_id'] and not f['automatic_retraining'] for f in feedback)
    freeze=read(out,'content_freeze.json')
    assert not freeze['prospective'] and 'book/periods.json' in freeze['sha256']


def test_completed_run_resumes_without_new_inference_checks_or_changed_book(complete):
    root,workflow,summary=complete
    before=(root/'out/book/holdings_by_period.csv').read_bytes()
    again=CompletePlatformWorkflow(root/'state',root/'out').run_demo()
    assert again==summary
    assert (root/'out/book/holdings_by_period.csv').read_bytes()==before


def test_tampered_publication_fails_before_new_orders(complete):
    root,workflow,_=complete
    pub=next((root/'state/research/forecasts').glob('*.json'))
    original=pub.read_bytes()
    try:
        pub.write_text('[]')
        with pytest.raises(ValueError,match='tamper'):
            CompletePlatformWorkflow(root/'state',root/'out').run_demo()
    finally:
        pub.write_bytes(original)


def test_incompatible_return_horizons_fail_before_allocation(complete,tmp_path):
    from pod_platform.orchestration.replay import ControlledReplay,load_fixture
    from pod_platform.contracts.errors import PodPlatformError
    import datetime as dt
    root,_,_=complete
    fixture=load_fixture(root/'out/fixture')
    first=fixture.forecasts[0]
    fixture.forecasts[0]=first.model_copy(update={'return_forecast':first.return_forecast.model_copy(
        update={'horizon_trading_days':42})}).sealed()
    replay=ControlledReplay(fixture,tmp_path)
    with pytest.raises(PodPlatformError,match='incompatible return horizons'):
        replay._active_forecasts(dt.datetime(2026,7,1,21,tzinfo=dt.timezone.utc))


def test_full_year_fixture_runs_twelve_actual_decisions_and_net_hedges(tmp_path):
    workflow=CompletePlatformWorkflow(tmp_path/'state',tmp_path/'out')
    summary=workflow.run_demo(year=2025)
    assert summary['book']['periods']==12 and summary['publication_count']==24
    receipts=read(tmp_path/'out','book/coordination_receipts.json')
    assert len(receipts)==12
    assert all(len(r['hedge_netting']['original_requests'])==3 and len(r['hedge_netting']['plan']['lines'])==1 for r in receipts)
    periods=read(tmp_path/'out','book/periods.json')
    assert periods[0]['period'].startswith('2025-01') and periods[-1]['period'].startswith('2025-12')
    assert all(p['actual_fill_risk_decision']=='PASS' for p in periods)
