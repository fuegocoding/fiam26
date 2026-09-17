import json
import pandas as pd
import pytest
from src.time_series_book_audit import run


@pytest.mark.parametrize('corruption',[.005,float('nan')])
def test_independent_saved_book_audit_detects_corrupted_cost(tmp_path,corruption):
    folder=tmp_path/'time_series_cost25'
    (folder/'results').mkdir(parents=True);(folder/'books').mkdir()
    dates=pd.date_range('2021-01-31','2026-08-31',freq='ME')
    pd.DataFrame({'permno':[1],'eom':[pd.Timestamp('2020-12-31')],
        'ret_exc_lead1m':[.1],'eligible':[True]}).to_parquet(folder/'panel.parquet')
    b=pd.DataFrame({'date':dates,'cost':0.,'turnover':0.,'gross_excess':0.,'net_excess':0.,
        'n_long':0,'n_short':0,'gross_exposure':0.,'net_exposure':0.,'cash':.001,'missing_held_returns':0})
    path=folder/'books/01_large_mega_raw.csv';b.to_csv(path,index=False)
    pd.DataFrame({'date':pd.Series(dtype='datetime64[ns]'),'permno':pd.Series(dtype='int64'),
        'weight':pd.Series(dtype='float32')}).to_parquet(folder/'books/01_large_mega_raw_holdings.parquet')
    (folder/'results/01_large_mega_raw.json').write_text(json.dumps({'experiment':'01'}))
    run(tmp_path)
    assert json.loads((tmp_path/'time_series_batch/book_audit.json').read_text())['passed']
    b.loc[4,'cost']=corruption;b.to_csv(path,index=False)
    with pytest.raises(RuntimeError,match='reconciliation failed'):
        run(tmp_path)
    result=json.loads((tmp_path/'time_series_batch/book_audit.json').read_text())
    assert not result['passed']
    if pd.isna(corruption):
        assert result['errors'][0]['error']=='Nonfinite saved accounting field.'
    else:
        assert result['errors'][0]['maximum_error']==.005


def test_partial_batch_report_can_render_before_baseline_books_exist(tmp_path):
    from src.time_series_overnight import summarize_batch
    row={'experiment':'B1','name':'baseline','universe':'large_mega','mode':'raw',
         'information_ratio':float('nan'),'missing_return_months':0,'position_compliant_share':0.,
         'beta_price_proxy':0.,'timesfm_cache_diagnostic':False,'active_months':0,'annual_net_excess':0.}
    for cost in [10,25,50]:
        folder=tmp_path/f'time_series_cost{cost}';folder.mkdir()
        pd.DataFrame([row]).to_csv(folder/'leaderboard.csv',index=False)
    audit={'target':'return','test_rank_ic':0.,'r2_oos_zero':0.}
    for model in ['29','30','31','32']:
        (tmp_path/f'time_series_cost25/model_{model}_audit.json').write_text(json.dumps([audit]))
    extra=tmp_path/'time_series_followups';extra.mkdir()
    pd.DataFrame([{**row,'experiment':'37','cost_bp':50}]).to_csv(extra/'leaderboard.csv',index=False)
    for model in ['37','38','39','40']:
        (extra/f'model_{model}_audit.json').write_text(json.dumps([{
            'target':'error','test_rank_ic':0.,'baseline_rows':100,'matched_test_mse':.8,'baseline_mse':1.}]))
    report=summarize_batch(tmp_path)
    assert 'Completed 4 experiment' in report.read_text()
    assert not (tmp_path/'time_series_batch/all_cost_paired_comparisons.csv').exists()
