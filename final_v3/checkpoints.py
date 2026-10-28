"""Portable, content-verified model/forecast/portfolio checkpoints.

These explicit snapshots are separate from output/, which a fresh run clears.
Replaying archived forecasts or weights records their original provenance and
does not relabel the broad training cache as a match for changed source code.
"""

from datetime import datetime, timezone
import fnmatch
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import config
import timing

MANIFEST = 'checkpoint_manifest.json'
SCHEMA = 1
_REQUIRED = ('predictions.parquet', 'portfolio.parquet', 'optimizer_config.json',
             'optimizer_diagnostics.csv', 'report/holdings.csv', 'report/returns.csv')
_OPTIONAL = ('sleeves.parquet', 'graph_forecasts.parquet', 'timing_manifest.json',
             'run_provenance.json', 'prediction_checkpoint_info.json',
             'replay_provenance.json', 'filing_events/model_recovery_receipt.json',
             'report/performance.json', 'report/compliance.json', 'report/part2_status.json',
             'run_logs/run_summary.json', 'run_logs/environment_packages.json',
             'run_logs/resume_provenance.json', 'run_logs/optimizer_fix.patch',
             'run_logs/preflight.json', 'run_logs/RUN_NOTES.md',
             'micro_sleeve_provenance.json', 'gap_micro_best/selection.json',
             'gap_micro_best/analyst_forward.parquet',
             'gap_micro_best/xs_predictions.parquet', 'gap_micro_best/pooled_predictions.parquet',
             'valuation_training_receipt.json', 'rerun_receipt.json',
             'report/cash_allocations.csv', 'report/cap_attribution_monthly.csv',
             'cash5_validation.json', 'turnover30_validation.json', 'report/annual_cash_exposure_turnover.csv',
             'annual400_validation.json','report/calendar_year_turnover.csv',
             'research_audit/updated_results.json', 'research_audit/annual_returns_sharpe.csv',
             'research_audit/cap_breakdown.csv', 'research_audit/cap_annual_contributions.csv')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def input_paths(data_dir, external_file, reporting_file):
    data_dir = Path(data_dir)
    paths = {f'data/{name}': data_dir / name for name in (
        config.PANEL_FILE, config.FACTOR_LIST_FILE, config.FILINGS_FILE,
        'dsf_mini_from_2014.csv')}
    if config.USE_IBES:
        paths['external_data/external.parquet'] = Path(external_file)
    paths['reporting'] = Path(reporting_file)
    return paths


def input_receipts(data_dir, external_file, reporting_file):
    memo, receipts = {}, {}
    for name, path in input_paths(data_dir, external_file, reporting_file).items():
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f'Missing checkpoint input: {path}')
        if path not in memo:
            memo[path] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
        receipts[name] = dict(memo[path])
    return receipts


def capture_predictions(work_dir, data_dir, external_file, reporting_file):
    """Bind a newly generated prediction file to its input contents and mode."""
    work_dir = Path(work_dir)
    info = {'created_utc': datetime.now(timezone.utc).isoformat(),
            'discovery_enabled': config.DISCOVERY, 'ibes_enabled': config.USE_IBES,
            'nav': config.NAV, 'test_years': list(config.TEST_YEARS),
            'last_target': str(config.LAST_TARGET),
            'inputs': input_receipts(data_dir, external_file, reporting_file),
            'predictions_sha256': digest(work_dir / 'predictions.parquet')}
    (work_dir / 'prediction_checkpoint_info.json').write_text(json.dumps(info, indent=2)+'\n')
    return info


def require_separate_paths(work_dir, checkpoint_dir):
    """Never let output clearing remove the checkpoint or its parent."""
    work, checkpoint = Path(work_dir).resolve(), Path(checkpoint_dir).resolve()
    if work.is_relative_to(checkpoint) or checkpoint.is_relative_to(work):
        raise ValueError('Checkpoint and output directories must not overlap')


def _bound_inputs(work, data_dir, external_file, reporting_file):
    receipts = input_receipts(data_dir, external_file, reporting_file)
    info_path = work / 'prediction_checkpoint_info.json'
    if info_path.exists():
        info = json.loads(info_path.read_text())
        if receipts != info['inputs'] or digest(work/'predictions.parquet') != info['predictions_sha256']:
            raise ValueError('Saved forecasts do not match their recorded inputs')
    else:
        # The first RunPod run has a per-file content manifest, not the new
        # production prediction receipt. Accept its verified original hashes.
        legacy_path = work / 'run_provenance.json'
        if not legacy_path.exists():
            raise ValueError('No content provenance for existing forecasts; rebuild before exporting')
        legacy = json.loads(legacy_path.read_text())
        original = {item['path']:item for item in legacy['files']}
        for name, receipt in receipts.items():
            source = 'external_data/external.parquet' if name == 'reporting' else name
            expected = original.get(source)
            if expected is None or receipt['sha256'] != expected['sha256']:
                raise ValueError(f'Original forecast input differs: {name}')
        resume_path = work / 'run_logs/resume_provenance.json'
        if resume_path.exists():
            expected_prediction = json.loads(resume_path.read_text())['predictions_sha256']
            if digest(work/'predictions.parquet') != expected_prediction:
                raise ValueError('Original forecast file differs from the continuation receipt')
    return receipts


def export(work_dir, destination, data_dir, external_file, reporting_file):
    """Archive weights, historical forecasts and allocations outside output/."""
    work, dest = Path(work_dir).resolve(), Path(destination).resolve()
    require_separate_paths(work, dest)
    missing = [name for name in _REQUIRED if not (work/name).is_file()]
    if missing:
        raise ValueError('Incomplete completed run: '+', '.join(missing))
    if dest.exists():
        raise ValueError(f'Checkpoint already exists; choose a new name: {dest}')
    receipts = _bound_inputs(work, data_dir, external_file, reporting_file)
    performance = json.loads((work/'report/performance.json').read_text())
    settings = json.loads((work/'optimizer_config.json').read_text())
    if bool(config.DISCOVERY) != bool(performance['discovery_enabled']):
        raise ValueError('Export mode differs from the completed run discovery setting')
    if bool(config.USE_IBES) != ('conf_lm_gated' not in performance.get('excluded_sleeves', [])):
        raise ValueError('Export mode differs from the completed run I/B/E/S setting')
    pred = pd.read_parquet(work/'predictions.parquet')
    timing.validate_predictions(pred)
    dest.mkdir(parents=True)
    names = [*_REQUIRED, *[n for n in _OPTIONAL if (work/n).is_file()]]
    names += [str(p.relative_to(work)).replace('\\','/') for pattern in (
        'graph/runs/*.pt', 'xgb_rank/model_*_seed*.json', 'filing_events/model_*_seed*.json')
        for p in sorted(work.glob(pattern))]
    files = {}
    for name in dict.fromkeys(names):
        source, target = work/name, dest/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files[name] = {'bytes':target.stat().st_size, 'sha256':digest(target)}
    portfolio = pd.read_parquet(work/'portfolio.parquet')
    weights = portfolio[['return_month','permno','ticker','company_name','target_weight']].copy()
    weights['return_month'] = pd.to_datetime(weights.return_month).dt.to_period('M').astype(str)
    weights = weights.rename(columns={'target_weight':'weight_fraction'})
    weights['weight_percent'] = weights.weight_fraction*100
    weights.to_csv(dest/'portfolio_weights.csv', index=False)
    weights.loc[weights.return_month.eq(weights.return_month.max())].to_csv(dest/'last_month_weights.csv', index=False)
    for name in ('portfolio_weights.csv','last_month_weights.csv'):
        files[name] = {'bytes':(dest/name).stat().st_size,'sha256':digest(dest/name)}
    for name in ('graph/model.py','graph/train.py','graph/dataset.py',
                 'sleeves/xgb_rank/model.py','sleeves/filing_events/model.py','config.py',
                 'sleeves/gap_micro_best/valuation.py','sleeves/gap_micro_best/sleeve.py',
                 'sleeves/runway_micro.py','combiner.py','optimizer.py','deployment.py','turnover.py',
                 'evaluate.py','official.py','risk.py','main.py','report.py','part2.py'):
        target = dest/'model_code'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config.FINAL/name, target)
        rel = target.relative_to(dest).as_posix()
        files[rel] = {'bytes':target.stat().st_size,'sha256':digest(target)}
    manifest = {'schema_version':SCHEMA, 'created_utc':datetime.now(timezone.utc).isoformat(),
                'period':performance['period'], 'nav':settings['nav'],
                'discovery_enabled':performance['discovery_enabled'],
                'ibes_enabled':not ('conf_lm_gated' in performance.get('excluded_sleeves',[])),
                'inputs':receipts, 'files':files,
                'neural_checkpoints':sum(n.startswith('graph/runs/') and n.endswith('.pt') for n in files),
                'xgboost_characteristic_models':sum(fnmatch.fnmatch(n,'xgb_rank/model_*_seed*.json') for n in files),
                'xgboost_filing_models':sum(fnmatch.fnmatch(n,'filing_events/model_*_seed*.json') for n in files),
                'holdings_rows':len(weights), 'holding_months':int(weights.return_month.nunique()),
                'weight_units':'weight_fraction is NAV fraction; weight_percent is percent of NAV',
                'scope':'Historical forecasts/allocations for the recorded period. New dates still require valid new features and appropriate walk-forward models.',
                'submission_ready':performance.get('submission_ready',False)}
    (dest/MANIFEST).write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def metadata(directory):
    path = Path(directory).resolve()/MANIFEST
    saved = json.loads(path.read_text())
    if saved.get('schema_version') != SCHEMA:
        raise ValueError('Unsupported checkpoint schema')
    return saved


def load(directory, data_dir, external_file, reporting_file):
    """Verify archived content and identical inputs before loading forecasts."""
    directory = Path(directory).resolve()
    saved = metadata(directory)
    if config.DISCOVERY != saved['discovery_enabled'] or config.USE_IBES != saved['ibes_enabled']:
        raise ValueError('Requested discovery/I/B/E/S mode differs from the archived forecasts')
    if float(config.NAV) != float(saved['nav']):
        raise ValueError('Checkpoint replay requires the recorded endowed NAV')
    expected_period = f'{config.TEST_YEARS[0]}-01..{config.LAST_TARGET:%Y-%m}'
    if saved['period'] != expected_period:
        raise ValueError('Checkpoint period differs from the configured evaluation window')
    for name in _REQUIRED:
        if name not in saved['files']:
            raise ValueError('Checkpoint is missing its required payload: '+name)
    for name, receipt in saved['files'].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError('Invalid checkpoint path: '+name)
        if path.stat().st_size != receipt['bytes'] or digest(path) != receipt['sha256']:
            raise ValueError('Checkpoint content checksum mismatch: '+name)
    for key, pattern in (('neural_checkpoints','graph/runs/*.pt'),
                         ('xgboost_characteristic_models','xgb_rank/model_*_seed*.json'),
                         ('xgboost_filing_models','filing_events/model_*_seed*.json')):
        if saved[key] != sum(fnmatch.fnmatch(name,pattern) for name in saved['files']):
            raise ValueError('Checkpoint model count differs from its payloads: '+key)
    if input_receipts(data_dir, external_file, reporting_file) != saved['inputs']:
        raise ValueError('Checkpoint input contents differ; generate new forecasts for changed data')
    predictions = pd.read_parquet(directory/'predictions.parquet')
    timing.validate_predictions(predictions)
    portfolio = pd.read_parquet(directory/'portfolio.parquet')
    needed = {'return_month','permno','target_weight','ticker','company_name','sector'}
    if not needed.issubset(portfolio.columns) or portfolio.duplicated(['return_month','permno']).any():
        raise ValueError('Invalid archived portfolio schema or duplicate holdings')
    if not np.isfinite(pd.to_numeric(portfolio.target_weight, errors='coerce')).all():
        raise ValueError('Archived weights contain nonfinite values')
    return saved, predictions, portfolio


def record_replay(work_dir, directory, saved, mode):
    receipt = {'mode':mode, 'checkpoint':str(Path(directory).resolve()),
               'checkpoint_manifest_sha256':digest(Path(directory)/MANIFEST),
               'original_period':saved['period'], 'verified_content_and_inputs':True,
               'models_retrained':False,'created_utc':datetime.now(timezone.utc).isoformat()}
    (Path(work_dir)/'replay_provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')


def restore_forecast_assets(work_dir, directory, saved):
    """Preserve verified model weights and score caches during a replay."""
    patterns = ('graph/runs/*.pt', 'xgb_rank/model_*_seed*.json',
                'filing_events/model_*_seed*.json', 'gap_micro_best/*',
                'sleeves.parquet', 'graph_forecasts.parquet', 'valuation_training_receipt.json')
    for name in saved['files']:
        if not any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
            continue
        source, target = Path(directory)/name, Path(work_dir)/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
