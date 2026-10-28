"""Freeze pre-OOS agent recipes before running the competition evaluation."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import date

from .configuration import config_path


def frozen_training_config(config):
    """Load actual training response receipts; never synthesize strategy recipes."""
    result = copy.deepcopy(config)
    policy = result['research_campaign']
    source = policy.get('frozen_training_campaign')
    if not source:
        return result
    path = config_path(source)
    if not path.is_file():
        raise ValueError(f'Frozen pre-2021 training campaign is missing: {path}. Run the training configuration first.')
    report = json.loads(path.read_text(encoding='utf-8'))
    if 'report' in report:
        path = config_path(report['report'])
        report = json.loads(path.read_text(encoding='utf-8'))
    source_config = json.loads((path.parent/'config.json').read_text(encoding='utf-8'))
    end = source_config['training_lab']['end']
    if (date.fromisoformat(end) >= date(2021,1,1)
        or date.fromisoformat(report['data']['development_end']) >= date(2021,1,1)):
        raise ValueError('Frozen training must end before 2021; OOS-selected recipes cannot be imported')
    if (report.get('evaluation_is_in_sample') is not False
        or report.get('portfolio_evidence_scope') != 'prior_settled'
        or not report.get('successful_strategies')):
        raise ValueError('Frozen training needs evaluated agent strategies and no retrospective selection')
    ledger = json.loads((path.parent/'attempt_ledger.json').read_text(encoding='utf-8'))
    hashes = {(r.get('model_provenance') or {}).get('response_hash') for r in ledger if r.get('status') == 'evaluated'} - {None}
    proposals = []
    for receipt in sorted(path.parent.glob('*_response.json')):
        row = json.loads(receipt.read_text(encoding='utf-8'))
        if row.get('response_hash') not in hashes or not all(row.get(k) for k in ('text','provider','model','request_hash')):
            continue
        text = row['text'].strip()
        if text.startswith('```') and text.endswith('```'):
            text = text.split('\n',1)[1].rsplit('\n',1)[0]
        if json.loads(text).get('strategies'):
            proposals.append(str(receipt))
    if not proposals:
        raise ValueError('No evaluated, provenance-linked agent responses in the frozen training campaign')
    policy.update(research_pods=[], recorded_code_proposals=proposals,
                  portfolio_evidence_scope='prior_settled', alt_discovery=[])
    policy.pop('research_resume',None)
    policy.pop('research_continuations',None)
    policy.pop('fixed_strategy_ids',None)
    policy['frozen_training_provenance'] = {
        'report':str(path),'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'training_end':end,'policy':'Freeze agent recipes before OOS; numerical forecasts and funding use settled past observations'}
    result['models']['online'] = False
    result['application']['online_papers'] = False
    result['carbon'].update(allow_remote=False,requests=[])
    return result
