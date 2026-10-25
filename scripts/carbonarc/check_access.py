#!/usr/bin/env python3
"""Read-only check of Carbon Arc API authentication and Block entitlements."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path

from download import ROOT, CATALOG_OUTPUT, atomic_json


ACTIVE = {'APPROVED', 'EVALUATION_ACTIVE', 'TRIAL_ACTIVE', 'CONTRACTED'}


def authorized_skus(response):
    """Only existing active approvals; never turn available/pending datasets into access."""
    entries = response.get('datasets')
    if not isinstance(entries, list):
        raise ValueError('Block discovery response has no datasets list')
    selected, seen = [], set()
    for dataset in entries:
        if dataset.get('platform_access_blocked') is True:
            continue
        for cut in dataset.get('cuts', []):
            for lag, status in cut.get('request_statuses', {}).items():
                if str(status).upper() not in ACTIVE:
                    continue
                key = (dataset['dataset_id'], cut.get('cut'), lag)
                if key in seen:
                    continue
                seen.add(key)
                selected.append({'dataset_id': key[0], 'cut': key[1], 'lag': lag,
                    'label': dataset.get('label'), 'access_status': status,
                    'platform_access_blocked': dataset.get('platform_access_blocked'),
                    'delivery': 'Polaris/Iceberg or S3; requires configured delivery credentials'})
    return selected


def error_status(exc):
    response = getattr(exc, 'response', None)
    return getattr(response, 'status_code', None) or getattr(exc, 'status_code', None)


def get_token(env_file):
    from dotenv import dotenv_values
    values = dotenv_values(env_file) if env_file.exists() else {}
    for name in ('CARBONARC_API_TOKEN', 'API_AUTH_TOKEN', 'CARBONARC_TOKEN'):
        value = os.environ.get(name) or values.get(name)
        if value and value.strip():
            return value.strip()
    return None


def probe(client):
    report = {'checked_at': dt.datetime.now(dt.timezone.utc).isoformat(),
              'scope': 'All accessible datasets; no company filter', 'read_only': True,
              'purchases_or_access_requests_submitted': 0, 'raw_data_downloaded': False}
    try:
        library = client.data.get_datasets()
        report['library_api_readable'] = True
        report['library_first_page_count'] = len(library.get('datasources', []))
        report['library_total_pages'] = library.get('total_pages')
    except Exception as exc:
        report.update(state='platform_api_error', error_type=type(exc).__name__, http_status=error_status(exc))
        return report, None
    try:
        block = client.block.list_datasets()
        skus = authorized_skus(block)
        report.update(state='bulk_access_found' if skus else 'no_active_bulk_entitlements',
                      block_catalog_datasets=len(block['datasets']), approved_skus=skus)
        return report, block
    except Exception as exc:
        report.update(state='bulk_access_unverified_or_denied', error_type=type(exc).__name__, http_status=error_status(exc))
        response = getattr(exc, 'response', None)
        try:
            detail = response.json().get('detail') if response is not None else None
        except (ValueError, AttributeError):
            detail = None
        if detail == 'Block Data is available on Enterprise plans only.':
            report.update(state='block_enterprise_plan_required', block_detail=detail)
        return report, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, default=ROOT / 'local/carbonarc-api.env')
    parser.add_argument('--output', type=Path, default=CATALOG_OUTPUT)
    args = parser.parse_args()
    token = get_token(args.env_file)
    if not token:
        report = {'state': 'api_token_needed', 'read_only': True, 'raw_data_downloaded': False,
                  'profile_url': 'https://app.carbonarc.ai/my/profile', 'env_file': str(args.env_file),
                  'variable': 'CARBONARC_API_TOKEN', 'mcp_login_is_separate': True}
        atomic_json(args.output / 'bulk_access_status.json', report)
        print(f'API token needed. Save CARBONARC_API_TOKEN in {args.env_file}, then rerun this command.')
        return 2
    from carbonarc import CarbonArcClient
    # The SDK's default .co admin host redirects to .ai and loses Authorization.
    report, catalog = probe(CarbonArcClient(token=token, cams_host='https://app.carbonarc.ai'))
    atomic_json(args.output / 'bulk_access_status.json', report)
    if catalog is not None:
        atomic_json(args.output / 'block_catalog.json', catalog)
    # No token, headers, server exception messages, or credential material in logs.
    print(json.dumps(report, indent=2))
    return 0 if report.get('state') == 'bulk_access_found' else 1


if __name__ == '__main__':
    raise SystemExit(main())
