"""Isolated SDK adapter in the installed Carbon virtual environment."""

import json
import os
from pathlib import Path
import sys


def forbidden(*args, **kwargs):
    raise PermissionError('Carbon purchases prohibited; budget is zero')


def run(payload):
    """Allow only catalogue reads and the SDK's free quote method."""
    from carbonarc import CarbonArcClient
    from carbonarc.explorer import ExplorerAPIClient

    ExplorerAPIClient.buy_frameworks = forbidden
    root = Path(__file__).resolve().parents[2]
    for line in (root / 'local/carbonarc-api.env').read_text().splitlines():
        line = line.strip().removeprefix('export ').strip()
        if line.startswith('CARBONARC_API_TOKEN='):
            os.environ.setdefault('CARBON_API_KEY', line.split('=', 1)[1].strip().strip('\"\''))
    client = CarbonArcClient(token=os.environ['CARBON_API_KEY'])
    ex = client.explorer
    ex.buy_frameworks = forbidden
    if payload['operation'] == 'search':
        # Dataset metadata is free and searched locally; insight search is server-side.
        import importlib.util
        spec = importlib.util.spec_from_file_location('carbon_rest', root / 'carbon/client.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        library = module.CarbonClient().get('/v2/library/data')
        words = payload['query'].lower().split()
        datasets = [{k: d[k] for k in ('dataset_id', 'dataset_name', 'description') if k in d}
                    for d in library.get('datasources', [])
                    if any(w in json.dumps(d).lower() for w in words)][:8]
        return dict(insights=client.ontology.get_insights(search=payload['query'], size=12),
                    datasets=datasets)
    if payload['operation'] != 'quote':
        forbidden()
    filters = dict(date_resolution='month',
                   date_range=dict(start_date='2015-01-01', end_date=payload['end']))
    info = client.ontology.get_insight_information(payload['insight_id'])
    if payload['insight_id'] == 491 or 'event' in json.dumps(info).lower():
        filters.update(location_resolution='ww', filters={
            'ticker_exclusively_mentioned': ['Exclusive Mention', 'Non-Exclusive Mention']})
    framework = ex.build_framework(entities=payload['entities'], insight=payload['insight_id'], filters=filters)
    return ex.check_framework_quote(framework)


if __name__ == '__main__':
    try:
        print(json.dumps(run(json.load(sys.stdin))))
    except Exception as exc:
        # Never print SDK bodies, headers, tokens or arbitrary exception strings.
        print(json.dumps(dict(error=type(exc).__name__)))
        sys.exit(1)
