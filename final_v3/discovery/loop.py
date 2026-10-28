"""Expanding annual evidence, fixed gates, and auditable discovery sleeves."""

import hashlib
import json
import re

import numpy as np
import pandas as pd

import config
from sleeves import base
from . import agents, metrics
from .carbon import Carbon
from .sources import Sources


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return str(value) if np.isinf(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def dumps(value):
    return json.dumps(clean(value), sort_keys=True, allow_nan=False)


def validate(proposal, pod, sources, names, specs):
    required = {'name', 'pod', 'hypothesis', 'mechanism', 'expected_direction', 'inputs', 'cohort'}
    if not isinstance(proposal, dict) or set(proposal) != required:
        raise ValueError('Proposal must have exactly: ' + ', '.join(sorted(required)))
    # The desk assigns the pod; models often echo the charter's long name instead.
    proposal['pod'] = pod
    name = proposal['name']
    if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', name) or name in names:
        raise ValueError('Invalid or duplicate name')
    for field in ('hypothesis', 'mechanism', 'expected_direction'):
        if not isinstance(proposal[field], str) or not proposal[field].strip():
            raise ValueError('Empty ' + field)
    inputs = proposal['inputs']
    if not isinstance(inputs, list) or not 1 <= len(inputs) <= 3:
        raise ValueError('Expected 1-3 inputs')
    for item in inputs:
        if not isinstance(item, dict) or not {'source', 'feature', 'sign'} <= set(item) or set(item) - {'source', 'feature', 'sign', 'transform'}:
            raise ValueError('Input requires source, feature, sign and optional transform')
        if item['source'] not in sources.features or item['feature'] not in sources.features[item['source']]:
            raise ValueError('Unknown source/feature')
        if type(item['sign']) is not int or item['sign'] not in (-1, 1):
            raise ValueError('Sign must be integer +1 or -1')
        if item.get('transform', 'level') not in ('level', 'chg3', 'sector_rank'):
            raise ValueError('Unknown transform')
    signatures = [(i['source'], i['feature'], i.get('transform', 'level')) for i in inputs]
    if len(set(signatures)) != len(inputs):
        raise ValueError('Repeated input')
    cohort = proposal['cohort']
    if not isinstance(cohort, dict) or set(cohort) != {'size_groups', 'gics2'}:
        raise ValueError('Invalid cohort fields')
    for field, allowed in [('size_groups', agents.SIZES), ('gics2', agents.SECTORS)]:
        values = cohort[field]
        if not isinstance(values, list) or any(v not in allowed for v in values) or len(set(values)) != len(values):
            raise ValueError('Invalid cohort ' + field)
    if not cohort['size_groups']:
        raise ValueError('Empty size cohort')
    spec = dumps([sorted((*s, i['sign']) for s, i in zip(signatures, inputs)),
                  sorted(cohort['size_groups']), sorted(cohort['gics2'])])
    if spec in specs:
        raise ValueError('Renamed duplicate specification')
    return spec


def gate(measured, spread, peers):
    reasons, correlations = [], {}
    if not measured['S'] >= .60:
        reasons.append('S below .60 or unavailable')
    if not measured['shuffled_check']:
        reasons.append('shuffled control failed')
    if not measured['spread_t'] >= 2:
        reasons.append('spread t below 2 or unavailable')
    for name, other in peers.items():
        corr = spread.corr(other)
        correlations[name] = corr
        if not np.isfinite(corr) or abs(corr) >= .7:
            reasons.append('spread correlation fails against ' + name)
    if len(peers) >= 2:
        reasons.append('pod cap reached')
    return reasons, correlations


def insight_ids(value):
    """Collect insight identifiers without accepting unrelated catalogue IDs."""
    if isinstance(value, dict):
        own = {value['insight_id']} if type(value.get('insight_id')) is int else set()
        return own.union(*(insight_ids(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(insight_ids(v) for v in value))
    return set()


def replay_directory(default):
    """The brief-05 harness changes WORK_DIR after parsing its source directory."""
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--work-dir', type=Path)
    args, _ = parser.parse_known_args()
    return args.work_dir or default


def scout(provider, carbon, pod, charter, year):
    results = carbon.search(' '.join(charter['mechanism_families']).replace('_', ' '))
    raw = provider.call(agents.scout_prompt(pod, charter, results, year))
    requests = agents.parse(raw)
    if not isinstance(requests, list) or len(requests) > 3:
        raise ValueError('Scout must return at most three requests')
    rows = []
    for request in requests:
        if not isinstance(request, dict) or set(request) != {'insight_id', 'why', 'expected_direction', 'entities'}:
            raise ValueError('Invalid scout request')
        if type(request['insight_id']) is not int or request['insight_id'] not in insight_ids(results.get('insights', [])):
            raise ValueError('Insight ID is absent from the free search results')
        if not all(isinstance(request[k], str) and request[k].strip() for k in ('why', 'expected_direction')):
            raise ValueError('Scout rationale must be nonempty')
        entities = request['entities']
        if entities != 'ticker' and (not isinstance(entities, list) or not entities or any(
                not isinstance(e, dict) or set(e) != {'carc_id', 'representation'}
                or type(e['carc_id']) is not int or e['representation'] != 'ticker' for e in entities)):
            raise ValueError('Invalid entity scope')
        quote = carbon.quote(request['insight_id'], entities, f'{year-1}-12-31')
        rows.append(dict(year=year, pod=pod, **request, quote=dumps(quote),
                         price=quote.get('price'), status='shopping_list_only'))
    return rows, raw


def research_batch(provider, messages, pod, sources, names, specs, round_id, record, transcript):
    """Validate a complete three-proposal batch with one repair opportunity."""
    raw = provider.call(messages)
    for attempt in range(2):
        try:
            batch = agents.parse(raw)
            if not isinstance(batch, list) or len(batch) != 3:
                raise ValueError('Expected exactly three proposals')
            local_names, local_specs, batch_specs = names.copy(), specs.copy(), []
            for proposal in batch:
                spec = validate(proposal, pod, sources, local_names, local_specs)
                local_names.add(proposal['name'])
                local_specs.add(spec)
                batch_specs.append(spec)
            return batch, batch_specs, raw
        except (ValueError, TypeError, KeyError) as exc:
            record.write(dumps(dict(pod=pod, round=round_id, attempt=attempt,
                valid=False, raw_text=raw, error=str(exc), fake=bool(getattr(provider, 'fake', False)))) + '\n')
            transcript.append(f'{pod} round {round_id}: invalid batch: {exc}')
            if attempt == 0:
                raw = provider.call(messages + [dict(role='assistant', content=raw),
                    dict(role='user', content='Repair this validation error: ' + str(exc))])
    return [], [], raw


def evaluate_proposal(sources, proposal, year, round_id, peers):
    """Measure native evidence and apply the fixed sequential gate."""
    frame = sources.frame(proposal, '2015-01-31', f'{year-1}-11-30', labels=True)
    measured, spread = metrics.evaluate(frame, proposal, year)
    reasons, correlations = gate(measured, spread, peers)
    row = dict(name=proposal['name'], pod=proposal['pod'], round=round_id, **measured,
               admitted=not reasons, reason='; '.join(reasons) or 'all fixed gates passed',
               correlations=dumps(correlations))
    return row, spread


def run(work_dir, years=None, provider=None, sources=None, carbon=None):
    """Always replay requests through cache so offline incompleteness is visible."""
    years = config.TEST_YEARS if years is None else years
    provider = provider or agents.Provider()
    sources = sources or Sources()
    carbon = carbon or Carbon()
    summaries, requests = [], []
    fake = bool(getattr(provider, 'fake', False))
    for year in years:
        folder = work_dir / 'discovery' / str(year)
        folder.mkdir(parents=True, exist_ok=True)
        catalogue = sources.catalogue(f'{year-1}-11-30')
        (folder / 'catalogue.json').write_text(dumps(catalogue))
        transcript = [f'# Discovery {year}', f'Fake provider: {fake}.',
            f'Evidence targets 2015-02 through {year-1}-12; formations end {year-1}-11-30.',
            'Adaptive selection evidence, not an unbiased holdout. Fixed gates: S >= .60, shuffled control, '
            'spread NW(1) t >= 2, absolute within-pod spread correlation < .7, cap 2.',
            'Owned Carbon extracts are explicitly allowed by brief 06; one-month lag is assumed. '
            'Vintage revisions are not reconstructable. Catalogue scouting is current metadata, never a predictor.']
        admitted, evaluations, names, specs = [], [], set(), set()
        with (folder / 'proposals.jsonl').open('w') as record:
            for pod, charter in agents.charters().items():
                feedback, peers, proposed, valid = [], {}, 0, 0
                for round_id in range(1, config.DISCOVERY_ROUNDS + 1):
                    messages = agents.researcher_prompt(pod, charter, catalogue, feedback, year)
                    proposed += 3
                    batch, batch_specs, raw = research_batch(
                        provider, messages, pod, sources, names, specs, round_id, record, transcript)
                    if not batch:
                        feedback.append(dict(invalid_batch_error='Batch rejected after repair; see transcript.'))
                    for proposal, spec in zip(batch, batch_specs):
                        valid += 1
                        names.add(proposal['name'])
                        specs.add(spec)
                        record.write(dumps(dict(pod=pod, round=round_id, proposal=proposal, valid=True,
                            raw_text=raw, raw_model_text_sha256=hashlib.sha256(raw.encode()).hexdigest(), fake=fake)) + '\n')
                        row, spread = evaluate_proposal(sources, proposal, year, round_id, peers)
                        measured, accepted = row, row['admitted']
                        if accepted:
                            admitted.append(proposal)
                            peers[proposal['name']] = spread
                        evaluations.append(row)
                        feedback.append(dict(name=proposal['name'], inputs=proposal['inputs'], cohort=proposal['cohort'],
                            S=clean(round(measured['S'], 3)), spread_t=clean(round(measured['spread_t'], 2)),
                            admitted=accepted, reason=row['reason']))
                        transcript.append(f"{pod} round {round_id}: `{dumps(proposal)}`\n\nMeasured: `{dumps(row)}`")
                summaries.append(dict(year=year, pod=pod, proposed=proposed, valid=valid, admitted=len(peers)))
                # The scout only builds the shopping list: its failures are recorded and printed,
                # never allowed to stop the research that feeds the book.
                try:
                    shopping, raw = scout(provider, carbon, pod, charter, year)
                    requests.extend(shopping)
                    transcript.append(f'{pod} scout: {len(shopping)} shopping-list requests. Raw: {raw}')
                    if not shopping:
                        requests.append(dict(year=year, pod=pod, status='no_requests'))
                except (ValueError, TypeError, KeyError) as exc:
                    requests.append(dict(year=year, pod=pod, status='invalid_scout_response', error=str(exc)))
                    transcript.append(f'{pod} scout invalid: {exc}')
                except RuntimeError as exc:
                    print(f'Discovery {year} {pod} scout: FAILED ({exc}); shopping list incomplete', flush=True)
                    requests.append(dict(year=year, pod=pod, status='scout_request_failed', error=str(exc)))
                    transcript.append(f'{pod} scout failed: {exc}')
        pd.DataFrame(evaluations).assign(fake=fake).to_csv(folder / 'evaluations.csv', index=False)
        (folder / 'admitted.json').write_text(dumps(dict(year=year, fake=fake,
            validation_start='2015-02', validation_end=f'{year-1}-12', proposals=admitted)))
        (folder / 'transcript.md').write_text('\n\n'.join(transcript) + '\n')
        yearly = [r for r in summaries if r['year'] == year]
        print(f"Discovery {year}: {sum(r['proposed'] for r in yearly)} proposed, "
              f"{sum(r['valid'] for r in yearly)} valid, {len(admitted)} admitted: "
              + (', '.join('desk_' + p['name'] for p in admitted) or '(none)'), flush=True)
    pd.DataFrame(summaries).to_csv(work_dir / 'discovery_summary.csv', index=False)
    request_frame = pd.DataFrame(requests).assign(fake=fake)
    if 'entities' in request_frame:
        request_frame['entities'] = request_frame.entities.map(dumps)
    request_frame.to_csv(work_dir / 'data_requests.csv', index=False)
    return load_sleeves(work_dir, allow_fake=fake, sources=sources)


class DiscoverySleeve(base.Sleeve):
    """A saved proposal is active only in its admitted return years."""

    def __init__(self, name, proposals, sources=None):
        super().__init__()
        self.name = 'desk_' + name
        self.proposals = proposals
        self.sources = sources
        self.frames = {}

    def load(self, context):
        self.sources = self.sources or Sources(context.data_dir)

    def score(self, year, broad=False):
        if year not in self.proposals:
            return pd.DataFrame({'return_month': pd.Series(dtype='period[M]'),
                                 'permno': pd.Series(dtype='int64'), 'score': pd.Series(dtype='float64')})
        proposal = self.proposals[year]
        if year not in self.frames:
            end = min(pd.Timestamp(f'{year}-11-30'), config.LAST_TARGET - pd.offsets.MonthEnd(1))
            self.frames[year] = self.sources.frame(proposal, f'{year-1}-12-31', end)
        frame = self.frames[year]
        score = metrics.scores(frame, proposal, native_only=not broad)
        return base.formation_output(frame.assign(score=score), year)


def load_sleeves(work_dir, allow_fake=False, sources=None):
    proposals = {}
    for year in config.TEST_YEARS:
        path = work_dir / 'discovery' / str(year) / 'admitted.json'
        if not path.exists():
            continue
        saved = json.loads(path.read_text())
        if saved['fake'] and not allow_fake:
            raise ValueError('Fake discovery artifacts cannot enter the production book')
        if saved['year'] != year or saved['validation_end'] != f'{year-1}-12':
            raise ValueError('Invalid discovery evidence cutoff')
        for proposal in saved['proposals']:
            proposals.setdefault(proposal['name'], {})[year] = proposal
    shared = sources or (Sources() if proposals else None)
    return tuple(DiscoverySleeve(name, years, shared) for name, years in proposals.items())
