"""Host-owned, durable research stage budgets; no financial/admission authority.

Reservations precede data access and are never refunded after crashes. Evidence and
mechanism identities come from reviewed executable metadata, not model prose.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
from pathlib import Path
from pydantic import Field, field_validator

from ..contracts.base import StrictModel, canonical_digest, require_aware
from .jobs import open_sqlite

STAGES = ('idea_excerpt', 'component_probe', 'integrated_design', 'independent_check_shadow', 'trade')


class ComponentGrant(StrictModel):
    """Trusted host capability: ids select reviewed executable objects only."""
    grant_id: str = Field(pattern=r'^[a-z][a-z0-9_]{0,63}$')
    namespace: str
    dataset_id: str = Field(pattern=r'^[a-z][a-z0-9_]{0,63}$')
    recipe_id: str
    recipe_digest: str = Field(pattern=r'^[0-9a-f]{64}$')
    feature_names: list[str] = Field(min_length=1, max_length=3)
    reviewed_by: str = Field(min_length=1, max_length=100)
    reviewed_at: dt.datetime
    code_digest: str = Field(pattern=r'^[0-9a-f]{64}$')

    @field_validator('reviewed_at')
    @classmethod
    def _aware(cls, value):
        return require_aware(value, 'reviewed_at')


class HypothesisRequest(StrictModel):
    """An unverified idea can request an existing diagnostic without a paper."""
    grant_id: str = Field(pattern=r'^[a-z][a-z0-9_]{0,63}$')
    hypothesis: str = Field(min_length=20, max_length=2000)
    falsification: str = Field(min_length=20, max_length=1000)
    competing_explanation: str = Field(min_length=20, max_length=1000)
    unavailable_controls: list[str] = Field(default_factory=list, max_length=8)

    @field_validator('unavailable_controls')
    @classmethod
    def _controls(cls, values):
        if any(not 1 <= len(v.strip()) <= 600 for v in values):
            raise ValueError('control limitations must be 1..600 characters')
        return values


@dataclasses.dataclass(frozen=True)
class ProgressionPolicy:
    policy_id: str = 'bounded_research_progression_v1'
    probe_only: bool = False
    max_probes_per_pod: int = 3
    max_designs_per_pod: int = 4
    max_total_cases: int = 28
    max_probe_features: int = 3
    max_probe_months: int = 12
    max_probe_names: int = 100
    max_probe_rows: int = 5000

    def __post_init__(self):
        if not isinstance(self.probe_only, bool):
            raise ValueError('probe_only must be boolean')
        for name, value in dataclasses.asdict(self).items():
            if name not in ('policy_id', 'probe_only') and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError('progression limits must be positive integers')
        if self.max_probe_features > 3 or self.max_probe_months > 12 or self.max_probe_names > 100 or self.max_probe_rows > 5000:
            raise ValueError('progression may not expand the bounded diagnostic envelope')


_SCHEMA = '''
CREATE TABLE IF NOT EXISTS progression_events (
 key TEXT PRIMARY KEY, kind TEXT NOT NULL, namespace TEXT NOT NULL, payload TEXT NOT NULL, hash TEXT NOT NULL);
CREATE TRIGGER IF NOT EXISTS progression_no_update BEFORE UPDATE ON progression_events BEGIN SELECT RAISE(ABORT,'immutable progression'); END;
CREATE TRIGGER IF NOT EXISTS progression_no_delete BEFORE DELETE ON progression_events BEGIN SELECT RAISE(ABORT,'append-only progression'); END;
'''


class ResearchProgression:
    def __init__(self, path, *, code_digest, clock, policy=None):
        self.path, self.clock = Path(path), clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.policy = policy or ProgressionPolicy()
        with open_sqlite(self.path, immediate=True) as conn:
            conn.executescript(_SCHEMA)
            self._verify(conn)
            binding = {'policy': dataclasses.asdict(self.policy), 'code_digest': code_digest, 'stages': list(STAGES)}
            self._put(conn, 'binding', 'binding', 'host', binding)

    @staticmethod
    def _verify(conn):
        for key, kind, ns, payload, sha in conn.execute('SELECT key,kind,namespace,payload,hash FROM progression_events'):
            if canonical_digest([key, kind, ns, json.loads(payload)]) != sha:
                raise ValueError('progression integrity failure')

    @staticmethod
    def _put(conn, key, kind, ns, value):
        sha = canonical_digest([key, kind, ns, value])
        row = conn.execute('SELECT hash FROM progression_events WHERE key=?', (key,)).fetchone()
        if row:
            if row[0] != sha:
                raise ValueError('immutable progression policy/result differs: ' + key)
            return
        conn.execute('INSERT INTO progression_events VALUES (?,?,?,?,?)',
                     (key, kind, ns, json.dumps(value, sort_keys=True, allow_nan=False), sha))

    @staticmethod
    def _all(conn, kind):
        return [json.loads(row[0]) for row in conn.execute('SELECT payload FROM progression_events WHERE kind=?', (kind,))]

    def reserve(self, namespace, *, stage, fingerprint, mechanism_key, cases=1, lineage=None):
        if stage not in ('component_probe', 'integrated_design'):
            raise ValueError('only research probes/designs can be reserved; later stages require their existing strict authority')
        if stage == 'integrated_design' and self.policy.probe_only:
            raise ValueError('host policy permits component probes only')
        if isinstance(cases, bool) or not isinstance(cases, int) or cases < 1:
            raise ValueError('bounded positive case count required')
        attempt_id = 'rp_' + canonical_digest([namespace, stage, fingerprint])[:24]
        with open_sqlite(self.path, immediate=True) as conn:
            self._verify(conn)
            attempts, results = self._all(conn, 'reservation'), self._all(conn, 'result')
            prior = next((a for a in attempts if a['attempt_id'] == attempt_id), None)
            if prior:
                if prior['mechanism_key'] != mechanism_key or prior['cases'] != cases:
                    raise ValueError('progression fingerprint rebound')
                result = next((r for r in results if r['attempt_id'] == attempt_id), None)
                return {**prior, 'status': 'reused', 'result': result}
            if any(r['negative'] and r['mechanism_key'] == mechanism_key and r['namespace'] == namespace for r in results):
                return {'status': 'stopped', 'reason': 'prior_negative_component_same_mechanism', 'attempt_id': attempt_id}
            limit = self.policy.max_probes_per_pod if stage == 'component_probe' else self.policy.max_designs_per_pod
            if sum(a['namespace'] == namespace and a['stage'] == stage for a in attempts) >= limit:
                return {'status': 'budget_exhausted', 'reason': 'pod_stage_budget_exhausted', 'attempt_id': attempt_id}
            if sum(a['cases'] for a in attempts) + cases > self.policy.max_total_cases:
                return {'status': 'budget_exhausted', 'reason': 'total_case_budget_exhausted', 'attempt_id': attempt_id}
            value = {'attempt_id': attempt_id, 'namespace': namespace, 'stage': stage, 'fingerprint': fingerprint,
                     'mechanism_key': mechanism_key, 'cases': cases, 'lineage': lineage or {},
                     'reserved_at': require_aware(self.clock(), 'clock').isoformat()}
            self._put(conn, attempt_id, 'reservation', namespace, value)
            return {**value, 'status': 'reserved'}

    def complete(self, attempt_id, result, *, negative=False):
        with open_sqlite(self.path, immediate=True) as conn:
            self._verify(conn)
            attempt = next((a for a in self._all(conn, 'reservation') if a['attempt_id'] == attempt_id), None)
            if attempt is None:
                raise ValueError('unreserved research result')
            value = {'attempt_id': attempt_id, 'namespace': attempt['namespace'], 'mechanism_key': attempt['mechanism_key'],
                     'negative': bool(negative), 'result': result}
            self._put(conn, 'result:' + attempt_id, 'result', attempt['namespace'], value)
            return value

    def snapshot(self, namespace=None):
        with open_sqlite(self.path) as conn:
            self._verify(conn)
            attempts = self._all(conn, 'reservation')
            results = self._all(conn, 'result')
        selected = [a for a in attempts if namespace is None or a['namespace'] == namespace]
        return {'policy': dataclasses.asdict(self.policy), 'stages': list(STAGES), 'cases_consumed': sum(a['cases'] for a in selected),
                'total_cases_consumed': sum(a['cases'] for a in attempts), 'attempts': selected,
                'results': [r for r in results if namespace is None or r['namespace'] == namespace]}
