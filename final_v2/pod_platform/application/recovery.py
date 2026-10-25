"""Idempotent settlement of interrupted deterministic research reservations.

Persistent registrations/trials/accesses stay append-only across stage attempts.
An unfinished deterministic comparison may be recomputed with identical bound
inputs and settle its existing reservation; nothing deletes or resets a trial.
"""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from ..research.registration import ResearchLedger,RegisteredExperiment,TrialRecord
from ..research.evaluation import RegisteredEvaluator,ComparisonPlan
from .common import canonical_json,digest

class DurableResearchLedger(ResearchLedger):
    def __init__(self,registry,path):
        super().__init__(registry)
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path)
        self.db.execute('CREATE TABLE IF NOT EXISTS entries (key TEXT PRIMARY KEY,kind TEXT NOT NULL,payload TEXT NOT NULL)')
        self.db.executescript("CREATE TRIGGER IF NOT EXISTS entries_no_update BEFORE UPDATE ON entries BEGIN SELECT RAISE(ABORT,'immutable ledger'); END; CREATE TRIGGER IF NOT EXISTS entries_no_delete BEFORE DELETE ON entries BEGIN SELECT RAISE(ABORT,'immutable ledger'); END;")
        for key,kind,payload in self.db.execute('SELECT key,kind,payload FROM entries ORDER BY rowid'):
            self._restore(key,kind,json.loads(payload))
        for registration in list(self._registrations.values()):
            self.register(registration)
    def _restore(self,key,kind,payload):
        if kind=='registration':self._registrations[payload['registration_id']]=RegisteredExperiment.model_validate(payload)
        elif kind=='trial':
            trial=TrialRecord.model_validate(payload);self._trials.append(trial)
            try:self._registry.count_trial(key,trial.kind,trial.spec_hash)
            except sqlite3.IntegrityError:pass
        elif kind=='access':
            self._outcome_access.append(payload)
            try:self._registry.record_evaluation_access(**payload,authorized=True)
            except sqlite3.IntegrityError:pass
    def _append(self,kind,payload):
        key=f'durable_{kind}_{digest(payload)}'
        prior=self.db.execute('SELECT payload FROM entries WHERE key=?',(key,)).fetchone()
        if prior:return False
        with self.db:self.db.execute('INSERT INTO entries VALUES(?,?,?)',(key,kind,canonical_json(payload)))
        self._restore(key,kind,payload)
        return True
    def register(self,experiment):
        prior=self._registrations.get(experiment.registration_id)
        if prior and prior.spec_content_hash!=experiment.spec_content_hash:raise ValueError('registration is immutable')
        self._append('registration',experiment.model_dump(mode='json'))
        self.record_trial(TrialRecord(kind='registered_specification',spec_hash=experiment.spec_content_hash,
            detail=f'registered {experiment.registration_id}',counted_at=experiment.registered_at))
        return experiment.registration_id
    def record_trial(self,trial):self._append('trial',trial.model_dump(mode='json'))
    def reserve_trial(self,registration_id,trial):
        payload=trial.model_dump(mode='json');key=f'durable_trial_{digest(payload)}'
        if self.db.execute('SELECT 1 FROM entries WHERE key=?',(key,)).fetchone():return
        self.enforce_variant_budget(registration_id)
        self.record_trial(trial)
    def record_outcome_access(self,access_id,partition,actor,purpose,record_count,*,authorized=False):
        if partition in ('check','replay','prospective') and not authorized:raise PermissionError('protected access denied')
        # Same immutable evaluation request is one logical access across retries.
        access_id='durable_access_'+digest(dict(partition=partition,actor=actor,purpose=purpose,record_count=record_count))
        self._append('access',dict(access_id=access_id,partition=partition,actor=actor,purpose=purpose,record_count=record_count))
        return access_id
    def close(self):self.db.close()

class RecoverableRegisteredEvaluator(RegisteredEvaluator):
    def evaluate(self,experiment_id,variant_id,observations,**kwargs):
        prior=self.connection.execute('SELECT input_hash,result FROM comparison_trials WHERE experiment_id=? AND variant_id=?',
            (experiment_id,variant_id)).fetchone()
        if prior and prior[1] is None:
            # Run the reviewed evaluator, including all temporal/partition/source
            # checks, in scratch memory. Keep and settle the original charged row.
            scratch=RegisteredEvaluator(':memory:')
            try:
                plan=self.connection.execute('SELECT payload FROM comparison_plans WHERE id=?',(experiment_id,)).fetchone()
                scratch.register(ComparisonPlan.model_validate_json(plan[0]))
                result=scratch.evaluate(experiment_id,variant_id,observations,**kwargs)
                if result['input_hash']!=prior[0]:raise ValueError('interrupted trial input hash changed')
                with self.connection:
                    self.connection.execute('UPDATE comparison_trials SET result=? WHERE experiment_id=? AND variant_id=? AND result IS NULL',
                        (json.dumps(result,sort_keys=True,allow_nan=False),experiment_id,variant_id))
                return result
            finally:scratch.close()
        return super().evaluate(experiment_id,variant_id,observations,**kwargs)
