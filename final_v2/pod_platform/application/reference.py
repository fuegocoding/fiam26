"""Runnable, resumable reference workflow joining existing research and replay controls."""
from __future__ import annotations
import datetime as dt
from pathlib import Path
from decimal import Decimal
from dataclasses import asdict
from .common import UTC, SYNTHETIC_LABEL, atomic_write_json, read_json, digest, file_sha256, tree_sha256, parse_aware, now_utc
from .stages import StageRunner
from .namespaces import RoleContext
from .synthetic import SyntheticConfig, build_world, write_fixture
from .integrity import CandidateRecord, IntegrityPolicy, IntegrityService, normalize_campaign, classify_research_candidate
from .publication import PublicationService, ModelVersionRecord
from .artifacts import ArtifactIndex
from .decision_store import DecisionStore
from .judge import build_judge_package, build_commitment, acceptance_report, verify_commitment
from .twin_exhibit import build_twin_exhibit

EVALUATION_CUTOFF=dt.datetime(2026,8,31,23,59,59,tzinfo=UTC)

class ReferenceWorkflow:
    def __init__(self,state_dir,output_dir,*,run_id='reference_v1',clock=None,synthetic_config=None,exception_agent=None):
        self.state=Path(state_dir);self.output=Path(output_dir)
        self.state.mkdir(parents=True,exist_ok=True);self.output.mkdir(parents=True,exist_ok=True)
        self.runner=StageRunner(self.state,run_id,**({'clock':clock} if clock else {}))
        self.run_id=run_id;self.config=synthetic_config or SyntheticConfig()
        self.exception_agent=exception_agent
        code_root=Path(__file__).resolve().parents[1]
        self.source_hashes={p.relative_to(code_root).as_posix():file_sha256(p) for p in sorted(code_root.rglob('*.py'))}
    def _stage(self,name,fn,inputs):
        return self.runner.run(name,fn,inputs={**inputs,'output_dir':str(self.output.resolve())},
            source_hashes=self.source_hashes,artifacts_root=self.output,
            owned_outputs={
                'challenge':('candidates.json','admitted_models.json','certificates.json','synthetic_world.json'),
                'publication':('fixture','publications.json'), 'book':('book',),
                'research':('research_campaign.json','candidates.json','publications.json'),
                'acceptance':('code_manifest.json','judge_package.json','commitment.json','acceptance.json','reference_summary.json'),
            }.get(name,()))
    def _receipt(self,**payload):
        return {**payload,'artifacts':{p.relative_to(self.output).as_posix():file_sha256(p)
            for p in sorted(self.output.rglob('*')) if p.is_file() and p.name not in ('reference_summary.json',)}}
    def _finish(self,mode):
        def finish():
            atomic_write_json(self.output/'code_manifest.json',{'sha256':self.source_hashes})
            judge=build_judge_package(self.output)
            # This is a reproducible content freeze, never a claim of real prospective returns.
            build_commitment(self.output,frozen_at=now_utc().isoformat())
            accepted=acceptance_report(self.output)
            summary={'mode':mode,'run_id':self.run_id,'evaluation_cutoff':EVALUATION_CUTOFF.isoformat(),
                'label':SYNTHETIC_LABEL if mode=='demo' else mode.upper(),
                'llm_called':bool(self.exception_agent and not getattr(self.exception_agent,'is_fixture',False) and not getattr(getattr(self.exception_agent,'provider',None),'is_mock',False) and self.exception_agent.calls),
                'certificates_authorize_competition_book':False,
                'judge_status':judge.status,'software_status':accepted['software_status'],
                'investment_validation':'UNPROVEN','prospective_status':accepted['prospective']['status'],
                'output_dir':str(self.output.resolve()),'stages':self.runner.completed()}
            atomic_write_json(self.output/'reference_summary.json',summary)
            receipt=self._receipt(summary=summary)
            receipt['artifacts']['reference_summary.json']=file_sha256(self.output/'reference_summary.json')
            return receipt
        result=self._stage('acceptance',finish,{'mode':mode})
        verify_commitment(self.output)
        index_path=self.state/'application/reference_index.json'
        index=read_json(index_path) if index_path.is_file() else {}
        index[self.run_id]={'output_dir':str(self.output.resolve()),'summary':result['summary']}
        atomic_write_json(index_path,index)
        return result['summary']
    def run_demo(self):
        world=build_world(self.config)
        def challenge():
            from ..research.registry import Registry
            from .recovery import DurableResearchLedger, RecoverableRegisteredEvaluator
            from ..research.evaluation import RegisteredEvaluator
            registry=Registry(self.state/'demo_registry.sqlite')
            evaluator=RecoverableRegisteredEvaluator(self.state/'demo_evaluation.sqlite')
            ledger=DurableResearchLedger(registry,self.state/'demo_ledger.sqlite')
            service=IntegrityService(registry=registry,ledger=ledger,evaluator=evaluator,
                policy=IntegrityPolicy(policy_id='fixture_challenge_v1',frozen_at=world['registered_at']+dt.timedelta(days=1)),
                source_cards=[world['card']])
            ctx=RoleContext('integrity','integrity',self.run_id,world['evaluation_asof'],('design','check'))
            candidates=[];certs=[];models=[]
            try:
                for i,charter in enumerate(world['charters']):
                    candidate=CandidateRecord(candidate_id=f'fixture_candidate_{i}',charter=charter,namespace=f'strategy_{i}',
                        hypothesis_id=f'fixture_hypothesis_{i}',experiment_spec_id=f'fixture_experiment_{i}',
                        mechanism_family=f'fixture_family_{i}',model_id=f'fixture_model_{i}',model_hash=digest({'i':i,'seed':self.config.seed}),
                        target_id='synthetic_belief',horizon_trading_days=20,source_ids=['synthetic_signal'],
                        question='Does the registered planted signal improve the baseline?',
                        mechanism='Planted fixture factor; tests challenge plumbing, no investment inference.',declared_sign=1,
                        approved_universe='synthetic_universe',feature_hashes={'planted_signal':digest({'seed':self.config.seed})},
                        data_hashes={'synthetic_signal':digest(world['config'])})
                    service.register_candidate(candidate,registered_at=world['registered_at'])
                    # Last charter is deliberately wrong-sign: actual measured rejection.
                    design=[o.model_copy(update={'prediction':-o.prediction}) for o in world['design']] if i==3 else world['design']
                    check=[o.model_copy(update={'prediction':-o.prediction}) for o in world['check']] if i==3 else world['check']
                    decision=service.evaluate(candidate,design,check,ctx=ctx,evaluation_asof=world['evaluation_asof'],multiplicity=4)
                    candidates.append(decision.as_dict())
                    if decision.admitted:
                        certs.append(service.certificate(candidate.candidate_id).model_dump(mode='json'))
                        models.append(candidate.model_dump(mode='json'))
                atomic_write_json(self.output/'candidates.json',candidates)
                atomic_write_json(self.output/'admitted_models.json',models)
                atomic_write_json(self.output/'certificates.json',certs)
                atomic_write_json(self.output/'synthetic_world.json',{'label':world['label'],'config':world['config'],
                    'design':[o.model_dump(mode='json') for o in world['design']],
                    'check':[o.model_dump(mode='json') for o in world['check']]})
                return self._receipt(admitted=len(certs),rejected=sum(c['outcome']=='REJECTED' for c in candidates))
            finally:
                evaluator.close()
                ledger.close()
                registry.close()
        result=self._stage('challenge',challenge,{'mode':'demo','config':asdict(self.config)})
        if not result['admitted']:raise ValueError('no fixture candidate passed independent challenge; no strategies may trade')
        def publish():
            from ..research.promotion import AdmissionCertificate
            from ..contracts.forecasts import Forecast,ReturnForecast,TargetPrediction
            service=PublicationService(self.state/'publications.sqlite')
            index=ArtifactIndex(self.state/'artifact_index.sqlite')
            forecasts=[];certificates=[];batches=[]
            cards=read_json(self.output/'admitted_models.json')
            certs={c['model_id']:AdmissionCertificate.model_validate(c) for c in read_json(self.output/'certificates.json')}
            for item in cards:
                record=CandidateRecord.model_validate(item);cert=certs[record.model_id];certificates.append(cert)
                at=world['evaluation_asof']+dt.timedelta(days=1)
                model=ModelVersionRecord(owner=record.namespace,created_at=at,model_id=record.model_id,
                    candidate_id=record.candidate_id,charter=record.charter,namespace=record.namespace,model_hash=record.model_hash,
                    bridge_id='synthetic_rank_spread',bridge_slope_bp='200',training_cutoff=world['registered_at'])
                for month in range(4):
                    formation=at+dt.timedelta(days=30*month);end=formation+dt.timedelta(days=30)
                    evidence=f'evidence_{record.candidate_id}_{month}';uncertainty=f'uncertainty_{record.candidate_id}_{month}'
                    index.register_payload(evidence,'fixture_challenge',record.namespace,{'candidate':record.candidate_id,'challenge':digest(result)})
                    index.register_payload(uncertainty,'fixture_uncertainty',record.namespace,{'assumed_sd':.02,'synthetic':True})
                    batch=[]
                    for n in range(self.config.names):
                        # Same declared integration scale for each admitted model.
                        alpha=Decimal(str((1-2*n/(self.config.names-1))*.03))
                        f=Forecast(owner=record.namespace,created_at=formation,forecast_id=f'fc_{record.candidate_id}_{month}_{n}',
                            strategy_id=record.candidate_id,model_id=record.model_id,admission_certificate_id=cert.certificate_id,
                            security_id=f'S{n:03}',information_cutoff=formation-dt.timedelta(days=1),earliest_formation_at=formation,
                            target=TargetPrediction(target_id=record.target_id,prediction=alpha,unit='decimal_return',
                                reference_prediction=Decimal('0'),reference_type='synthetic_naive',expected_resolution_at=end),
                            return_forecast=ReturnForecast(expected_excess_return=alpha,unit='decimal_return',horizon_trading_days=20,
                                bridge_id=model.bridge_id,uncertainty_artifact_id=uncertainty),mechanism_ids=[record.mechanism_family],
                            source_family_ids=record.source_ids,evidence_artifact_ids=[evidence],expires_at=formation+dt.timedelta(days=20)).sealed()
                        batch.append(f)
                    ctx=RoleContext('strategy',record.namespace,self.run_id,formation)
                    batches.append(service.publish(ctx,forecasts=batch,certificate=cert,model=model,published_at=formation,
                        artifact_exists=index.exists,source_snapshots={'synthetic_signal':world['card'].snapshot_id}).model_dump(mode='json'))
                    forecasts.extend(batch)
            write_fixture(world,forecasts,certificates,self.output/'fixture')
            atomic_write_json(self.output/'publications.json',batches)
            return self._receipt(forecasts=len(forecasts),scope='fixture')
        self._stage('publication',publish,{'challenge':result})
        self._run_book(self.output/'fixture','test')
        return self._finish('demo')
    def _run_book(self,fixture_dir,label):
        from ..orchestration.replay import ControlledReplay,load_fixture
        from ..research.certificates import CertificatePolicy
        fixture_dir=Path(fixture_dir)
        fingerprint=tree_sha256(fixture_dir)
        def book():
            fx=load_fixture(fixture_dir)
            if max(p[1] for p in fx.periods)>EVALUATION_CUTOFF.date():
                raise ValueError('replay outcomes exceed mandatory 2026-08-31 evaluation cutoff')
            outcome=ControlledReplay(fx,self.output/'book',certificate_policy=CertificatePolicy.production() if label=='formation' else CertificatePolicy.test()).run()
            store=DecisionStore(self.state/'decision_store.sqlite')
            store.ingest_replay_run(self.output/'book',run_id=self.run_id,forecasts=fx.forecasts,certificates=fx.certificates)
            twin=build_twin_exhibit(self.output/'book',fx,agent=self.exception_agent,
                certificate_policy=CertificatePolicy.production() if label=='formation' else CertificatePolicy.test())
            if twin['status']!='COMPLETED':raise ValueError('paired replay failed; inspect all-arm failure evidence')
            from .exhibits import build_exhibits, build_removal_controls
            build_removal_controls(self.output/'book',fx,certificate_policy=CertificatePolicy.production() if label=='formation' else CertificatePolicy.test())
            build_exhibits(self.output/'book',fx,synthetic='synthetic_signal' in fx.cards)
            return self._receipt(book_summary=outcome['summary'],label=label)
        return self._stage('book',book,{'fixture_hash':fingerprint,'label':label,'exception_agent':self.exception_agent.identity() if self.exception_agent else None})
    def run_replay(self,fixture_dir,label='test'):
        if label not in ('test','formation'):raise ValueError('label must be test or formation')
        self._run_book(fixture_dir,label)
        return self._finish('replay')
    def run_research(self,manifest_path,dataset_id,research_asof,data_paths=()):
        from ..strategies.charter_campaign import run_charter_campaign
        asof=parse_aware(research_asof) if isinstance(research_asof,str) else research_asof
        from ..research.builtins.manifest import load_manifest
        dataset=load_manifest(manifest_path).dataset(dataset_id)
        def research():
            report=run_charter_campaign(manifest_path,self.state/'campaign',dataset_id=dataset_id,research_asof=asof)
            candidates=[classify_research_candidate(s,IntegrityPolicy(policy_id='research_v1',frozen_at=asof), {dataset.spec.source_id:dataset.source_card}).as_dict()
                for s in normalize_campaign(report)]
            atomic_write_json(self.output/'research_campaign.json',report)
            atomic_write_json(self.output/'candidates.json',candidates)
            atomic_write_json(self.output/'publications.json',[])
            return self._receipt(candidate_count=len(candidates),admitted=0)
        self._stage('research',research,{'manifest':file_sha256(manifest_path),'dataset_id':dataset_id,'research_asof':asof.isoformat(),
            'dataset_hash':file_sha256(dataset.path),'source_card_hash':dataset.source_card_hash,
            'data':{str(p):tree_sha256(p) for p in data_paths}})
        return self._finish('research')
