import csv, json
import pytest
from pod_platform.reporting.arcana import make_bundle,write_bundle,publish_bundle

BASE=[{'date':'2026-01-02','security_id':'ABC','weight':.25,'source_classification':'research'}]
def test_review_only_and_explicit_mapping_units():
 b=make_bundle(BASE); assert b['status']=='canonical_review_only' and not b['vendor_compatible'] and not b['live_uploaded']
 mapping={'date':'asof','security_id':'id','weight':'allocation','source_classification':'evidence'}
 schema={'asof':'date','id':'string','allocation':'number','evidence':'string'}
 b=make_bundle(BASE,mapping=mapping,vendor_schema=schema,weight_unit='percent')
 assert b['rows'][0]=={'asof':'2026-01-02','id':'ABC','allocation':25,'evidence':'research'}
 assert b['status']=='vendor_schema_validated_export'
def test_schema_mapping_and_duplicate_validation():
 with pytest.raises(ValueError,match='duplicate'): make_bundle(BASE*2)
 with pytest.raises(ValueError,match='finite'): make_bundle([{**BASE[0],'weight':float('nan')}])
 with pytest.raises(ValueError,match='after'): make_bundle([{**BASE[0],'date':'2026-09-01'}])
 with pytest.raises(ValueError,match='collision'): make_bundle(BASE,mapping={'date':'x','security_id':'x','weight':'w','source_classification':'c'})
 with pytest.raises(ValueError,match='schema missing'): make_bundle(BASE,mapping={'date':'d','security_id':'s','weight':'w','source_classification':'c'},vendor_schema={'d':'date'})
def test_atomic_export_overwrite(tmp_path):
 b=make_bundle(BASE); c,j=write_bundle(b,tmp_path/'bundle'); assert next(csv.DictReader(c.open()))['security_id']=='ABC'
 with pytest.raises(FileExistsError): write_bundle(b,tmp_path/'bundle')
 write_bundle(b,tmp_path/'bundle',overwrite=True)
def test_publish_requires_consent_and_hides_secret_errors(tmp_path,monkeypatch):
 b=make_bundle(BASE); calls=[]; monkeypatch.setenv('ARCANA_TEST_SECRET','do-not-leak')
 def send(**kw): calls.append(kw); raise ValueError('bad response do-not-leak')
 with pytest.raises(PermissionError): publish_bundle(b,endpoint='https://example.invalid/api',key_env='ARCANA_TEST_SECRET',transport=send,consent=False,receipt_path=tmp_path/'receipt.json')
 with pytest.raises(RuntimeError) as e: publish_bundle(b,endpoint='https://example.invalid/api',key_env='ARCANA_TEST_SECRET',transport=send,consent=True,receipt_path=tmp_path/'receipt.json')
 assert 'do-not-leak' not in str(e.value) and calls[0]['allow_redirects'] is False
def test_receipt_idempotency_blocks_changed_payload_before_transport(tmp_path,monkeypatch):
 monkeypatch.setenv('ARCANA_TEST_SECRET','secret'); calls=[]
 def send(**kw): calls.append(kw); return {'receipt_id':'r-1'}
 b=make_bundle(BASE); receipt=tmp_path/'receipt.json'
 a=publish_bundle(b,endpoint='https://example.invalid/api',key_env='ARCANA_TEST_SECRET',transport=send,consent=True,receipt_path=receipt)
 assert publish_bundle(b,endpoint='https://example.invalid/api',key_env='ARCANA_TEST_SECRET',transport=send,consent=True,receipt_path=receipt)==a
 changed=make_bundle([{**BASE[0],'weight':.4}])
 with pytest.raises(ValueError,match='idempotency conflict'): publish_bundle(changed,endpoint='https://example.invalid/api',key_env='ARCANA_TEST_SECRET',transport=send,consent=True,receipt_path=receipt)
 assert len(calls)==1
