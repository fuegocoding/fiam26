#!/usr/bin/env python3
"""Create a local Arcana positions review bundle from explicit canonical JSON."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pod_platform.reporting.arcana import make_bundle,write_bundle
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--out',required=True);p.add_argument('--mapping');p.add_argument('--schema');p.add_argument('--weight-unit',choices=['decimal','percent'],default='decimal');p.add_argument('--source-file',action='append',default=[]);p.add_argument('--descriptive',action='store_true');p.add_argument('--overwrite',action='store_true');a=p.parse_args()
 records=json.loads(Path(a.input).read_text(encoding='utf-8'));mapping=json.loads(Path(a.mapping).read_text()) if a.mapping else None;schema=json.loads(Path(a.schema).read_text()) if a.schema else None
 b=make_bundle(records,mapping=mapping,vendor_schema=schema,weight_unit=a.weight_unit,source_files=[a.input,*a.source_file],descriptive=a.descriptive);cp,jp=write_bundle(b,a.out,overwrite=a.overwrite)
 print(json.dumps({'status':b['status'],'csv':str(cp),'metadata':str(jp),'payload_sha256':b['payload_sha256']}));return 0
if __name__=='__main__':
 try:raise SystemExit(main())
 except Exception as e:print(f'Arcana export failed: {e}',file=sys.stderr);raise SystemExit(2)
