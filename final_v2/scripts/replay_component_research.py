#!/usr/bin/env python3
"""Replay saved research through bounded component tests without calling model APIs."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pod_platform.application.component_replay import replay_components


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-research', required=True, type=Path)
    parser.add_argument('--state-dir', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--papers-db', required=True, type=Path)
    args = parser.parse_args()
    result = replay_components(from_research=args.from_research, state_dir=args.state_dir,
        manifest_path=args.manifest, papers_db=args.papers_db)
    print(json.dumps({'new_model_api_calls': 0, 'admissions': result['admissions'],
        'comparisons': [{'namespace': c['namespace'], 'status': c['route']['status'],
            'measurement': c['route'].get('probe_report', {}).get('status')} for c in result['comparisons']]}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
