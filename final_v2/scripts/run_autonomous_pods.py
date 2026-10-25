#!/usr/bin/env python3
"""Run a finite four-pod research tick. Offline labelled fixtures by default opt-in."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pod_platform.application.research_loop import AutonomousResearchLoop, RoleProviders, build_fixture_loop
from pod_platform.agents.strategy_models import load_profiles, StrategyModelProvider


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state-dir', required=True, type=Path)
    p.add_argument('--fixture', action='store_true')
    p.add_argument('--online', action='store_true')
    p.add_argument('--serving-profiles', type=Path)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--run-id', default='research_v1')
    p.add_argument('--max-jobs', type=int, default=16)
    p.add_argument('--max-iterations', type=int, default=4)
    a = p.parse_args(argv)
    if a.fixture and a.online:
        p.error('--fixture and --online are separate modes')
    if a.fixture:
        loop = build_fixture_loop(a.state_dir)
    else:
        if not (a.online and a.serving_profiles and a.manifest):
            print(json.dumps({'status': 'BLOCKED_BACKEND_UNAVAILABLE',
                'requires': ['--online', '--serving-profiles', '--manifest', 'three explicit model roles']}))
            return 2
        profiles = load_profiles(a.serving_profiles)
        if set(profiles) != {'lead','challenger','extractor'}:
            p.error('exactly three explicit serving roles are required')
        loop = AutonomousResearchLoop(a.state_dir, providers=RoleProviders(**{r: StrategyModelProvider(profiles[r])
            for r in profiles}), manifest_path=a.manifest, run_id=a.run_id)
    print(json.dumps(loop.tick(max_jobs=a.max_jobs,max_iterations=a.max_iterations),default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
