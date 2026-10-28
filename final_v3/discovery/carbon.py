"""Free Carbon catalogue and quote facade; purchases are unreachable."""

import hashlib
import json
import os
import subprocess
import sys

import config


def buy_frameworks(*args, **kwargs):
    raise PermissionError('Carbon purchases prohibited: CARBON_PURCHASE_BUDGET_USD = 0')


class Carbon:
    buy_frameworks = staticmethod(buy_frameworks)

    def __init__(self, cache_dir=None):
        if config.CARBON_PURCHASE_BUDGET_USD != 0:
            raise ValueError('Discovery requires zero Carbon purchase budget')
        self.cache_dir = cache_dir or config.FINAL / 'discovery_cache' / 'carbon'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _call(self, operation, **arguments):
        if operation not in ('search', 'quote'):
            raise PermissionError('Only free search and quote operations are allowed')
        payload = dict(operation=operation, adapter_version=3, **arguments)
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / (key + '.json')
        if path.exists():
            return json.loads(path.read_text())
        if os.environ.get('FIAM_DISCOVERY_OFFLINE') == '1':
            raise RuntimeError('Offline Carbon cache miss: ' + key)
        worker = config.FINAL / 'discovery/carbon_worker.py'
        python = os.environ.get('FIAM_CARBON_PYTHON', sys.executable)
        result = subprocess.run([str(python), '-B', str(worker)], input=json.dumps(payload),
                                capture_output=True, text=True, timeout=240)
        if result.returncode:
            raise RuntimeError('Free Carbon request failed: ' + result.stdout[-300:])
        value = json.loads(result.stdout)
        path.write_text(json.dumps(value))
        return value

    def search(self, query):
        return self._call('search', query=query)

    def quote(self, insight_id, entities, end):
        return self._call('quote', insight_id=insight_id, entities=entities, end=end)
