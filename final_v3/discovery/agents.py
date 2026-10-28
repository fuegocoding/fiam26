"""Cached Groq researcher and scout requests with bounded retries."""

import ast
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request

import config

MODELS = ('openai/gpt-oss-120b', 'openai/gpt-oss-20b')
# Free local fallback once both hosted models reach their daily token limits.
LOCAL_MODEL = 'qwen3:8b'
LOCAL_URL = 'http://localhost:11434/v1/chat/completions'
SIZES = ['mega', 'large', 'small', 'micro', 'nano']
SECTORS = ['10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60']


def load_env(path, wanted):
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip().removeprefix('export ').strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                if key.strip() == wanted:
                    os.environ.setdefault(wanted, value.strip().strip('\"\''))
    return os.environ.get(wanted)


def charters():
    """Read static pod declarations without importing the v2 runtime."""
    path = config.FINAL.parent / 'final_v2/pod_platform/pods/charters.py'
    result = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'PodSpec':
            fields = {k.arg: ast.literal_eval(k.value) for k in node.keywords
                      if k.arg in ('pod_id', 'mission', 'mechanism_families')}
            result[fields.pop('pod_id')] = fields
    if len(result) != 4:
        raise ValueError('Expected four pod charters')
    return result


def parse(raw):
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.S)  # local reasoning models
    return json.loads(re.sub(r'^\s*```(?:json)?\s*|\s*```\s*$', '', raw))


class Provider:
    """Cache raw model responses; offline misses always raise."""

    fake = False

    def __init__(self, cache_dir=None, max_requests=None):
        self.cache_dir = cache_dir or config.FINAL / 'discovery_cache' / 'responses'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last = -float('inf')
        self.day_limited = set()
        self.max_requests = max_requests
        self.requests = 0

    def call(self, messages):
        prompt = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        paths = [(model, self.cache_dir / (hashlib.sha256((model + prompt).encode()).hexdigest() + '.json'))
                 for model in (*MODELS, LOCAL_MODEL)]
        # Prefer any complete cached response even if 120b is usable today.
        for model, path in paths:
            if path.exists():
                saved = json.loads(path.read_text())
                if saved['model'] != model or saved['prompt'] != prompt:
                    raise ValueError('Corrupt discovery response cache')
                return saved['response']
        if os.environ.get('FIAM_DISCOVERY_OFFLINE') == '1':
            raise RuntimeError('Offline discovery cache miss: ' + paths[0][1].name)
        key = load_env(config.FINAL.parent / 'local/groq-api.env', 'GROQ_API_KEY')
        if not key:
            raise RuntimeError('GROQ_API_KEY is not configured')
        for model, path in paths[:len(MODELS)]:
            if model in self.day_limited:
                continue
            for attempt in range(3):
                time.sleep(max(0, 60 - (time.monotonic() - self.last)))
                body = json.dumps(dict(model=model, messages=messages, temperature=0,
                                       max_tokens=2500, reasoning_effort='low')).encode()
                request = urllib.request.Request('https://api.groq.com/openai/v1/chat/completions', data=body,
                    headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json',
                             'User-Agent': 'fiam26-discovery'})
                if self.max_requests is not None and self.requests >= self.max_requests:
                    raise RuntimeError('Discovery request limit reached')
                self.requests += 1
                self.last = time.monotonic()
                try:
                    with urllib.request.urlopen(request, timeout=120) as response:
                        raw = json.load(response)['choices'][0]['message']['content']
                    temporary = path.with_suffix('.tmp')
                    temporary.write_text(json.dumps(dict(model=model, prompt=prompt, response=raw)))
                    temporary.replace(path)
                    return raw
                except urllib.error.HTTPError as exc:
                    detail = exc.read().decode(errors='replace').lower()
                    daily = exc.code == 429 and ('tokens per day' in detail or 'tpd' in detail)
                    print(f'Discovery provider: {model}, HTTP {exc.code}, daily_limit={daily}', flush=True)
                    if daily:
                        self.day_limited.add(model)
                        break
                    if exc.code != 429 or attempt == 2:
                        raise RuntimeError(f'Discovery provider failed: HTTP {exc.code}') from None
                    retry = exc.headers.get('Retry-After', '120')
                    time.sleep(max(120, float(retry) if retry.isdigit() else 120))
        return self._local(messages, prompt, paths[-1][1])

    def _local(self, messages, prompt, path):
        """Answer with the local model, announced so the switch is never silent."""
        print(f'Discovery provider: hosted models at daily limit; using local {LOCAL_MODEL}', flush=True)
        body = json.dumps(dict(model=LOCAL_MODEL, messages=messages, temperature=0)).encode()
        request = urllib.request.Request(LOCAL_URL, data=body, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                raw = json.load(response)['choices'][0]['message']['content']
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f'Discovery providers exhausted and local model unavailable: {exc}') from None
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(model=LOCAL_MODEL, prompt=prompt, response=raw)))
        temporary.replace(path)
        return raw


def researcher_prompt(pod, charter, catalogue, feedback, year):
    return [dict(role='user', content=f'''Propose exactly 3 new economic sleeves for {pod} for return year {year}.
Return only a strict JSON array. Mission: {json.dumps(charter)}.
Each object has exactly name (unique snake_case), pod, hypothesis, mechanism, expected_direction (strings),
inputs (1-3 objects with source, feature, integer sign +1 or -1, optional transform: level|chg3|sector_rank),
cohort (size_groups: nonempty subset of {SIZES}, gics2: subset of {SECTORS}, [] means all).
Use only loadable features below. Higher signed mean within-month percentile predicts higher return.
Formation price >= 1. No imputation. chg3 is a calendar three-month difference.
Evidence targets: 2015-02 through {year-1}-12. Never claim unseen test performance.
Gate: S >= .60, shuffled control, spread NW(1) t >= 2, absolute pod spread correlation < .7, cap two.
Catalogue: {json.dumps(catalogue, sort_keys=True)}
Compact prior feedback: {json.dumps(feedback)}''')]


def scout_prompt(pod, charter, results, year):
    return [dict(role='user', content=f'''Data scout for {pod}, year {year}. Mission: {json.dumps(charter)}.
From this free catalogue search return a strict JSON array of at most 3 data requests.
Each request must have insight_id (integer), why, expected_direction (strings), entities ("ticker" for all tickers
or a list of {{"carc_id": integer, "representation": "ticker"}}).
Use only insight IDs in the search results. These requests are a shopping list only, never trading inputs.
No purchases are permitted. Search results: {json.dumps(results)}''')]
