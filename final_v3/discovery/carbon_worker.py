"""Carbon discovery through the project MCP/OAuth transport only."""
import json
from pathlib import Path
import sys

def run(payload):
    operation = payload.get('operation')
    if operation == 'quote':
        return {'status':'not_available_via_mcp','price':None,
                'note':'REST quotes retired; use team-account MCP allocation and asset entitlement.',
                'insight_id':payload['insight_id']}
    if operation != 'search':
        raise PermissionError('Unsupported Carbon operation')
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'final_v2'))
    from pod_platform.data.carbonarc_requests import CarbonArcOAuthTransport
    transport = CarbonArcOAuthTransport(root,online=True)
    query = payload['query']
    return {'transport':'MCP','account_policy':'competition team account',
            'insights':transport('search_insights',{'query':query,'limit':12}),
            'datasets':transport('data_library',{'query':query})}

if __name__ == '__main__':
    try:
        print(json.dumps(run(json.load(sys.stdin))))
    except Exception as exc:
        print(json.dumps({'error':type(exc).__name__,'transport':'MCP'}))
        sys.exit(1)
