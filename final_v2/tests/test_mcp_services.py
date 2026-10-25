import io
import json

from pydantic import BaseModel, ConfigDict

from pod_platform.services.mcp import MCPServer, Tool


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str


def server():
    return MCPServer([Tool("papers.search", "Search papers", Args,
                           lambda args: {"query": args.query})])


def request(method, params=None, request_id=1):
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}


def test_handshake_and_strict_tool_calls():
    s = server()
    assert s.dispatch(request("tools/list"))["error"]["code"] == -32002
    init = s.dispatch(request("initialize", {"protocolVersion": "2025-06-18"}))
    assert init["result"]["protocolVersion"] == "2025-06-18"
    assert s.dispatch(request("tools/list"))["result"]["tools"][0]["name"] == "papers.search"
    result = s.dispatch(request("tools/call", {"name": "papers.search", "arguments": {"query": "earnings"}}))
    assert json.loads(result["result"]["content"][0]["text"]) == {"query": "earnings"}
    bad = s.dispatch(request("tools/call", {"name": "papers.search", "arguments": {"query": "x", "pod": "other"}}))
    assert bad["result"]["isError"]


def test_protocol_errors_and_notifications():
    s = server()
    assert s.dispatch([])["error"]["code"] == -32600
    assert s.dispatch({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert s.dispatch(request("ping"))["result"] == {}
    s.dispatch(request("initialize"))
    assert s.dispatch(request("unknown"))["error"]["code"] == -32601
    assert s.dispatch(request("tools/call", {"name": []}))["error"]["code"] == -32602


def test_stdio_only_protocol_and_bounded_messages():
    source = io.StringIO("invalid\n" + json.dumps(request("initialize")) + "\n")
    output = io.StringIO()
    server().serve(source, output)
    replies = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(replies) == 2
    assert replies[0]["error"]["code"] == -32700
    output = io.StringIO()
    server().serve(io.StringIO("x" * 21), output, max_message_chars=20)
    assert json.loads(output.getvalue())["error"]["message"] == "message too large"
