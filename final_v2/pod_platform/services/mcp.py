"""Small MCP stdio transport; no listening network socket or shell tool.

Protocol reference:
https://modelcontextprotocol.io/specification/2025-06-18/basic/transports

The parent process fixes the pod identity and registers bounded handlers. Tool
arguments cannot change the identity or open arbitrary files. JSON-RPC messages
are newline delimited; stdout contains protocol messages only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, TextIO

from pydantic import BaseModel, ValidationError


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    arguments: type[BaseModel]
    handler: Callable[[BaseModel], Any]
    read_only: bool = True

    def describe(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.arguments.model_json_schema(),
            "annotations": {
                "readOnlyHint": self.read_only,
                "destructiveHint": False,
                "openWorldHint": False,
            },
        }


def _error(request_id, code: int, message: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


class MCPServer:
    def __init__(self, tools: list[Tool], *, name: str = "fiam-pod-platform"):
        self.tools = {tool.name: tool for tool in tools}
        if len(self.tools) != len(tools):
            raise ValueError("duplicate tool names")
        self.name = name
        self.initialized = False

    def dispatch(self, request: object) -> dict | None:
        if not isinstance(request, dict):
            return _error(None, -32600, "request must be a JSON-RPC object")
        request_id = request.get("id")
        if request.get("jsonrpc") != "2.0" or not isinstance(
            request.get("method"), str
        ):
            return _error(request_id, -32600, "invalid JSON-RPC request")
        if "id" in request and (
            isinstance(request_id, bool)
            or not isinstance(request_id, (str, int, type(None)))
        ):
            return _error(None, -32600, "invalid request id")
        method = request["method"]
        if "id" not in request:
            # Notifications never have a JSON-RPC response.
            return None
        params = request.get("params", {})
        if not isinstance(params, dict):
            return _error(request_id, -32602, "params must be an object")
        if method == "initialize":
            offered = params.get("protocolVersion")
            version = (
                offered
                if offered in {"2024-11-05", "2025-03-26", "2025-06-18"}
                else "2025-06-18"
            )
            self.initialized = True
            result = {
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": self.name, "version": "0.1.0"},
                "instructions": "Pod identity is fixed by the server process. Research outputs never authorize trading.",
            }
        elif method == "ping":
            result = {}
        elif not self.initialized:
            return _error(request_id, -32002, "initialize the server first")
        elif method == "tools/list":
            result = {
                "tools": [tool.describe() for tool in self.tools.values()]
            }
        elif method == "tools/call":
            name = params.get("name")
            if not isinstance(name, str) or name not in self.tools:
                return _error(request_id, -32602, "unknown tool")
            try:
                tool = self.tools[name]
                arguments = tool.arguments.model_validate(
                    params.get("arguments", {})
                )
                value = tool.handler(arguments)
                if isinstance(value, BaseModel):
                    value = value.model_dump(mode="json")
                content = json.dumps(
                    value, sort_keys=True, default=str, allow_nan=False
                )
                result = {
                    "content": [{"type": "text", "text": content}],
                    "isError": False,
                }
            except (
                ValidationError,
                ValueError,
                PermissionError,
                KeyError,
            ) as exc:
                result = {
                    "content": [{"type": "text", "text": str(exc)}],
                    "isError": True,
                }
            except Exception:
                # Provider failures may contain credentials/URLs in their repr.
                result = {
                    "content": [
                        {
                            "type": "text",
                            "text": "Tool failed; inspect local diagnostics.",
                        }
                    ],
                    "isError": True,
                }
        else:
            return _error(request_id, -32601, "method not found")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def serve(
        self,
        source: TextIO,
        destination: TextIO,
        *,
        max_message_chars: int = 1_048_576,
    ):
        while True:
            line = source.readline(max_message_chars + 1)
            if not line:
                return
            if len(line) > max_message_chars:
                destination.write(
                    json.dumps(_error(None, -32600, "message too large")) + "\n"
                )
                destination.flush()
                return  # do not parse the tail of an oversized message as another request
            try:
                request = json.loads(line)
            except (ValueError, RecursionError):
                response = _error(None, -32700, "invalid JSON")
            else:
                response = self.dispatch(request)
            if response is not None:
                destination.write(json.dumps(response, allow_nan=False) + "\n")
                destination.flush()
