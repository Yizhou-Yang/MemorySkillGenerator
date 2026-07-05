#!/usr/bin/env python3
"""Smoke test for the GAIA2 native tool-calling path.

Run this on the server (where codebuddy_agent_sdk is installed) BEFORE a full
GAIA2 sweep. It exercises the real helpers in gaia2_runner.py:

  1. Wiring test (always): builds a one-tool in-process MCP server via the same
     _make_tool_def / SdkMcpServer / _drive_native_query used in prod,
     and asks HY3 to call it. Confirms the SDK MCP shape works with the gateway.

  2. Scenario test (optional): pass a GAIA2 scenario.json path as argv[1] to run
     one real task through _gaia2_native_sync and print event-log / action counts.

    python scripts/latest/gaia2_native_smoke.py
    python scripts/latest/gaia2_native_smoke.py /path/to/environment/scenario.json

Exit code 0 = the native path fired tools; non-zero = it did not (keep the text
protocol with GAIA2_NATIVE_TOOLS=0 and send me the printed error).
"""
import asyncio
import sys

import gaia2_runner as G


def wiring_test() -> bool:
    if not G._MCP_AVAILABLE:
        print("FAIL: SDK has no MCP support (_MCP_AVAILABLE=False). "
              "create_sdk_mcp_server / SdkMcpTool not importable.")
        return False

    called = {"hit": False, "arg": None}

    async def echo_handler(args):
        called["hit"] = True
        called["arg"] = (args or {}).get("message")
        return {"content": [{"type": "text",
                             "text": f"echoed: {called['arg']}"}]}

    echo_def = G._make_tool_def(
        "echo", "Echo the message back to the caller.",
        {"type": "object",
         "properties": {"message": {"type": "string",
                                    "description": "Text to echo."}},
         "required": ["message"]},
        echo_handler,
    )
    server = G.SdkMcpServer(G.SdkMcpServerOptions(name="are", version="1.0.0", tools=[echo_def]))
    server_cfg = {"type": "sdk", "name": "are", "server": server}

    result: dict = {}
    prompt = ("Call the echo tool once with message='ping'. "
              "After you see the result, stop.")
    system = "You are a test agent. Use the provided tool, then stop."

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(G._drive_native_query(
            prompt, system, server_cfg, ["mcp__are__echo"], 4, result))
    finally:
        G._shutdown_loop(loop)

    print(f"  tool fired : {called['hit']}  (arg={called['arg']!r})")
    print(f"  response   : {(result.get('response') or '')[:200]!r}")
    if result.get("error"):
        print(f"  error      : {result['error']}")

    if called["hit"]:
        print("PASS: native tool-calling reached the handler.")
        return True
    print("FAIL: model did not call the tool. Likely an allowed_tools name or "
          "tool-schema mismatch. Inspect the error above.")
    return False


def scenario_test(scenario_path: str) -> bool:
    base = {"task_id": "smoke", "expected": [], "oracle_answer": "",
            "description": "", "metadata": {}, "response": "", "error": None,
            "time_cost": 0, "augmented": False, "group": "A",
            "actions": [], "event_log": []}
    res = G._gaia2_native_sync(scenario_path, "", "", base, 20)
    print(f"  actions    : {len(res.get('actions', []))}")
    print(f"  event_log  : {len(res.get('event_log', []))}")
    print(f"  error      : {res.get('error')}")
    print(f"  _native_failed: {res.get('_native_failed', False)}")
    ok = len(res.get("event_log", [])) > 1 and not res.get("_native_failed")
    print("PASS: real scenario produced tool events."
          if ok else "WARN: few/no events — check the error above.")
    return ok


def main() -> int:
    print(f"GAIA2_NATIVE_TOOLS={G._GAIA2_NATIVE_TOOLS}  _MCP_AVAILABLE={G._MCP_AVAILABLE}")
    print("[1] wiring test (echo tool) ...")
    ok = wiring_test()
    if len(sys.argv) > 1:
        print(f"[2] scenario test ({sys.argv[1]}) ...")
        ok = scenario_test(sys.argv[1]) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
