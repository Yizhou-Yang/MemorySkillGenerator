#!/usr/bin/env python3
"""Tiny OpenAI-compatible mock endpoint (stdlib only, no model, no GPU).

Purpose: validate the pipeline -> local-endpoint plumbing on the laptop before
the GPU window, so the only genuinely-new step in the window is `vllm serve`.
It answers /v1/models and /v1/chat/completions with a canned assistant message
(no tool calls, so an agent turn terminates immediately) and a usage payload,
which exercises our client (openai_tool_chat / _openai_notool_sync / token
accounting / _msg_text / _window_messages) end to end.

    python scripts/latest/mock_openai.py --port 8999      # terminal 1
    # terminal 2:
    LLM_PROVIDER=openrouter OPENROUTER_BASE_URL=http://localhost:8999/v1 \
      OPENROUTER_API_KEY=EMPTY CODEBUDDY_MODEL=mock \
      python scripts/latest/openrouter_smoke.py
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class H(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/v1/models"):
            self._send({"object": "list", "data": [{"id": "mock", "object": "model"}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            req = {}
        msgs = req.get("messages", [])
        prompt_tokens = sum(len(str(m.get("content", ""))) for m in msgs) // 4
        answer = "READY"  # canned final answer, no tool_calls -> loop ends in 1 turn
        self._send({
            "id": "mock-1", "object": "chat.completion", "created": int(time.time()),
            "model": req.get("model", "mock"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": answer}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 1,
                      "total_tokens": prompt_tokens + 1},
        })

    def log_message(self, *a):
        pass  # quiet


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8999)
    p = ap.parse_args().port
    print(f"mock OpenAI endpoint on http://localhost:{p}/v1  (Ctrl-C to stop)")
    ThreadingHTTPServer(("127.0.0.1", p), H).serve_forever()
