#!/usr/bin/env python3
"""Manual tool execution layer for controlled multi-turn agent loops.

Replaces CodeBuddy SDK's black-box tool calling with manual execution,
enabling within-task EvoMem patch injection between turns.

Tools implemented:
- op-001: web_search(query)  — DuckDuckGo text search
- op-002: web_fetch(url)     — HTTP GET with text extraction
- op-003: file_read(path)    — Read local file
- op-004: file_write(path, content) — Write local file
- op-005: python_exec(code)  — Execute Python code in subprocess
- op-000: FINISH(answer)     — Return final answer
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
import urllib.request
import urllib.parse
import urllib.error
from typing import Any


class ManualToolExecutor:
    """Executes tools manually for GAIA/SWE-bench controlled loops.

    Provides the same tool capabilities as CodeBuddy SDK but with explicit
    execution control, enabling between-turn EvoMem patch injection.
    """

    # Maximum output size per tool result (chars)
    MAX_RESULT_SIZE = 8000

    # Maximum web fetch size (bytes)
    MAX_FETCH_SIZE = 500_000

    def __init__(self, working_dir: str = "/tmp/skillforge_gaia"):
        self.working_dir = working_dir
        os.makedirs(working_dir, exist_ok=True)
        self._tool_registry = {
            "op-001": ("web_search", self._web_search),
            "op-002": ("web_fetch", self._web_fetch),
            "op-003": ("file_read", self._file_read),
            "op-004": ("file_write", self._file_write),
            "op-005": ("python_exec", self._python_exec),
        }
        self.call_history: list[dict] = []

    def get_tool_list_text(self) -> str:
        """Generate the tool list text for the system prompt."""
        lines = [
            "op-001: Search the web for information | query:str [required]",
            "op-002: Fetch and read content from a URL | url:str [required]",
            "op-003: Read a local file | path:str [required]",
            "op-004: Write content to a local file | path:str [required] | content:str [required]",
            "op-005: Execute Python code and return output | code:str [required]",
            "op-000: FINISH — output final answer | answer:str [required]",
        ]
        return "\n".join(lines)

    def execute(self, tool_id: str, args: dict) -> str:
        """Execute a tool and return the result string."""
        if tool_id == "op-000":
            return f"[FINISH] {args.get('answer', '')}"

        entry = self._tool_registry.get(tool_id)
        if entry is None:
            return f"ERROR: Unknown tool {tool_id}. Available: {list(self._tool_registry.keys())}"

        tool_name, handler = entry
        try:
            result = handler(args)
            self.call_history.append({
                "tool": tool_name,
                "tool_id": tool_id,
                "args": args,
                "result_preview": str(result)[:500],
                "timestamp": time.time(),
            })
            # Truncate long results
            result_str = str(result)
            if len(result_str) > self.MAX_RESULT_SIZE:
                result_str = result_str[:self.MAX_RESULT_SIZE] + "\n...[truncated]"
            return result_str
        except Exception as e:
            error_msg = f"ERROR executing {tool_name}: {e}"
            self.call_history.append({
                "tool": tool_name,
                "tool_id": tool_id,
                "args": args,
                "error": str(e),
                "timestamp": time.time(),
            })
            return error_msg

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _web_search(self, args: dict) -> str:
        """Search the web using DuckDuckGo HTML (no API key needed)."""
        query = args.get("query", "")
        if not query:
            return "ERROR: query parameter is required for web_search"

        # Use DuckDuckGo Lite HTML search
        url = "https://lite.duckduckgo.com/lite/?" + urllib.parse.urlencode({"q": query})
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; SkillForge/1.0)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            return f"ERROR: web_search failed: {e}"

        # Extract result snippets from DuckDuckGo Lite HTML
        results = []
        # Pattern: <a rel="nofollow" href="...">title</a><span class="...">snippet</span>
        link_pattern = re.compile(
            r'<a\s+rel="nofollow"\s+(?:class="[^"]*"\s+)?href="([^"]+)"[^>]*>(.*?)</a>'
            r'\s*<span\s+class="[^"]*">(.*?)</span>',
            re.DOTALL,
        )
        for m in link_pattern.finditer(html):
            link_url = m.group(1)
            title = re.sub(r'<[^>]+>', '', m.group(2)).strip()
            snippet = re.sub(r'<[^>]+>', '', m.group(3)).strip()
            if title and snippet:
                results.append(f"Title: {title}\nURL: {link_url}\nSnippet: {snippet}")

        if not results:
            # Fallback: try simpler extraction
            snippet_pattern = re.compile(
                r'<td[^>]*>\s*<a[^>]*href="([^"]*)"[^>]*>([^<]*)</a>\s*</td>\s*'
                r'<td[^>]*>(.*?)</td>',
                re.DOTALL,
            )
            for m in snippet_pattern.finditer(html):
                link_url = m.group(1)
                title = m.group(2).strip()
                snippet = re.sub(r'<[^>]+>', '', m.group(3)).strip()
                if title and snippet:
                    results.append(f"Title: {title}\nURL: {link_url}\nSnippet: {snippet}")

        if not results:
            return f"No results found for: {query}"

        return f"Search results for '{query}':\n\n" + "\n\n".join(results[:10])

    def _web_fetch(self, args: dict) -> str:
        """Fetch content from a URL and extract readable text."""
        url = args.get("url", "")
        if not url:
            return "ERROR: url parameter is required for web_fetch"

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; SkillForge/1.0)",
                "Accept": "text/html,text/plain,*/*",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                # Check content type
                content_type = resp.headers.get("Content-Type", "")
                raw = resp.read(self.MAX_FETCH_SIZE)

                if "text/html" in content_type:
                    html = raw.decode("utf-8", errors="replace")
                    text = self._extract_text_from_html(html)
                else:
                    text = raw.decode("utf-8", errors="replace")

        except urllib.error.HTTPError as e:
            return f"ERROR: HTTP {e.code} when fetching {url}"
        except Exception as e:
            return f"ERROR: web_fetch failed for {url}: {e}"

        if len(text) > self.MAX_RESULT_SIZE:
            text = text[:self.MAX_RESULT_SIZE] + "\n...[truncated]"

        return f"Content from {url}:\n\n{text}"

    @staticmethod
    def _extract_text_from_html(html: str) -> str:
        """Extract readable text from HTML (strip tags, scripts, styles)."""
        # Remove scripts and styles
        html = re.sub(r'<script[^>]*>.*?</script>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html)
        # Normalize whitespace
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&quot;', '"', text)
        text = re.sub(r'&#?\w+;', ' ', text)
        # Collapse whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

    def _file_read(self, args: dict) -> str:
        """Read a local file."""
        path = args.get("path", "")
        if not path:
            return "ERROR: path parameter is required for file_read"

        # Resolve relative paths against working_dir
        if not os.path.isabs(path):
            path = os.path.join(self.working_dir, path)

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(self.MAX_RESULT_SIZE)
        except FileNotFoundError:
            return f"ERROR: File not found: {path}"
        except Exception as e:
            return f"ERROR reading {path}: {e}"

        if len(content) > self.MAX_RESULT_SIZE:
            content = content[:self.MAX_RESULT_SIZE] + "\n...[truncated]"

        return f"Content of {path}:\n\n{content}"

    def _file_write(self, args: dict) -> str:
        """Write content to a local file."""
        path = args.get("path", "")
        content = args.get("content", "")
        if not path:
            return "ERROR: path parameter is required for file_write"

        if not os.path.isabs(path):
            path = os.path.join(self.working_dir, path)

        os.makedirs(os.path.dirname(path) or self.working_dir, exist_ok=True)

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            return f"ERROR writing {path}: {e}"

        return f"Successfully wrote {len(content)} bytes to {path}"

    def _python_exec(self, args: dict) -> str:
        """Execute Python code and return stdout/stderr."""
        code = args.get("code", "")
        if not code:
            return "ERROR: code parameter is required for python_exec"

        # Write code to temp file
        tmp_path = os.path.join(self.working_dir, f"_exec_{int(time.time()*1000)}.py")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(code)

            result = subprocess.run(
                ["python3", tmp_path],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.working_dir,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            output = ""
            if stdout:
                output += f"stdout:\n{stdout}"
            if stderr:
                output += f"\nstderr:\n{stderr}"
            if not output:
                output = "(no output)"
            return output
        except subprocess.TimeoutExpired:
            return "ERROR: Python execution timed out (30s limit)"
        except Exception as e:
            return f"ERROR executing Python: {e}"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ══════════════════════════════════════════════════════════════════════════════
#  GAIA System Prompt
# ══════════════════════════════════════════════════════════════════════════════

GAIA_SYSTEM_PROMPT_TEMPLATE = """You are an expert research assistant solving multi-step QA tasks.
You have access to web search, web browsing, file I/O, and Python execution tools.

OUTPUT FORMAT (exactly 3 lines per turn):
REASONING: <one brief sentence about what you're doing and why>
NEXT_OP: <op-id>
PARAMS: <key>:<value> | <key>:<value>

EXAMPLES:
REASONING: Searching for the release date of Python 3.12.
NEXT_OP: op-001
PARAMS: query:Python 3.12 release date

REASONING: Fetching the Wikipedia page to verify the date.
NEXT_OP: op-002
PARAMS: url:https://en.wikipedia.org/wiki/Python

REASONING: Computing 2 + 2 to confirm the answer.
NEXT_OP: op-005
PARAMS: code:print(2 + 2)

REASONING: All evidence gathered. Confident answer is 42.
NEXT_OP: op-000
PARAMS: answer:42

RULES:
1. Always include a REASONING line. Keep it to ONE brief sentence.
2. ONE operation per turn. Never output multiple NEXT_OP lines.
3. Use ONLY real data from tool results. Never invent facts.
4. CRITICAL: ALWAYS search the web (op-001) before answering. NEVER answer from memory. You MUST use at least one search or fetch operation before calling op-000.
5. Cross-check information from multiple sources when possible.
6. For numerical answers, use Python execution (op-005) for calculations.
7. Call op-000 with your final answer ONLY when you are confident AND have gathered evidence.

STRATEGY:
• Search the web with short, specific queries (1-4 keywords).
• When you find a promising result, fetch the page content with op-002.
• Use Python (op-005) for any math, date calculations, or data processing.
• If a search returns no results, try different keywords.
• Do NOT repeat the same search query.
"""

SWE_SYSTEM_PROMPT_TEMPLATE = """You are an expert software engineer debugging code issues.
You have access to file I/O and Python execution tools.

OUTPUT FORMAT (exactly 2 lines, nothing else):
NEXT_OP: <op-id>
PARAMS: <key>:<value> | <key>:<value>

EXAMPLES:
NEXT_OP: op-003
PARAMS: path:src/main.py

NEXT_OP: op-005
PARAMS: code:import sys; print(sys.version)

NEXT_OP: op-000
PARAMS: answer:The bug is in line 42: should use >= instead of >

RULES:
1. Output EXACTLY 2 lines per turn: NEXT_OP + PARAMS. NO other text.
2. ONE operation per turn. Never output multiple NEXT_OP lines.
3. Read files first to understand the codebase before proposing fixes.
4. Write proposed fixes using op-004, then verify with op-005.
5. Call op-000 with your diagnosis and fix explanation ONLY when confident.

STRATEGY:
• Read relevant source files first to understand the issue.
• Identify the root cause before writing any fix.
• Write minimal, targeted fixes — do not refactor unrelated code.
• Verify fixes by running tests or executing the code.
"""
