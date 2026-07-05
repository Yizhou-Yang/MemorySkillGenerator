#!/usr/bin/env python3
"""End-to-end test for Terminal-Bench-2.0 Docker execution."""
import asyncio, json, os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "latest"))

os.environ["LLM_PROVIDER"] = "codebuddy"
os.environ["CODEBUDDY_MODEL"] = "deepseek-v4-pro"
os.environ.setdefault("CODEBUDDY_INTERNET_ENVIRONMENT", "ioa")

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from latest.agent.terminus2 import Terminus2Agent


async def main():
    agent = Terminus2Agent(timeout=600)
    task = {
        "task_id": "adaptive-rejection-sampler",
        "description": (
            "Implement an adaptive rejection sampler for a target distribution. "
            "The solution should include the ARS algorithm and pass all tests."
        ),
        "expected": "[Terminal-Bench 2.0 task]",
        "metadata": {"category": "scientific-computing", "difficulty": "medium"},
    }

    print("=== Terminal-Bench-2.0 E2E Test ===")
    print(f"Task: {task['task_id']}")
    result = await agent.run_task(task, "", "A")

    print("\n=== RESULT ===")
    for k, v in result.items():
        if isinstance(v, str) and len(v) > 500:
            print(f"  {k}: [{len(v)} chars] {v[:300]}...")
        else:
            print(f"  {k}: {v}")

    passed = result.get("test_passed", False)
    mode = result.get("execution_mode", "?")
    cost = result.get("time_cost", 0)
    print(f"\ntest_passed={passed}, mode={mode}, time={cost:.1f}s")

    # Save full result
    safe_result = {}
    for k, v in result.items():
        safe_result[k] = str(v)[:5000] if isinstance(v, str) else v
    with open("/tmp/tb2_test_result.json", "w") as f:
        json.dump(safe_result, f, indent=2, default=str)
    print("Result saved to /tmp/tb2_test_result.json")

    return passed


if __name__ == "__main__":
    success = asyncio.run(main())
    print(f"\n{'PASS' if success else 'FAIL'}: test_passed={success}")
