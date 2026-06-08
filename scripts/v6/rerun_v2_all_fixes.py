"""DEPRECATED — use scripts/v6/latest_runner.py instead.

The runner has been renamed to `latest_runner.py` to reflect that the latest
experiment uses cross-agent skill quality critic (not oracle-driven retry)
and Exact Match / pass@1 metrics aligned with competing papers.
"""
import sys
sys.stderr.write(
    "rerun_v2_all_fixes.py is deprecated. "
    "Run `python scripts/v6/latest_runner.py` instead.\n"
)
sys.exit(1)
