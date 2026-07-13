#!/usr/bin/env python3
"""Launcher that registers CuratedTau2Agent in tau2's global registry,
then delegates to the official tau2 CLI. This is the replacement for
--agent-import-path (which tau2 CLI does NOT support).

Usage (server):
  PYTHONPATH=$PWD python scripts/latest/tau2_launcher.py run --domain airline ...

The bridge sets TAU2_ARM and TAU2_MEM_STATE env vars; CuratedTau2Agent
reads them at call time (not at construction) so the factory is identical
for arms B and C.
"""
import sys

# Register our custom agent BEFORE tau2's CLI parses --agent
from tau2.registry import registry
from scripts.latest.tau2_agent import create_curated_tau2_agent

registry.register_agent_factory(create_curated_tau2_agent, "curated_tau2_agent")

# Delegate to the official tau2 CLI
from tau2.cli import main
main()
