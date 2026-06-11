"""
Agent abstraction layer for SkillForge.

SkillForge is an experience quality-assurance pipeline that augments
existing agents (OpenHands, Terminus 2, Memento-Skills, A-Mem) with:

1. CRITIC GATE — evaluates execution success before storing experience.
   Unlike Memento-Skills (which stores all reflections), we only keep
   experiences from verified successful executions. This prevents
   pollution of the experience library with bad strategies.

2. AI REFINE — rewrites raw execution traces into structured, reusable
   lessons. Turns "I tried X and got error Y, then tried Z" into
   "When facing Y, prefer Z over X because...". This improves retrieval
   relevance and reduces noise.

3. A/B/C MEASUREMENT — quantifies the value of each pipeline component:
     A = baseline (no experience)
     B = raw experience (retrieval only)
     C = refined experience (critic + refine)
   This provides scientific evidence that the pipeline works.

These agents do not have this capability internally:
  - OpenHands: no persistent cross-session experience library
  - Terminus 2: pure executor, no learning at all
  - Memento-Skills: stores raw reflections (no critic gate, no refine)
  - A-Mem: stores raw conversation memories (no refine)

Agent delegation map (EvoMem-style):
  gaia                  -> MementoSAgent    (CodeBuddy web-search tools)
  gaia2                 -> Terminus2Agent   (Harbor Docker / shell / prompt)
  locomo                -> AmemAgent        (A-Mem conversation memory QA)
  terminal_bench_2      -> Terminus2Agent   (Harbor Docker / shell / prompt)
  persona_mem_evo       -> AmemAgent        (A-Mem persona memory)
  swe_chain_evo         -> OpenHandsAgent   (LEGACY: code engineering, not published)
"""
from .base import BaseAgent, AgentFactory
from .terminal import TerminalAgent
from .amem import AmemAgent
from .memento import MementoSAgent
from .terminus2 import Terminus2Agent
from .openhands import OpenHandsAgent

__all__ = [
    "BaseAgent", "AgentFactory",
    "TerminalAgent", "AmemAgent",
    "MementoSAgent", "Terminus2Agent", "OpenHandsAgent",
]
