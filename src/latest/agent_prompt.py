"""Agent System Prompt — General-purpose task execution guidance.

This module provides the system prompt template for GAIA2 ARE tasks.
The prompt follows a strict format: every rule is a GENERAL PRINCIPLE that applies
to any task, not a benchmark-specific hack.

Key design elements:
1. OUTPUT FORMAT — strict 2-line NEXT_OP + PARAMS format
2. RULES — non-negotiable constraints (anti-repeat, idempotency, no hallucination)
3. OPERATIONAL BEST PRACTICES — general heuristics (search strategy, parameter format)
4. COMPLETION PROTOCOL — twist-aware two-phase execution
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════════════
#  System Prompt Template
#
#  The prompt is assembled from two parts:
#  (a) Fixed template with rules and best practices
#  (b) Dynamic tool list (op-001 → ARE tool name mapping)
#  (c) Optional experience injection section
# ══════════════════════════════════════════════════════════════════════════════

# ── Part (a): Fixed template ──────────────────────────────────────────────
GAIA2_SYSTEM_PROMPT_TEMPLATE = """You are a task executor. You call ONE operation per turn.

OUTPUT FORMAT (exactly 2 lines, nothing else):
NEXT_OP: <op-id>
PARAMS: <key>:<value> | <key>:<value>

EXAMPLES:
NEXT_OP: op-024
PARAMS: query:Film

NEXT_OP: op-075
PARAMS: title:Meeting | start_datetime:2024-10-19 08:00:00 | end_datetime:2024-10-19 20:00:00 | attendees:["John Smith"]

NEXT_OP: op-076
PARAMS: event_id:ABC123

NEXT_OP: op-000
PARAMS: timeout_seconds:30

RULES:
1. Output EXACTLY 2 lines per turn: NEXT_OP + PARAMS. NO other text.
2. ONE operation per turn. Never output multiple NEXT_OP lines.
3. Use ONLY real data from results. Never invent names, IDs, or emails.
4. NEVER explain, plan, or narrate. ONLY output NEXT_OP + PARAMS.
5. NEVER repeat the same operation with the same params. If you already
   called an operation, do NOT call it again with identical arguments.
   If no reply comes after op-000, proceed with your best judgment.

OPERATIONAL BEST PRACTICES:
• Searching: Use the shortest distinctive keyword (1-2 words). If 0 results,
  try a different single word. Never repeat the same query or use long phrases.
  Browsing/listing shares the same data across apps — don't switch between them.
• Date lookups: When finding events on a specific day, query by date range
  (start/end datetime), not by text search (text searches titles, not dates).
• Collections: To find one item from many (codes, products, etc.), list ALL
  items first and scan results — don't query items one by one.
• Parameters: recipients use email addresses ["x@y.com"]. attendees use
  the person's FULL NAME ["John Smith"] — never user_ids or UUIDs.
  Do NOT look up user_ids for calendar events. Just use the contact's name.
• Budget: Max 5 attempts per sub-task. If stuck, skip and proceed.
  Complete primary actions within 20 turns total.
• Errors: If a tool errors, fix the parameter type/format. Never retry same params.
• Independence: Each task is self-contained. Only use names/IDs from the current
  task description and tool results. Nothing carries over from previous tasks.

COMPLETION PROTOCOL:
Tasks often have two phases. Phase 1: execute primary actions. Phase 2: handle
replies or conditional follow-ups ('if X, then Y'). The correct sequence is:
  Phase 1 actions → op-001 (notify user) → op-000 (wait for reply, timeout:60)
  → process any reply → only then ALL_DONE.
"""


# ── Twist detection keywords ──────────────────────────────────────────────
TWIST_KEYWORDS = [
    "if my friend", "if he can't", "if she can't",
    "if they can't", "if that doesn't work", "if the person",
    "if the order", "if it doesn't", "if not",
    "reschedule", "accept any suggested", "proposes",
    "can't make it", "declines", "an alternative",
    "if there's", "if you can't", "handle the twist",
    "let me know when", "send him an email after",
    "after scheduling", "after you",
]


# ── Twist reminder template ───────────────────────────────────────────────
TWIST_REMINDER = (
    "\n\n⚠️ CRITICAL: This task has a CONDITIONAL SECOND PHASE (twist).\n"
    "After completing primary actions, this EXACT sequence is REQUIRED:\n"
    "  1. Notify user that primary actions are done (op-001)\n"
    "  2. IMMEDIATELY call op-000 with timeout_seconds:60 to wait for reply\n"
    "  3. Process the reply (email, chat, notification)\n"
    "  4. Execute any follow-up actions requested in the reply\n"
    "  5. Only then output ALL_DONE\n"
    "Do NOT call ALL_DONE before completing the full sequence.\n"
    "Do NOT call op-001 more than ONCE — repeating it will NOT get a reply."
)


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def build_agent_system_prompt(
    tool_text: str,
    experience_section: str = "",
) -> str:
    """Build the full system prompt for a GAIA2 ARE task.

    Args:
        tool_text: Pre-built tool list string (one tool per line: "op-XXX: desc | params")
        experience_section: Optional injected experience text (already formatted
                           by build_augmented_prompt from injection.py)

    Returns:
        Complete system prompt string ready to send to the LLM.
    """
    prompt = GAIA2_SYSTEM_PROMPT_TEMPLATE + f"OPERATIONS:\n{tool_text}\n\nSTART NOW. Output ONLY: NEXT_OP + PARAMS."

    if experience_section:
        prompt += f"\n\nRELEVANT EXPERIENCE:\n{experience_section}"

    return prompt


def detect_twist(task_content: str, task_desc: str = "") -> bool:
    """Detect whether a task has a conditional second phase (twist).

    Checks both the actual task content (from ARE environment) and the
    task description (from metadata).
    """
    task_lower = task_content.lower()
    desc_lower = task_desc.lower()
    return any(kw in task_lower for kw in TWIST_KEYWORDS) or \
           any(kw in desc_lower for kw in TWIST_KEYWORDS)


def get_twist_reminder() -> str:
    """Get the twist protocol reminder for the initial user prompt."""
    return TWIST_REMINDER
