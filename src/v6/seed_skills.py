"""Seed Skills — Task-specific strategies extracted from failed execution traces.

These are NOT part of the system prompt (universal rules). They are injected
via the experience/skill layer, enabling clean ablation:
  - Group A (no augmentation): system prompt only → baseline
  - Group B/C (with augmentation): system prompt + these skills → our method

Theoretical role:
  - These skills reduce δ_sem by providing high-relevance retrieval targets
  - Their structured format reduces δ_att (format clarity)
  - They encode CONCRETE lessons from failures, not generic rules

Design principle: Each seed skill is a pre-refined Experience with:
  - ai_refined=True (passes quality gate in injection.py)
  - causal_lesson explaining WHY the strategy works
  - generalized_steps providing actionable instructions
  - transferability indicating when to apply
"""
from __future__ import annotations
import time
from .experience import Experience, ExperienceLibrary


# ══════════════════════════════════════════════════════════════════════════════
#  Seed Skill Definitions
#  Source: Extracted from GAIA2 failed traces (tasks 0601, 0602, 0603)
# ══════════════════════════════════════════════════════════════════════════════

SEED_SKILLS: list[dict] = [
    # ── Skill 1: Contact Search Efficiency ──────────────────────────────────
    # Source: Task 0601 — agent searched "Film Producer Stockholm" 15+ times
    {
        "task_id": "seed_contact_search_v1",
        "task_desc": "Find a specific contact (e.g., a Film Producer friend in Stockholm) and use their info for calendar/email tasks",
        "outcome": "success",
        "score": 0.95,
        "failure_taxonomy": {
            "ai_refined": True,
            "causal_lesson": (
                "Contact search APIs use keyword matching, not semantic understanding. "
                "Long phrases like 'Film Producer Stockholm' match NOTHING because the API "
                "looks for exact substring matches. Short single-word keywords ('Film', 'Producer') "
                "are far more likely to hit. Also, the two contact apps (list_contacts variants) "
                "share the same underlying data — switching between them wastes turns."
            ),
            "generalized_steps": (
                "1. Search with the SHORTEST distinctive keyword from the task description\n"
                "   - 'Film Producer in Stockholm' → search 'Film' or 'Producer'\n"
                "   - 'my friend Nalani' → search 'Nalani'\n"
                "   - 'a lawyer in NYC' → search 'lawyer'\n"
                "2. If 0 results: try ONE alternate keyword (job title OR city, not both)\n"
                "3. If still 0: browse contacts with offset:0, scan ALL fields (name, job_title, city)\n"
                "4. If still not found: browse offset:10, then STOP\n"
                "5. MAX 5 total operations for contact search. Use whatever info you have.\n"
                "NEVER: use multi-word phrases | browse >3 pages | switch contact apps | repeat keywords"
            ),
            "avoidance_note": (
                "FATAL PATTERN: Searching 'Film Producer Stockholm' → 0 results → "
                "paginating through 70+ contacts → context exhaustion → task fails. "
                "This wastes 15+ turns and leaves no budget for the actual task."
            ),
            "transferability": "Any task requiring contact lookup before calendar/email actions",
            "evolution_insight": "Discovered through 3 failed attempts that all used long search phrases",
        },
        "action_commands": [],
        "tool_sequence": ["search_contacts", "list_contacts"],
        "missing_steps": [],
        "extra_steps": [],
        "failure_reason": "",
    },

    # ── Skill 2: Calendar Date-Range Query ──────────────────────────────────
    # Source: Task 0602 — agent used text search for "Thursday dinner" (failed)
    {
        "task_id": "seed_calendar_date_query_v1",
        "task_desc": "Find and modify calendar events on a specific day (e.g., Thursday's dinner, Saturday's appointments)",
        "outcome": "success",
        "score": 0.90,
        "failure_taxonomy": {
            "ai_refined": True,
            "causal_lesson": (
                "Calendar APIs have two query modes: text search (searches event titles/descriptions) "
                "and date-range query (returns all events in a time window). When looking for events "
                "on a specific DAY (Thursday, Saturday, etc.), date-range query is the correct choice. "
                "Text search for 'Thursday dinner' fails because event titles rarely contain day names."
            ),
            "generalized_steps": (
                "1. Determine the target date (e.g., 'this Thursday' = calculate actual date)\n"
                "2. Use date-range query: start_datetime=YYYY-MM-DD 00:00:00, end_datetime=YYYY-MM-DD 23:59:59\n"
                "3. From results, identify the target event by title/time/attendees\n"
                "4. Use the event_id for subsequent operations (delete, modify)\n"
                "NEVER: use text search for date-based lookups (it searches titles, not dates)"
            ),
            "avoidance_note": (
                "Text search for 'Thursday dinner' or 'Saturday meeting' returns 0 results "
                "because event titles are like 'Dinner with Friends', not 'Thursday dinner'."
            ),
            "transferability": "Any task involving finding/modifying calendar events by day of week",
            "evolution_insight": "First attempt used text search (2 wasted turns), second used date-range (immediate success)",
        },
        "action_commands": [],
        "tool_sequence": ["get_calendar_events_by_date"],
        "missing_steps": [],
        "extra_steps": [],
        "failure_reason": "",
    },

    # ── Skill 3: Twist Handling Pattern ─────────────────────────────────────
    # Source: All 3 tasks — agent never waited for async replies
    {
        "task_id": "seed_twist_handling_v1",
        "task_desc": "Complete a multi-phase task where the second phase depends on someone's reply (e.g., rescheduling if friend can't make it)",
        "outcome": "success",
        "score": 0.85,
        "failure_taxonomy": {
            "ai_refined": True,
            "causal_lesson": (
                "GAIA2 tasks with 'If X, then Y' patterns have TWO phases: "
                "(1) Execute primary actions (create event, send email) "
                "(2) Wait for a reply and handle the twist (reschedule, cancel, modify). "
                "The twist phase requires WAITING for an async notification — the agent must "
                "explicitly call the wait operation after completing phase 1. Without waiting, "
                "the agent misses the reply and scores 0 on the entire twist portion (~50% of points)."
            ),
            "generalized_steps": (
                "1. Complete ALL primary actions (create event, send email, notify user)\n"
                "2. Notify the user that primary actions are done (op-001)\n"
                "3. WAIT for async reply: op-000 with timeout_seconds:60\n"
                "4. When notification arrives: read it carefully, extract the proposed change\n"
                "5. Execute the change (delete old event, create new one, send confirmation)\n"
                "6. If the change involves other people: email them about the update\n"
                "7. Only output ALL_DONE after handling the twist completely"
            ),
            "avoidance_note": (
                "FATAL PATTERN: Completing phase 1 → immediately outputting ALL_DONE → "
                "missing the twist entirely → losing 50%+ of the score. "
                "The twist is where most points come from."
            ),
            "transferability": "Any task with conditional second phase ('If X happens, then Y')",
            "evolution_insight": "All 3 initial attempts scored <40% because they never reached the twist phase",
        },
        "action_commands": [],
        "tool_sequence": ["notify_user", "wait_for_notification"],
        "missing_steps": [],
        "extra_steps": [],
        "failure_reason": "",
    },

    # ── Skill 4: Batch Operations over Individual Queries ───────────────────
    # Source: Task 0603 — agent checked 6 discount codes one by one
    {
        "task_id": "seed_batch_operations_v1",
        "task_desc": "Find a specific item from a list (e.g., a discount code with 52% off, a product under $50)",
        "outcome": "success",
        "score": 0.90,
        "failure_taxonomy": {
            "ai_refined": True,
            "causal_lesson": (
                "When looking for a specific item from a collection (discount codes, products, etc.), "
                "listing ALL items first and filtering client-side is far more efficient than "
                "querying each item individually. Individual queries cost N operations (one per item), "
                "while a list-all query costs 1-2 operations regardless of collection size."
            ),
            "generalized_steps": (
                "1. Use a LIST/GET-ALL operation to retrieve the entire collection\n"
                "2. Scan results for the target item (matching percentage, price, name, etc.)\n"
                "3. If collection is paginated: retrieve 2-3 pages max, then work with what you have\n"
                "NEVER: query items one-by-one (wastes N turns instead of 1-2)"
            ),
            "avoidance_note": (
                "Checking discount codes individually (6 API calls) when a single list-all "
                "call would return all codes with their percentages visible."
            ),
            "transferability": "Any task requiring finding a specific item from a collection",
            "evolution_insight": "Task 3 wasted 6 turns checking codes individually; list-all would have taken 1 turn",
        },
        "action_commands": [],
        "tool_sequence": ["list_items", "filter_results"],
        "missing_steps": [],
        "extra_steps": [],
        "failure_reason": "",
    },

    # ── Skill 5: Cross-Task Independence ────────────────────────────────────
    # Source: Task 0602 retry — agent searched for "Nalani" from previous task
    {
        "task_id": "seed_task_independence_v1",
        "task_desc": "Execute a task that involves people or entities mentioned in the task description only",
        "outcome": "success",
        "score": 0.85,
        "failure_taxonomy": {
            "ai_refined": True,
            "causal_lesson": (
                "Each task is completely independent. Names, IDs, and context from previous "
                "tasks are INVALID in the current task. The agent must derive ALL information "
                "from the current task description and tool results only. Searching for names "
                "from previous tasks wastes turns and may lead to incorrect actions."
            ),
            "generalized_steps": (
                "1. Read the current task description carefully\n"
                "2. Identify ONLY the people/entities mentioned in THIS task\n"
                "3. Search for contacts/items based on THIS task's description only\n"
                "4. Never assume information carries over between tasks\n"
                "NEVER: search for names/IDs from previous tasks"
            ),
            "avoidance_note": (
                "Searching for 'Nalani' in a task that doesn't mention Nalani — "
                "this was a hallucination from a previous task's context bleeding through."
            ),
            "transferability": "All tasks — universal principle of task independence",
            "evolution_insight": "Observed context pollution between consecutive tasks in the same session",
        },
        "action_commands": [],
        "tool_sequence": [],
        "missing_steps": [],
        "extra_steps": [],
        "failure_reason": "",
    },
]


def inject_seed_skills(library: ExperienceLibrary) -> int:
    """Inject seed skills into the experience library.

    These seeds provide the initial skill base that enables the agent to
    avoid known failure patterns from the very first task. They are:
    - Pre-refined (ai_refined=True) → pass quality gates in injection.py
    - Structured with causal lessons → high format clarity (low δ_att)
    - Targeted by task_desc similarity → retrieved only for relevant tasks (low δ_sem)

    Returns the number of seeds injected.
    """
    injected = 0
    existing_ids = {exp.task_id for exp in library.experiences}

    for skill_data in SEED_SKILLS:
        if skill_data["task_id"] in existing_ids:
            continue  # Don't duplicate

        exp = Experience(
            task_id=skill_data["task_id"],
            task_desc=skill_data["task_desc"],
            tool_sequence=skill_data["tool_sequence"],
            action_commands=skill_data["action_commands"],
            outcome=skill_data["outcome"],
            score=skill_data["score"],
            missing_steps=skill_data["missing_steps"],
            extra_steps=skill_data["extra_steps"],
            failure_reason=skill_data["failure_reason"],
            failure_taxonomy=skill_data["failure_taxonomy"],
            timestamp=time.time(),
        )
        library.record(exp)
        injected += 1

    return injected
