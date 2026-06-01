"""
SkillForge V6 — EvoMem + Applicability Gate + Cost-Aware Injection

Inspired by TRACE严选评测 findings:
  1. "有Skill不一定更好" → Applicability Gate: decide BEFORE injection whether augmentation helps
  2. "Skill虹吸" → Anti-siphon: simple tasks should NOT trigger experience injection
  3. "48% token增加" → Cost-aware: track token overhead, enforce budget
  4. "稳定性来自工具链" → Structured failure taxonomy: distinguish model vs tool-chain failures
  5. "推理强度收益不均匀" → Adaptive injection depth based on task complexity

Core principle remains: ALWAYS ADD INFORMATION, NEVER REDUCE.
But V6 adds: "Only add information WHEN IT HELPS" — the gate decides.

Architecture:
  Phase 1: Applicability Assessment
    - Score task complexity (simple → no augment; complex → full augment)
    - Score experience relevance (low relevance → skip; high → inject)
    
  Phase 2: Cost-Aware Injection  
    - Budget: max tokens for experience section
    - Track injection overhead per task
    - Adaptive: if past injections increased cost without improving outcome, reduce

  Phase 3: Structured Experience with Failure Taxonomy
    - model_failure: reasoning/planning errors
    - tool_failure: CLI/API errors, timeouts, format issues
    - task_mismatch: agent misunderstood the task
    - over_action: agent did too much (TRACE finding #1)
"""
from __future__ import annotations
import json
import os
import hashlib
import time
import re
from dataclasses import dataclass, field
from typing import Any


# ─── Enhanced Experience Store ────────────────────────────────────────────

@dataclass
class FailureTaxonomy:
    """Structured failure analysis (TRACE finding #6)."""
    category: str = ""          # model_failure | tool_failure | task_mismatch | over_action
    root_cause: str = ""        # Specific root cause description
    is_tool_chain: bool = False # True if failure came from tool/env, not model reasoning
    recoverable: bool = True    # Could a retry with more info fix this?


@dataclass 
class Experience:
    """Enhanced experience record with version history, cost tracking and failure taxonomy."""
    task_id: str
    task_desc: str
    tool_sequence: list[str]
    action_commands: list[str]
    outcome: str                    # "success" | "partial" | "failure"
    score: float
    missing_steps: list[str]
    extra_steps: list[str]
    failure_reason: str
    # V6 additions
    failure_taxonomy: dict = field(default_factory=dict)  # FailureTaxonomy + AI refinement
    token_cost: int = 0             # Total tokens consumed
    time_cost: float = 0.0          # Total wall-clock time (seconds)
    task_complexity: str = ""       # "simple" | "moderate" | "complex"
    augmentation_used: str = ""     # What augmentation was injected (empty = none)
    augmentation_helped: bool | None = None  # Did augmentation improve outcome?
    # Version history (EvoMem-inspired patch tracking)
    version: int = 1                # Which attempt is this? (1st, 2nd, 3rd...)
    patch_history: list = field(default_factory=list)  # [{version, delta, lesson_delta}]
    timestamp: float = 0.0


class ExperienceLibrary:
    """Enhanced library with cost tracking and applicability analysis."""
    
    def __init__(self):
        self.experiences: list[Experience] = []
        # V6: Track augmentation effectiveness
        self._augment_stats: dict[str, dict] = {}  # task_type → {helped, hurt, neutral}
    
    def record(self, exp: Experience):
        self.experiences.append(exp)
        # Update augmentation effectiveness tracking
        if exp.augmentation_used:
            key = exp.task_complexity or "unknown"
            if key not in self._augment_stats:
                self._augment_stats[key] = {"helped": 0, "hurt": 0, "neutral": 0}
            if exp.augmentation_helped is True:
                self._augment_stats[key]["helped"] += 1
            elif exp.augmentation_helped is False:
                self._augment_stats[key]["hurt"] += 1
            else:
                self._augment_stats[key]["neutral"] += 1
    
    def retrieve_similar(self, task_desc: str, top_k: int = 3,
                         outcome_filter: str | None = None,
                         exclude_tool_failures: bool = False) -> list[Experience]:
        """Retrieve similar experiences with optional tool-failure filtering."""
        candidates = self.experiences
        if outcome_filter:
            candidates = [e for e in candidates if e.outcome == outcome_filter]
        if exclude_tool_failures:
            candidates = [e for e in candidates 
                         if not e.failure_taxonomy.get("is_tool_chain", False)]
        
        if not candidates:
            return []
        
        task_words = set(task_desc.lower().split())
        # Remove common stop words for better matching
        stop_words = {"the", "a", "an", "to", "and", "or", "in", "on", "at", "for", 
                      "of", "with", "is", "are", "was", "were", "be", "been", "being",
                      "that", "this", "it", "my", "all", "i", "me"}
        task_words -= stop_words
        
        scored = []
        for exp in candidates:
            exp_words = set(exp.task_desc.lower().split()) - stop_words
            if not exp_words:
                continue
            overlap = len(task_words & exp_words) / max(len(task_words | exp_words), 1)
            scored.append((overlap, exp))
        
        scored.sort(key=lambda x: -x[0])
        return [exp for _, exp in scored[:top_k]]
    
    def get_augmentation_effectiveness(self, task_complexity: str) -> float:
        """Get historical effectiveness rate for this complexity level.
        
        Returns: ratio of "helped" / total augmented tasks (0-1).
        """
        stats = self._augment_stats.get(task_complexity, {})
        total = stats.get("helped", 0) + stats.get("hurt", 0) + stats.get("neutral", 0)
        if total < 3:  # Not enough data → default to augmenting
            return 0.7
        return stats.get("helped", 0) / total
    
    def get_avg_token_overhead(self) -> float:
        """Get average token overhead ratio for augmented vs non-augmented tasks."""
        augmented = [e for e in self.experiences if e.augmentation_used and e.token_cost > 0]
        non_augmented = [e for e in self.experiences if not e.augmentation_used and e.token_cost > 0]
        
        if not augmented or not non_augmented:
            return 1.0
        
        avg_aug = sum(e.token_cost for e in augmented) / len(augmented)
        avg_no = sum(e.token_cost for e in non_augmented) / len(non_augmented)
        return avg_aug / avg_no if avg_no > 0 else 1.0
    
    def get_successful(self) -> list[Experience]:
        return [e for e in self.experiences if e.outcome == "success"]
    
    def get_failed(self) -> list[Experience]:
        return [e for e in self.experiences if e.outcome in ("failure", "partial")]
    
    def to_dict(self) -> dict:
        return {
            "experiences": [
                {
                    "task_id": e.task_id, "task_desc": e.task_desc,
                    "tool_sequence": e.tool_sequence, "action_commands": e.action_commands,
                    "outcome": e.outcome, "score": e.score,
                    "missing_steps": e.missing_steps, "extra_steps": e.extra_steps,
                    "failure_reason": e.failure_reason,
                    "failure_taxonomy": e.failure_taxonomy,
                    "token_cost": e.token_cost, "time_cost": e.time_cost,
                    "task_complexity": e.task_complexity,
                    "augmentation_used": e.augmentation_used,
                    "augmentation_helped": e.augmentation_helped,
                    "timestamp": e.timestamp,
                    "version": e.version,
                    "patch_history": e.patch_history,
                }
                for e in self.experiences
            ],
            "augment_stats": self._augment_stats,
        }
    
    def from_dict(self, data: dict | list):
        # Backward compat with V5 format (plain list)
        if isinstance(data, list):
            for d in data:
                d.setdefault("failure_taxonomy", {})
                d.setdefault("token_cost", 0)
                d.setdefault("time_cost", 0.0)
                d.setdefault("task_complexity", "")
                d.setdefault("augmentation_used", "")
                d.setdefault("augmentation_helped", None)
                d.setdefault("version", 1)
                d.setdefault("patch_history", [])
                self.experiences.append(Experience(**d))
        else:
            for d in data.get("experiences", []):
                d.setdefault("version", 1)
                d.setdefault("patch_history", [])
                self.experiences.append(Experience(**d))
            self._augment_stats = data.get("augment_stats", {})
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def load(self, path: str):
        if os.path.exists(path):
            with open(path) as f:
                self.from_dict(json.load(f))


# ─── Applicability Gate (TRACE findings #1, #2) ──────────────────────────

def assess_task_complexity(task_desc: str) -> str:
    """Classify task complexity to decide augmentation depth.

    TRACE finding #7: 推理强度收益不均匀
    - Simple tasks (single action, clear intent) → likely no augmentation needed
    - Moderate tasks (2-3 steps, some ambiguity) → light augmentation
    - Complex tasks (multi-step, constraints, cross-app) → full augmentation
    """
    indicators_complex = [
        # Multi-step indicators
        r'\band\b.*\band\b',           # Multiple "and" conjunctions
        r'after.*(?:then|also|next)',   # Sequential steps
        r'(?:all|every|each)\b',       # Iteration over sets
        # Cross-app indicators
        r'(?:email|message|chat).*(?:calendar|event|schedule)',
        r'(?:calendar|event).*(?:email|message|contact)',
        # Constraint indicators
        r'(?:except|unless|only if|but not)',
        r'(?:last|first|most recent|latest)',
    ]

    indicators_simple = [
        r'^(?:get|show|list|find|search)\b',  # Simple retrieval
        r'^(?:what|who|when|where)\b',         # Simple questions
    ]

    desc_lower = task_desc.lower()

    # Count complexity signals
    complex_score = sum(1 for p in indicators_complex if re.search(p, desc_lower))
    simple_score = sum(1 for p in indicators_simple if re.search(p, desc_lower))

    # Word count as proxy for complexity
    word_count = len(task_desc.split())
    if word_count > 80:
        complex_score += 2
    elif word_count > 40:
        complex_score += 1
    elif word_count < 15:
        simple_score += 1

    # Sentence count
    sentence_count = len(re.split(r'[.!?\n]', task_desc))
    if sentence_count >= 4:
        complex_score += 1

    # Gaia2 tasks are typically multi-step → lower the threshold
    if complex_score >= 2:
        return "complex"
    elif simple_score >= 1 and complex_score == 0 and word_count < 20:
        return "simple"
    else:
        return "moderate"


def should_augment(task_desc: str, library: ExperienceLibrary,
                   relevance_threshold: float = 0.15) -> tuple[bool, str]:
    """Applicability gate: decide whether to inject experience augmentation.
    
    TRACE findings:
    - #1: Skill不一定更好 → only augment when there's genuine signal
    - #2: Skill虹吸 → don't augment simple tasks just because keywords match
    
    Returns:
        (should_augment: bool, reason: str)
    """
    complexity = assess_task_complexity(task_desc)
    
    # Simple tasks: default to no augmentation (anti-siphon)
    if complexity == "simple":
        return False, f"simple_task (complexity={complexity})"
    
    # Check if we have relevant experiences
    candidates = library.retrieve_similar(task_desc, top_k=1)
    if not candidates:
        return False, "no_relevant_experiences"
    
    best_relevance = 0.0
    task_words = set(task_desc.lower().split())
    stop_words = {"the", "a", "an", "to", "and", "or", "in", "on", "at", "for",
                  "of", "with", "is", "are", "was", "were", "be", "been", "being",
                  "that", "this", "it", "my", "all", "i", "me"}
    task_words -= stop_words
    
    for exp in candidates:
        exp_words = set(exp.task_desc.lower().split()) - stop_words
        if exp_words:
            overlap = len(task_words & exp_words) / max(len(task_words | exp_words), 1)
            best_relevance = max(best_relevance, overlap)
    
    if best_relevance < relevance_threshold:
        return False, f"low_relevance ({best_relevance:.3f} < {relevance_threshold})"
    
    # Check historical effectiveness for this complexity level
    effectiveness = library.get_augmentation_effectiveness(complexity)
    if effectiveness < 0.3:  # Augmentation historically hurt more than helped
        return False, f"historically_ineffective ({effectiveness:.1%} help rate for {complexity})"
    
    return True, f"approved (complexity={complexity}, relevance={best_relevance:.3f}, effectiveness={effectiveness:.1%})"


# ─── Cost-Aware Prompt Augmentation (TRACE finding #3) ────────────────────

def estimate_token_count(text: str) -> int:
    """Rough token count estimation (1 token ≈ 4 chars for English)."""
    return len(text) // 4


def format_success_experience(exp: Experience, budget_tokens: int = 500) -> str:
    """Format successful experience — use AI-refined version if available."""
    taxonomy = exp.failure_taxonomy
    
    # Use AI-refined generalized steps if available
    if taxonomy.get("ai_refined") and taxonomy.get("generalized_steps"):
        result = f"""[Successful approach for similar task]
Task: {exp.task_desc[:150]}
Causal lesson: {taxonomy.get('causal_lesson', '')}
Generalized steps:
{taxonomy['generalized_steps']}
Transferable to: {taxonomy.get('transferability', '')}
Original score: {exp.score:.0%}"""
    else:
        # Fallback: raw steps
        max_steps = 8 if exp.task_complexity == "complex" else 5
        steps = "\n".join(f"  {i+1}. {cmd}" for i, cmd in enumerate(exp.action_commands[:max_steps]))
        result = f"""[Successful approach for similar task]
Task: {exp.task_desc[:150]}
Steps taken:
{steps}
Result: Task completed successfully (score: {exp.score:.0%})."""
    
    # Truncate if over budget
    if estimate_token_count(result) > budget_tokens:
        key_steps = "\n".join(f"  {i+1}. {cmd}" for i, cmd in enumerate(exp.action_commands[:3]))
        result = f"""[Successful approach for similar task]
Task: {exp.task_desc[:100]}
Key steps: {key_steps}"""
    
    return result


def format_failure_experience(exp: Experience, budget_tokens: int = 400) -> str:
    """Format failed experience — use AI-refined avoidance notes if available."""
    taxonomy = exp.failure_taxonomy
    
    # Use AI-refined version if available
    if taxonomy.get("ai_refined") and taxonomy.get("causal_lesson"):
        lines = ["[⚠️ Lesson from similar failed task]"]
        lines.append(f"Task: {exp.task_desc[:120]}")
        lines.append(f"Why it failed: {taxonomy['causal_lesson']}")
        if taxonomy.get("avoidance_note"):
            lines.append(f"Avoid: {taxonomy['avoidance_note']}")
        if taxonomy.get("generalized_steps"):
            lines.append(f"What was attempted:\n{taxonomy['generalized_steps']}")
        if exp.missing_steps:
            lines.append("Steps that were MISSING:")
            for step in exp.missing_steps[:4]:
                lines.append(f"  - {step}")
        lines.append(f"Transferable to: {taxonomy.get('transferability', '')}")
    else:
        # Fallback: raw format
        lines = ["[⚠️ Warning from similar failed task]"]
        lines.append(f"Task: {exp.task_desc[:120]}")
        
        category = taxonomy.get("category", "")
        if category == "tool_failure":
            lines.append(f"⚠️ Tool/environment issue: {taxonomy.get('root_cause', exp.failure_reason)}")
        elif category == "over_action":
            lines.append(f"⚠️ Over-action: Agent did too much. {taxonomy.get('root_cause', '')}")
        elif category == "task_mismatch":
            lines.append(f"⚠️ Task misunderstood: {taxonomy.get('root_cause', exp.failure_reason)}")
        else:
            if exp.failure_reason:
                lines.append(f"What went wrong: {exp.failure_reason}")
        
        if exp.missing_steps:
            lines.append("Steps that were MISSING (make sure to do these):")
            for step in exp.missing_steps[:4]:
                lines.append(f"  - {step}")
    
    result = "\n".join(lines)
    if estimate_token_count(result) > budget_tokens:
        result = "\n".join(lines[:5])
    
    return result


def classify_task_type(task_desc: str, expected: str = "", metadata: dict | None = None) -> str:
    """Auto-classify task type based on structural features of the task.
    
    This classifier works on ANY new benchmark without configuration by analyzing:
    1. Task description structure (imperative commands vs questions)
    2. Expected answer structure (action sequence vs short text)
    3. Metadata hints (task_type field, tools, environment)
    
    Types:
    - "agentic": multi-step tool/action tasks → full experience injection
    - "qa": single-turn knowledge/reasoning → lightweight reasoning hints only
    - "embodied": sequential physical actions in environment → action pattern injection
    
    Returns one of: "agentic", "qa", "embodied"
    """
    metadata = metadata or {}
    desc_lower = task_desc.lower()
    
    # ─── Signal 1: Metadata hints (strongest signal) ─────────────────
    if metadata:
        # Explicit task_type in metadata
        meta_type = metadata.get("task_type", "").lower()
        if any(kw in meta_type for kw in ("pick", "place", "look_at", "clean", "heat", "cool")):
            return "embodied"
        if metadata.get("scenario_path") or metadata.get("tools") or metadata.get("apps"):
            return "agentic"
        if metadata.get("benchmark") in ("gaia2", "swebench"):
            return "agentic"
    
    # ─── Signal 2: Expected answer structure ──────────────────────────
    if expected:
        expected_lower = expected.lower()
        # Action sequence indicators
        if "->" in expected or expected.count("\n") >= 3:
            # Multi-step expected = likely action sequence
            action_verbs = ["go to", "take", "put", "use", "open", "pick", "move", "clean", "heat"]
            if any(v in expected_lower for v in action_verbs):
                return "embodied"
            return "agentic"
        # Short factual answer
        if len(expected.split()) < 20:
            return "qa"
    
    # ─── Signal 3: Description structure analysis ─────────────────────
    
    # Imperative multi-command detection (agentic)
    imperative_verbs = [
        r'\b(?:create|cancel|send|book|add|remove|save|delete|update|schedule|reply)\b',
        r'\b(?:search|buy|order|transfer|forward|move|set|change)\b',
    ]
    imperative_count = sum(1 for p in imperative_verbs if re.search(p, desc_lower))
    
    # Question detection (qa)
    question_patterns = [
        r'^(?:answer|what|who|when|where|which|how|find|name|list)\b',
        r'(?:question|following question|based on)',
        r'\?\s*$',
    ]
    question_score = sum(1 for p in question_patterns if re.search(p, desc_lower))
    
    # Physical action detection (embodied)
    physical_patterns = [
        r'\b(?:pick up|put down|go to|move to|open door|close door)\b',
        r'\b(?:take .* from|place .* on|examine .* with)\b',
        r'\b(?:heat|cool|clean|slice|toggle)\b.*\b(?:with|using)\b',
    ]
    physical_score = sum(1 for p in physical_patterns if re.search(p, desc_lower))
    
    # Cross-app/tool indicators (agentic)
    tool_patterns = [
        r'\b(?:calendar|email|contact|message|shopping|terminal|api|endpoint)\b',
        r'\b(?:file|database|server|deploy|commit|pull request)\b',
    ]
    tool_score = sum(1 for p in tool_patterns if re.search(p, desc_lower))
    
    # ─── Decision ─────────────────────────────────────────────────────
    if physical_score >= 2:
        return "embodied"
    if tool_score >= 1 or imperative_count >= 2:
        return "agentic"
    if question_score >= 1 and imperative_count == 0:
        return "qa"
    
    # Fallback: if description is long with multiple sentences → likely agentic
    sentence_count = len(re.split(r'[.!?\n]', task_desc))
    word_count = len(task_desc.split())
    if sentence_count >= 3 and word_count > 50 and imperative_count >= 1:
        return "agentic"
    
    # Default: qa (safest — lightweight hints won't hurt)
    return "qa"


def _build_qa_hint(task_desc: str, library: ExperienceLibrary, token_budget: int = 400) -> str:
    """Build lightweight reasoning hints for QA tasks.
    
    Instead of injecting full tool-calling experiences (which mislead QA models),
    extract only the REASONING STRATEGY from similar experiences:
    - How to parse the question
    - What type of answer is expected (name, number, date, etc.)
    - Common pitfalls for this question type
    
    Max ~100 tokens to avoid overwhelming the model.
    """
    # Find similar experiences with AI-refined causal lessons
    candidates = library.retrieve_similar(task_desc, top_k=5)
    if not candidates:
        return ""
    
    # Tool-related keywords to filter out (these mislead QA models)
    tool_noise = {"websearch", "webfetch", "tool", "bash", "cli", "api", "curl",
                  "search for", "fetch", "download", "execute", "invoke"}
    
    hints = []
    for exp in candidates:
        ft = exp.failure_taxonomy
        if not ft.get("ai_refined"):
            continue
        
        # Try transferability first (most abstract/useful for QA)
        transferability = ft.get("transferability", "")
        lesson = ft.get("causal_lesson", "")
        avoid = ft.get("avoidance_note", "")
        
        # Filter: skip anything mentioning tools
        for text in [transferability, lesson, avoid]:
            if not text:
                continue
            text_lower = text.lower()
            if any(kw in text_lower for kw in tool_noise):
                continue
            if len(text) > 30:  # Skip trivially short
                hints.append(text[:150])
                break
    
    if not hints:
        # Fallback: extract answer-type hint from task description patterns
        return ""
    
    # Deduplicate
    seen = set()
    unique_hints = []
    for h in hints:
        key = h[:50].lower()
        if key not in seen:
            seen.add(key)
            unique_hints.append(h)
    
    if not unique_hints:
        return ""
    
    # Keep it very short — just a reasoning nudge
    hint_text = "\n".join(f"- {h}" for h in unique_hints[:2])
    result = f"## Reasoning Hints\n{hint_text}"
    
    # Hard cap
    if estimate_token_count(result) > token_budget:
        result = result[:token_budget * 4]
    
    return result


def build_augmented_prompt(task_desc: str, library: ExperienceLibrary,
                           token_budget: int = 2000,
                           top_k_success: int = 2, top_k_failure: int = 2,
                           expected: str = "", metadata: dict | None = None) -> str:
    """Build task-type-aware experience-augmented prompt section.
    
    V6.1 improvements:
    1. Task-type auto-classification: agentic vs QA vs embodied (works on any benchmark)
    2. Agentic tasks: full experience injection (success + failure + generalized steps)
    3. QA tasks: lightweight reasoning hints only (no tool steps, max ~100 tokens)
    4. Applicability gate: skip augmentation for simple/irrelevant tasks
    5. Token budget: never exceed budget
    6. Exclude tool-chain failures: only learn from model-recoverable failures
    """
    # Task type determines injection strategy
    task_type = classify_task_type(task_desc, expected=expected, metadata=metadata)
    
    # QA tasks: lightweight hints only (avoid overwhelming the model)
    if task_type == "qa":
        return _build_qa_hint(task_desc, library, token_budget=min(token_budget, 400))
    
    # Agentic/Embodied tasks: full experience injection
    # Phase 1: Applicability gate
    do_augment, reason = should_augment(task_desc, library)
    if not do_augment:
        return f"<!-- augmentation skipped: {reason} -->"
    
    sections = []
    remaining_budget = token_budget
    
    # Phase 2: Inject successes (EvoMem)
    successes = library.retrieve_similar(task_desc, top_k=top_k_success, outcome_filter="success")
    if successes:
        sections.append("## Relevant Experience (from similar successful tasks)\n")
        for exp in successes:
            entry = format_success_experience(exp, budget_tokens=remaining_budget // 3)
            entry_tokens = estimate_token_count(entry)
            if remaining_budget - entry_tokens > 200:  # Keep reserve for failures
                sections.append(entry)
                sections.append("")
                remaining_budget -= entry_tokens
    
    # Phase 3: Inject failures (excluding tool-chain issues)
    failures = library.retrieve_similar(
        task_desc, top_k=top_k_failure, 
        outcome_filter="failure",
        exclude_tool_failures=True  # V6: Don't warn about env issues
    )
    if not failures:
        failures = library.retrieve_similar(
            task_desc, top_k=top_k_failure, outcome_filter="partial",
            exclude_tool_failures=True
        )
    
    if failures and remaining_budget > 200:
        sections.append("## Lessons from Similar Failed Attempts\n")
        for exp in failures:
            entry = format_failure_experience(exp, budget_tokens=remaining_budget // 2)
            entry_tokens = estimate_token_count(entry)
            if remaining_budget - entry_tokens > 0:
                sections.append(entry)
                sections.append("")
                remaining_budget -= entry_tokens
    
    return "\n".join(sections) if sections else ""


# ─── Enhanced Experience Recording ────────────────────────────────────────

def classify_failure(agent_actions: list[dict], oracle_actions: list[dict],
                     score: float, missing: list[str], extra: list[str]) -> FailureTaxonomy:
    """Classify failure into taxonomy (TRACE finding #6).
    
    Categories:
    - tool_failure: CLI errors, timeouts, permission issues
    - over_action: agent did 2x+ more actions than needed
    - task_mismatch: wrong app/function entirely
    - model_failure: correct tools but wrong args/logic
    """
    # Check for tool-chain failures in agent output
    tool_error_patterns = [
        r'error:', r'timeout', r'permission denied', r'not found',
        r'connection refused', r'traceback', r'exception',
    ]
    
    error_count = 0
    for a in agent_actions:
        output = str(a.get("output", "")).lower()
        for pat in tool_error_patterns:
            if re.search(pat, output):
                error_count += 1
                break
    
    if error_count >= 3:
        return FailureTaxonomy(
            category="tool_failure",
            root_cause=f"Multiple tool errors ({error_count} actions had errors)",
            is_tool_chain=True,
            recoverable=True,
        )
    
    # Over-action check (TRACE finding #1)
    if len(agent_actions) > len(oracle_actions) * 2.5 and score < 0.5:
        return FailureTaxonomy(
            category="over_action",
            root_cause=f"Agent used {len(agent_actions)} actions vs {len(oracle_actions)} expected",
            is_tool_chain=False,
            recoverable=True,
        )
    
    # Task mismatch: most oracle steps missing
    if score < 0.3 and len(missing) > len(oracle_actions) * 0.7:
        return FailureTaxonomy(
            category="task_mismatch",
            root_cause=f"Agent missed {len(missing)}/{len(oracle_actions)} required steps",
            is_tool_chain=False,
            recoverable=True,
        )
    
    # Default: model reasoning failure
    return FailureTaxonomy(
        category="model_failure",
        root_cause=f"Correct approach but {len(missing)} missing steps",
        is_tool_chain=False,
        recoverable=True,
    )


def analyze_execution(task_id: str, task_desc: str,
                      agent_actions: list[dict], oracle_actions: list[dict],
                      token_cost: int = 0, time_cost: float = 0.0,
                      augmentation_used: str = "") -> Experience:
    """Analyze execution with V6 enhancements: taxonomy, cost, complexity.
    
    Supports multiple action formats:
    - Gaia2: {"tool": "Bash", "input": {"command": "calendar list"}}
    - Generic: {"tool": "action", "command": "go to desk"}  
    - LLM output: {"tool": "LLM", "output": "response text"}
    - Oracle (Gaia2): {"app": "Calendar", "fn": "list_events", "args": {...}}
    - Oracle (generic): {"tool": "action", "command": "go to desk"}
    """
    # ─── Extract agent tool sequence (format-adaptive) ────────────────
    agent_tools = []
    agent_cmds = []
    for a in agent_actions:
        if a.get('tool') == 'Bash':
            # Gaia2 format
            cmd = a.get('input', {}).get('command', '') if isinstance(a.get('input'), dict) else ''
            agent_cmds.append(cmd)
            clean = re.sub(r'GAIA2_STATE_DIR=\S+\s*', '', cmd).strip()
            parts = clean.split()
            if parts:
                tool_fn = f"{parts[0]} {parts[1]}" if len(parts) > 1 and not parts[1].startswith('-') else parts[0]
                agent_tools.append(tool_fn)
        elif a.get('command'):
            # Generic action format: {"tool": "action", "command": "go to desk"}
            cmd = a['command']
            agent_cmds.append(cmd)
            agent_tools.append(cmd.lower())
        elif a.get('output'):
            # LLM output format: parse response for action-like content
            output = a['output']
            agent_cmds.append(output[:200])
            # Extract action verbs from output
            lines = output.split('\n')
            for line in lines:
                line_clean = line.strip().lower()
                if any(line_clean.startswith(v) for v in ('go to', 'take', 'put', 'use', 'open', 'close', 'clean', 'heat', 'cool')):
                    agent_tools.append(line_clean)
            if not agent_tools:
                agent_tools.append(output[:50].lower())
    
    # ─── Extract oracle tool sequence (format-adaptive) ───────────────
    oracle_tools = []
    for o in oracle_actions:
        if o.get('app') and o.get('fn'):
            # Gaia2 oracle format
            oracle_tools.append(f"{o['app']}.{o['fn']}")
        elif o.get('command'):
            # Generic oracle format: {"tool": "action", "command": "go to desk"}
            oracle_tools.append(o['command'].lower())
        elif o.get('output'):
            # LLM expected output
            oracle_tools.append(o['output'][:50].lower())
    
    # ─── Match (adaptive: exact substring or word overlap) ────────────
    matched = 0
    used = set()
    for ot in oracle_tools:
        ot_parts = set(ot.lower().replace('.', ' ').replace('_', ' ').replace('-', ' ').split())
        for j, at in enumerate(agent_tools):
            if j not in used:
                at_parts = set(at.lower().replace('.', ' ').replace('_', ' ').replace('-', ' ').split())
                # Match if significant word overlap (≥2 words or exact substring)
                overlap = ot_parts & at_parts
                if len(overlap) >= 2 or ot.lower() in at.lower() or at.lower() in ot.lower():
                    matched += 1
                    used.add(j)
                    break
    
    score = matched / len(oracle_tools) if oracle_tools else 0
    
    # Missing steps
    missing = []
    for ot in oracle_tools:
        found = False
        ot_parts = set(ot.lower().replace('.', ' ').replace('_', ' ').replace('-', ' ').split())
        for at in agent_tools:
            at_parts = set(at.lower().replace('.', ' ').replace('_', ' ').replace('-', ' ').split())
            overlap = ot_parts & at_parts
            if len(overlap) >= 2 or ot.lower() in at.lower() or at.lower() in ot.lower():
                found = True
                break
        if not found:
            missing.append(ot)
    
    extra = [at for j, at in enumerate(agent_tools) if j not in used]
    
    # Outcome
    if score >= 1.0:
        outcome = "success"
    elif score >= 0.5:
        outcome = "partial"
    else:
        outcome = "failure"
    
    # V6: Failure taxonomy
    taxonomy = FailureTaxonomy()
    failure_reason = ""
    if outcome != "success":
        taxonomy = classify_failure(agent_actions, oracle_actions, score, missing, extra)
        failure_reason = taxonomy.root_cause or f"Missing {len(missing)} required steps"
    
    # V6: Task complexity
    complexity = assess_task_complexity(task_desc)
    
    return Experience(
        task_id=task_id,
        task_desc=task_desc,
        tool_sequence=agent_tools,
        action_commands=agent_cmds[:15],
        outcome=outcome,
        score=score,
        missing_steps=missing,
        extra_steps=extra[:10],
        failure_reason=failure_reason,
        failure_taxonomy={
            "category": taxonomy.category,
            "root_cause": taxonomy.root_cause,
            "is_tool_chain": taxonomy.is_tool_chain,
            "recoverable": taxonomy.recoverable,
        },
        token_cost=token_cost,
        time_cost=time_cost,
        task_complexity=complexity,
        augmentation_used=augmentation_used,
        augmentation_helped=None,  # Will be set after comparing with baseline
        timestamp=time.time(),
    )


# ─── AI Review Gate (Experience Quality & Generalizability) ───────────────

AI_REVIEW_PROMPT = """You are a skill quality optimizer. Your job is to REFINE an experience record to maximize its value for future similar tasks.

## Experience to Refine
Task: {task_desc}
Outcome: {outcome} (score: {score:.0%})
Steps taken:
{steps}
Missing steps: {missing}
Failure reason: {failure_reason}
{version_history_section}
## Your Job

1. **Generalize** the experience: replace hard-coded IDs, dates, and names with descriptive placeholders, but KEEP all structural details.
2. **Extract causal lesson**: WHY did this succeed/fail? What's the transferable insight?
3. **Preserve all details**: Do NOT remove steps or simplify. Add context, don't subtract.
4. **Learn from version history** (if available): identify what improved across attempts, what patterns of failure recurred, and synthesize a cumulative lesson that incorporates all prior attempts.

## Response Format (JSON only)
{{
  "generalized_steps": "The steps rewritten with placeholders instead of hard-coded values, but same level of detail",
  "causal_lesson": "One clear sentence: why this worked / why this failed, informed by version history if available",
  "avoidance_note": "What to avoid next time (empty if success), incorporating patterns from prior failed versions",
  "transferability": "What types of tasks can benefit from this experience",
  "evolution_insight": "What the version history reveals about solving this type of task (empty if no history)",
  "quality_score": 0-10
}}"""


def _format_patch_history(patch_history: list) -> str:
    """Format patch history into a readable version-diff section for the AI reviewer."""
    if not patch_history:
        return ""
    
    lines = ["\n## Version History (previous attempts on this same/similar task)"]
    lines.append("Each entry shows what changed between consecutive attempts:\n")
    
    for p in patch_history:
        from_v = p.get("from_version", "?")
        to_v = p.get("to_version", "?")
        score_delta = p.get("score_delta", 0)
        outcome_change = p.get("outcome_change", "")
        new_steps = p.get("new_steps", [])
        removed_steps = p.get("removed_steps", [])
        fixed_missing = p.get("fixed_missing", [])
        new_missing = p.get("new_missing", [])
        
        lines.append(f"### v{from_v} → v{to_v} (score: {score_delta:+.0%})")
        if outcome_change:
            lines.append(f"  Outcome: {outcome_change}")
        if fixed_missing:
            lines.append(f"  ✅ Fixed (previously missing): {fixed_missing}")
        if new_missing:
            lines.append(f"  ❌ New gaps: {new_missing}")
        if new_steps:
            lines.append(f"  ➕ Added steps: {new_steps}")
        if removed_steps:
            lines.append(f"  ➖ Removed steps: {removed_steps}")
        lines.append("")
    
    lines.append("Use this history to produce a STRONGER refinement — learn from what improved and what regressed across versions.")
    lines.append("")
    return "\n".join(lines)


def ai_review_experience(exp: Experience, llm_fn=None) -> dict:
    """Version-conditioned AI refinement: optimize generalizability using full evolution trace.
    
    Unlike stateless reflection (Reflexion, ExpeL), this refinement is VERSION-CONDITIONED:
    each iteration sees the full patch_history (v1→v2→...→vN diff chain), enabling the
    refinement to synthesize cumulative lessons across all prior attempts.
    
    This is NOT a gate (accept/reject). It ALWAYS produces a refined version.
    The refinement adds generalization + causal explanation WITHOUT losing any detail.
    
    Args:
        exp: The experience to refine (may contain patch_history from prior versions)
        llm_fn: Callable that takes (prompt: str) -> str (JSON response)
                 If None, returns passthrough (no refinement)
    
    Returns:
        dict with keys: generalized_steps, causal_lesson, avoidance_note,
                       transferability, evolution_insight, quality_score
    """
    if llm_fn is None:
        # No LLM available: passthrough without refinement
        return {
            "generalized_steps": "\n".join(exp.action_commands[:10]),
            "causal_lesson": exp.failure_reason if exp.outcome != "success" else "Completed all required steps",
            "avoidance_note": exp.failure_reason if exp.outcome != "success" else "",
            "transferability": f"Tasks involving: {', '.join(exp.tool_sequence[:3])}",
            "evolution_insight": "",
            "quality_score": int(exp.score * 10),
            "refined": False,
        }
    
    steps_str = "\n".join(f"  {i+1}. {cmd[:120]}" for i, cmd in enumerate(exp.action_commands[:10]))
    missing_str = ", ".join(exp.missing_steps[:5]) if exp.missing_steps else "(none)"
    
    # Version-conditioned: inject patch history diff into the prompt
    version_history_section = _format_patch_history(exp.patch_history)
    
    prompt = AI_REVIEW_PROMPT.format(
        task_desc=exp.task_desc[:200],
        outcome=exp.outcome,
        score=exp.score,
        steps=steps_str,
        missing=missing_str,
        failure_reason=exp.failure_reason or "(none)",
        version_history_section=version_history_section,
    )
    
    try:
        response = llm_fn(prompt)
        import json as _json
        if "{" in response:
            json_str = response[response.index("{"):response.rindex("}") + 1]
            result = _json.loads(json_str)
            result["refined"] = True
            result.setdefault("evolution_insight", "")
            return result
    except Exception:
        pass
    
    # Fallback: passthrough
    return {
        "generalized_steps": "\n".join(exp.action_commands[:10]),
        "causal_lesson": exp.failure_reason if exp.outcome != "success" else "Completed all required steps",
        "avoidance_note": exp.failure_reason if exp.outcome != "success" else "",
        "transferability": f"Tasks involving: {', '.join(exp.tool_sequence[:3])}",
        "evolution_insight": "",
        "quality_score": int(exp.score * 10),
        "refined": False,
    }


# ─── SkillForge V6 Module ─────────────────────────────────────────────────

class SkillForgeV6:
    """
    EvoMem + Applicability Gate + Cost-Aware Injection.
    
    Key improvements over V5:
    1. Applicability Gate: Don't augment simple/irrelevant tasks (anti-siphon)
    2. Cost Tracking: Monitor token/time overhead per task
    3. Failure Taxonomy: Distinguish model vs tool-chain failures
    4. Adaptive: Learn which task types benefit from augmentation
    
    Principle: ADD INFORMATION WHEN IT HELPS.
    """
    
    def __init__(self, library_path: str | None = None, token_budget: int = 2000):
        self.library = ExperienceLibrary()
        self.token_budget = token_budget
        self._gate_log: list[dict] = []  # Log gate decisions for analysis
        
        if library_path:
            self.library.load(library_path)
    
    def get_augmentation(self, task_desc: str) -> tuple[str, dict]:
        """Get experience-based prompt augmentation with gate decision.
        
        Returns:
            (augmentation_text, metadata)
            metadata includes: gated, reason, complexity, token_estimate
        """
        complexity = assess_task_complexity(task_desc)
        do_augment, reason = should_augment(task_desc, self.library)
        
        meta = {
            "gated": not do_augment,
            "reason": reason,
            "complexity": complexity,
            "token_estimate": 0,
        }
        
        if not do_augment:
            self._gate_log.append(meta)
            return "", meta
        
        augmentation = build_augmented_prompt(
            task_desc, self.library,
            token_budget=self.token_budget,
        )
        
        meta["token_estimate"] = estimate_token_count(augmentation)
        self._gate_log.append(meta)
        
        return augmentation, meta
    
    def record_experience(self, task_id: str, task_desc: str,
                          agent_actions: list[dict], oracle_actions: list[dict],
                          token_cost: int = 0, time_cost: float = 0.0,
                          augmentation_used: str = "",
                          baseline_score: float | None = None,
                          llm_reviewer=None):
        """Record experience with version history + AI refinement.
        
        Version history: if a similar task was attempted before, this creates
        a new version with a patch (what changed from last attempt).
        AI refinement: optimizes the experience for generalizability without
        losing any details. Accumulates lessons across versions.
        """
        exp = analyze_execution(
            task_id, task_desc, agent_actions, oracle_actions,
            token_cost=token_cost, time_cost=time_cost,
            augmentation_used=augmentation_used,
        )
        
        # Determine if augmentation helped
        if baseline_score is not None and augmentation_used:
            exp.augmentation_helped = exp.score > baseline_score
        
        # ─── Version History (EvoMem-inspired) ────────────────────────
        # Find previous attempts on the same/similar task
        prev_versions = self._find_previous_versions(task_id, task_desc)
        
        if prev_versions:
            latest = prev_versions[-1]
            exp.version = latest.version + 1
            
            # Compute patch: what changed between versions?
            patch = {
                "from_version": latest.version,
                "to_version": exp.version,
                "score_delta": exp.score - latest.score,
                "outcome_change": f"{latest.outcome} → {exp.outcome}",
                "new_steps": [s for s in exp.tool_sequence if s not in latest.tool_sequence],
                "removed_steps": [s for s in latest.tool_sequence if s not in exp.tool_sequence],
                "fixed_missing": [s for s in latest.missing_steps if s not in exp.missing_steps],
                "new_missing": [s for s in exp.missing_steps if s not in latest.missing_steps],
            }
            
            # Accumulate patch history from previous versions
            exp.patch_history = latest.patch_history + [patch]
        
        # ─── AI Refinement (version-aware) ────────────────────────────
        review_result = ai_review_experience(exp, llm_fn=llm_reviewer)
        
        # Attach refined metadata
        exp.failure_taxonomy["ai_refined"] = review_result.get("refined", False)
        exp.failure_taxonomy["causal_lesson"] = review_result.get("causal_lesson", "")
        exp.failure_taxonomy["avoidance_note"] = review_result.get("avoidance_note", "")
        exp.failure_taxonomy["transferability"] = review_result.get("transferability", "")
        exp.failure_taxonomy["generalized_steps"] = review_result.get("generalized_steps", "")
        exp.failure_taxonomy["evolution_insight"] = review_result.get("evolution_insight", "")
        exp.failure_taxonomy["quality_score"] = review_result.get("quality_score", 0)
        
        # If we have version history, add accumulated lessons
        if exp.patch_history:
            lessons = []
            for p in exp.patch_history:
                if p.get("fixed_missing"):
                    lessons.append(f"v{p['from_version']}→v{p['to_version']}: fixed {p['fixed_missing']}")
                if p.get("score_delta", 0) > 0:
                    lessons.append(f"v{p['from_version']}→v{p['to_version']}: +{p['score_delta']:.0%} by adding {p.get('new_steps', [])[:2]}")
            exp.failure_taxonomy["evolution_trace"] = lessons
        
        self.library.record(exp)
    
    def _find_previous_versions(self, task_id: str, task_desc: str) -> list[Experience]:
        """Find previous attempts on the same or very similar task."""
        # Exact task_id match first
        exact = [e for e in self.library.experiences if e.task_id == task_id]
        if exact:
            return sorted(exact, key=lambda e: e.version)
        
        # Fuzzy match: same task description (high overlap)
        task_words = set(task_desc.lower().split())
        stop_words = {"the", "a", "an", "to", "and", "or", "in", "on", "at", "for",
                      "of", "with", "is", "are", "was", "were", "be", "been",
                      "that", "this", "it", "my", "all", "i", "me"}
        task_words -= stop_words
        
        similar = []
        for exp in self.library.experiences:
            exp_words = set(exp.task_desc.lower().split()) - stop_words
            if exp_words:
                overlap = len(task_words & exp_words) / max(len(task_words | exp_words), 1)
                if overlap > 0.7:  # Very similar task
                    similar.append(exp)
        
        return sorted(similar, key=lambda e: e.version) if similar else []
    
    def save(self, path: str):
        data = self.library.to_dict()
        data["gate_log"] = self._gate_log
        data["token_budget"] = self.token_budget
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load(self, path: str):
        self.library.load(path)
    
    @property
    def stats(self) -> dict:
        gated = sum(1 for g in self._gate_log if g.get("gated"))
        augmented = sum(1 for g in self._gate_log if not g.get("gated"))
        avg_overhead = self.library.get_avg_token_overhead()
        return {
            "total_experiences": len(self.library.experiences),
            "success": len(self.library.get_successful()),
            "failed": len(self.library.get_failed()),
            "gate_decisions": len(self._gate_log),
            "gated_out": gated,
            "augmented": augmented,
            "avg_token_overhead": f"{avg_overhead:.2f}x",
        }
    
    @property
    def cost_report(self) -> dict:
        """TRACE-style cost report: token倍率 and 耗时倍率."""
        augmented = [e for e in self.library.experiences if e.augmentation_used]
        baseline = [e for e in self.library.experiences if not e.augmentation_used]
        
        if not augmented or not baseline:
            return {"token_ratio": "N/A", "time_ratio": "N/A"}
        
        avg_tok_aug = sum(e.token_cost for e in augmented) / len(augmented)
        avg_tok_base = sum(e.token_cost for e in baseline) / len(baseline)
        avg_time_aug = sum(e.time_cost for e in augmented) / len(augmented)
        avg_time_base = sum(e.time_cost for e in baseline) / len(baseline)
        
        return {
            "token_ratio": f"{avg_tok_aug / avg_tok_base:.2f}x" if avg_tok_base > 0 else "N/A",
            "time_ratio": f"{avg_time_aug / avg_time_base:.2f}x" if avg_time_base > 0 else "N/A",
            "avg_tokens_augmented": int(avg_tok_aug),
            "avg_tokens_baseline": int(avg_tok_base),
            "avg_time_augmented": f"{avg_time_aug:.1f}s",
            "avg_time_baseline": f"{avg_time_base:.1f}s",
        }
