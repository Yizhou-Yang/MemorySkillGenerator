"""
SkillForge V6 — Applicability Gate + Task Type Classification.

Decides WHETHER to inject experience (gate) and HOW to inject (task type routing).
"""
from __future__ import annotations
import re
from .experience import ExperienceLibrary


# ══════════════════════════════════════════════════════════════════════════════
#  Task Complexity Assessment
# ══════════════════════════════════════════════════════════════════════════════

def assess_task_complexity(task_desc: str) -> str:
    """Classify task complexity: simple / moderate / complex.
    
    Based on TRACE finding #7 (推理强度收益不均匀):
    - Simple: single action, clear intent → NO augmentation needed
    - Moderate: 2-3 steps, some ambiguity → light augmentation
    - Complex: multi-step, constraints, cross-app → full augmentation
    """
    indicators_complex = [
        r'\band\b.*\band\b',
        r'after.*(?:then|also|next)',
        r'(?:all|every|each)\b',
        r'(?:email|message|chat).*(?:calendar|event|schedule)',
        r'(?:calendar|event).*(?:email|message|contact)',
        r'(?:except|unless|only if|but not)',
        r'(?:last|first|most recent|latest)',
    ]
    indicators_simple = [
        r'^(?:get|show|list|find|search)\b',
        r'^(?:what|who|when|where)\b',
    ]

    desc_lower = task_desc.lower()
    complex_score = sum(1 for p in indicators_complex if re.search(p, desc_lower))
    simple_score = sum(1 for p in indicators_simple if re.search(p, desc_lower))

    word_count = len(task_desc.split())
    if word_count > 80:
        complex_score += 2
    elif word_count > 40:
        complex_score += 1
    elif word_count < 15:
        simple_score += 1

    sentence_count = len(re.split(r'[.!?\n]', task_desc))
    if sentence_count >= 4:
        complex_score += 1

    if complex_score >= 2:
        return "complex"
    elif simple_score >= 1 and complex_score == 0 and word_count < 20:
        return "simple"
    return "moderate"


# ══════════════════════════════════════════════════════════════════════════════
#  Applicability Gate
# ══════════════════════════════════════════════════════════════════════════════

def should_augment(task_desc: str, library: ExperienceLibrary,
                   relevance_threshold: float = 0.15) -> tuple[bool, str]:
    """Three-layer gate: complexity → relevance → historical effectiveness.
    
    TRACE findings addressed:
    - #1: "有Skill不一定更好" → only augment when genuine signal exists
    - #2: "Skill虹吸" → don't augment simple tasks (anti-siphon)
    
    Returns: (should_augment: bool, reason: str)
    """
    complexity = assess_task_complexity(task_desc)
    
    # Layer 1: Simple tasks → no augmentation
    if complexity == "simple":
        return False, f"simple_task (complexity={complexity})"
    
    # Layer 2: Check relevance of available experiences
    candidates = library.retrieve_similar(task_desc, top_k=1)
    if not candidates:
        return False, "no_relevant_experiences"
    
    stop_words = {"the", "a", "an", "to", "and", "or", "in", "on", "at", "for",
                  "of", "with", "is", "are", "was", "were", "be", "been", "being",
                  "that", "this", "it", "my", "all", "i", "me"}
    task_words = set(task_desc.lower().split()) - stop_words
    
    best_relevance = 0.0
    for exp in candidates:
        exp_words = set(exp.task_desc.lower().split()) - stop_words
        if exp_words:
            overlap = len(task_words & exp_words) / max(len(task_words | exp_words), 1)
            best_relevance = max(best_relevance, overlap)
    
    if best_relevance < relevance_threshold:
        return False, f"low_relevance ({best_relevance:.3f} < {relevance_threshold})"
    
    # Layer 3: Historical effectiveness check
    effectiveness = library.get_augmentation_effectiveness(complexity)
    if effectiveness < 0.3:
        return False, f"historically_ineffective ({effectiveness:.1%} help rate for {complexity})"
    
    return True, f"approved (complexity={complexity}, relevance={best_relevance:.3f}, effectiveness={effectiveness:.1%})"


# ══════════════════════════════════════════════════════════════════════════════
#  Task Type Auto-Classification (zero-config for new benchmarks)
# ══════════════════════════════════════════════════════════════════════════════

def classify_task_type(task_desc: str, expected: str = "", metadata: dict | None = None) -> str:
    """Auto-classify task type from structural features. No per-benchmark config needed.
    
    Signal layers (strongest → weakest):
    1. Metadata: task_type field, scenario_path, tools → instant classification
    2. Expected answer structure: action sequence → agentic/embodied; short text → qa
    3. Description syntax: imperative verbs → agentic; question words → qa; physical verbs → embodied
    
    Returns: "agentic" | "qa" | "embodied"
    """
    metadata = metadata or {}
    desc_lower = task_desc.lower()
    
    # ─── Signal 1: Metadata ───────────────────────────────────────────
    if metadata:
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
        if "->" in expected or expected.count("\n") >= 3:
            action_verbs = ["go to", "take", "put", "use", "open", "pick", "move", "clean", "heat"]
            if any(v in expected_lower for v in action_verbs):
                return "embodied"
            return "agentic"
        if len(expected.split()) < 20:
            return "qa"
    
    # ─── Signal 3: Description syntax ─────────────────────────────────
    imperative_verbs = [
        r'\b(?:create|cancel|send|book|add|remove|save|delete|update|schedule|reply)\b',
        r'\b(?:search|buy|order|transfer|forward|move|set|change)\b',
    ]
    question_patterns = [
        r'^(?:answer|what|who|when|where|which|how|find|name|list)\b',
        r'(?:question|following question|based on)',
        r'\?\s*$',
    ]
    physical_patterns = [
        r'\b(?:pick up|put down|go to|move to|open door|close door)\b',
        r'\b(?:take .* from|place .* on|examine .* with)\b',
        r'\b(?:heat|cool|clean|slice|toggle)\b.*\b(?:with|using)\b',
    ]
    tool_patterns = [
        r'\b(?:calendar|email|contact|message|shopping|terminal|api|endpoint)\b',
        r'\b(?:file|database|server|deploy|commit|pull request)\b',
    ]

    imperative_count = sum(1 for p in imperative_verbs if re.search(p, desc_lower))
    question_score = sum(1 for p in question_patterns if re.search(p, desc_lower))
    physical_score = sum(1 for p in physical_patterns if re.search(p, desc_lower))
    tool_score = sum(1 for p in tool_patterns if re.search(p, desc_lower))

    if physical_score >= 2:
        return "embodied"
    if tool_score >= 1 or imperative_count >= 2:
        return "agentic"
    if question_score >= 1 and imperative_count == 0:
        return "qa"
    
    # Fallback heuristic
    sentence_count = len(re.split(r'[.!?\n]', task_desc))
    word_count = len(task_desc.split())
    if sentence_count >= 3 and word_count > 50 and imperative_count >= 1:
        return "agentic"
    
    return "qa"  # Default: safest (lightweight hints won't hurt)
