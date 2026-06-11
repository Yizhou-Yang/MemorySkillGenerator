"""Applicability Gate + Task Type Classification — structural signals only.

SRDP Theory Connection (Section 3.3: Selective Injection Gate):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The gate module controls WHEN to inject skills, implementing a key insight
from the SRDP gap bound:

    gap(N) = C₁·ε_LLM(r_M) + C₂·δ_sem + E[δ_att]

Injecting irrelevant skills increases δ_att (attention dilution) without
reducing ε_LLM(r_M). The gate ensures injection only occurs when:
1. Relevant experiences exist (similarity > threshold) — prevents δ_att inflation
2. The library has sufficient coverage for the task region — r_M is locally small

Key design choice: The threshold is kept LOW (0.1) rather than high because:
- With 1M context window, δ_att from slightly-irrelevant skills is bounded
- A high threshold would effectively increase r_M by refusing to inject
  experiences that ARE in the library but don't meet the gate
- Better to inject and let the LLM's attention mechanism handle relevance
  than to artificially restrict coverage

This is the "permissive gate" strategy: preserve r_M at the cost of
slightly higher δ_att, which is acceptable under long-context models.
"""
from __future__ import annotations
import re
from .experience import ExperienceLibrary


def assess_task_complexity(task_desc: str) -> str:
    """simple / moderate / complex based on structural signals.

    SRDP Theory: Complexity assessment informs the expected δ_att budget.
    Complex tasks have more steps → more opportunities for attention degradation.
    This metadata helps downstream analysis correlate task complexity with
    the empirical gap between V* and V^π_M.
    """
    desc_lower = task_desc.lower()
    word_count = len(task_desc.split())
    sentence_count = len([s for s in re.split(r'[.!?\n]', task_desc) if s.strip()])

    complex_score = 0
    complex_score += len(re.findall(r'\band\b', desc_lower)) // 2
    complex_score += len(re.findall(r'(?:then|also|next|after that|finally|additionally)', desc_lower))
    complex_score += len(re.findall(r'(?:all|every|each)\b', desc_lower))
    complex_score += len(re.findall(r'(?:except|unless|only if|but not|make sure|must)', desc_lower))
    if word_count > 80: complex_score += 2
    elif word_count > 40: complex_score += 1
    if sentence_count >= 4: complex_score += 1

    simple_score = 0
    if re.match(r'^(?:what|who|when|where|how|which|is|are|does|did)\b', desc_lower):
        simple_score += 1
    if word_count < 15:
        simple_score += 1
    if sentence_count <= 1:
        simple_score += 1

    if complex_score >= 2: return "complex"
    if simple_score >= 2 and complex_score == 0: return "simple"
    return "moderate"


def should_augment(task_desc: str, library: ExperienceLibrary,
                   relevance_threshold: float = 0.1) -> tuple[bool, str]:
    """Inject only if semantically relevant experiences exist above threshold.

    SRDP Theory (Proposition 4: Permissive Gating):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    The gate implements a binary decision: inject or skip.

    - If we SKIP when relevant experiences exist → effective r_M increases
      (the agent can't access knowledge that IS in the library)
    - If we INJECT irrelevant experiences → δ_att increases
      (noise dilutes attention on the actual task)

    Under 1M context (where δ_att from mild irrelevance is bounded),
    the optimal strategy is PERMISSIVE gating (low threshold = 0.1):

        E[gap | permissive] = C₁·r_M^full + C₂·δ_sem + E[δ_att^mild]
        E[gap | strict]     = C₁·r_M^effective↑ + C₂·δ_sem + E[δ_att^low]

    Since r_M^effective↑ >> δ_att^mild under long context, permissive wins.

    The relevance_threshold (default 0.1) is kept low to ensure early-stage
    experiences are still injected when the library is small. With 1M context,
    the model can handle slightly less relevant experiences gracefully.
    """
    candidates = library.retrieve_similar(task_desc, top_k=1,
                                          min_similarity=relevance_threshold)
    if not candidates:
        return False, "no_relevant_experiences"
    return True, "relevant_experience_found"


def classify_task_type(task_desc: str, expected: str = "", metadata: dict | None = None) -> str:
    """Classify task type for logging/metadata purposes.

    Returns: agentic / qa / embodied

    SRDP Theory: Task type classification enables per-domain analysis of
    the gap bound components. Different domains exhibit different δ_att
    profiles (e.g., embodied tasks have higher δ_att due to longer action
    sequences, while QA tasks have lower δ_att but higher δ_sem sensitivity).

    NOTE: This classification is used for metadata tracking only.
    Injection behavior is identical regardless of task type — full experience
    injection is always applied. The previous qa→lightweight hints routing
    was removed because dynamic and static benchmarks benefit equally from
    full experience context.
    """
    metadata = metadata or {}
    desc_lower = task_desc.lower()

    # Metadata-based classification (from benchmark loader)
    if metadata:
        meta_type = metadata.get("task_type", "").lower()
        if meta_type and any(kw in meta_type for kw in ("pick", "place", "look_at", "clean", "heat", "cool")):
            return "embodied"
        if metadata.get("scenario_path") or metadata.get("tools") or metadata.get("apps"):
            return "agentic"
        bench = metadata.get("benchmark", "").lower()
        if bench in ("gaia2", "swebench", "swebench_dynamic", "gaia"):
            return "agentic"
        if bench in ("locomo", "longmemeval"):
            return "qa"

    # Structural signal detection
    if "conversation history" in desc_lower or "based on the conversation" in desc_lower:
        return "qa"

    # Physical action detection
    physical_verbs_in_desc = len(re.findall(
        r'\b(?:pick up|put down|go to|move to|take .{1,20} from|place .{1,20} on|examine .{1,20} with)\b', desc_lower))
    if physical_verbs_in_desc >= 2:
        return "embodied"

    # Expected answer structure
    if expected:
        expected_lower = expected.lower()
        if "->" in expected or expected.count("\n") >= 3:
            physical_verbs = ("go to", "take", "put", "use", "open", "pick", "move", "clean", "heat")
            if any(v in expected_lower for v in physical_verbs):
                return "embodied"
            return "agentic"
        if len(expected.split()) < 20:
            return "qa"

    # Description structure
    sentences = [s.strip() for s in re.split(r'[.!?\n]', task_desc) if s.strip()]
    question_count = sum(1 for s in sentences
                        if s.rstrip().endswith('?')
                        or re.match(r'^(?:what|who|when|where|which|how|is|are|does|did|can|could)\b', s.lower()))
    if question_count >= 1:
        return "qa"

    if len(sentences) >= 3 and len(task_desc.split()) > 50:
        return "agentic"
    return "qa"
