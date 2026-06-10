"""SkillForge V6 — Theoretical Foundation (SRDP Framework).

This module documents the theoretical grounding of SkillForge in the
Skill Retrieval-Dependent Policy (SRDP) framework and provides a centralized
reference for how each code module maps to the formal gap bound.

═══════════════════════════════════════════════════════════════════════════════
SRDP GAP BOUND (Corollary 15, adapted from Memento 2)
═══════════════════════════════════════════════════════════════════════════════

    sup_s |V*(s) - V^{π_M}(s)| ≤ (2R_max / (1-γ)²) · [ε_LLM(r_M) + δ_M]

Where:
    V*(s)           = optimal value function
    V^{π_M}(s)     = value under skill-augmented policy π_M
    ε_LLM(r_M)     = coverage error (how far library skills are from optimal)
    r_M             = coverage radius of skill library M
    δ_M             = retrieval error (how often wrong skill is selected/used)

The composite policy is:
    π_μ(a|s, M) = Σ_c μ(c|s,M) · p_LLM(a|s,c)

Where μ is the retrieval strategy and p_LLM is the frozen LLM's action
distribution conditioned on skill c. Since LLM parameters are frozen,
ALL improvement must come from manipulating M (the skill library).

═══════════════════════════════════════════════════════════════════════════════
OUR CONTRIBUTION: δ_M DECOMPOSITION
═══════════════════════════════════════════════════════════════════════════════

We decompose the retrieval error into two independent dimensions:

    δ_M ≤ δ_sem + E[δ_att]

    δ_sem : Semantic retrieval error — probability of selecting the WRONG skill.
            Caused by: embedding space limitations, retrieval dilution from
            redundant skills, Boltzmann temperature miscalibration.

    δ_att : Attention degradation error — even when the RIGHT skill is retrieved,
            the LLM fails to properly utilize it due to attention allocation issues.
            Caused by: Lost-in-the-Middle effect, format parsing ambiguity,
            consistency collapse from contradictory skills, negation priming.

Key insight: δ_sem and δ_att are INDEPENDENT — you can freeze skill content
and only change presentation (position, format, signal separation) to reduce
δ_att without affecting δ_sem.

═══════════════════════════════════════════════════════════════════════════════
MODULE → THEORY MAPPING
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────┬──────────────────────┬─────────────────────────────┐
│ Module              │ Theory Target        │ Mechanism                   │
├─────────────────────┼──────────────────────┼─────────────────────────────┤
│ experience.py       │ r_M (frozen)         │ Zero Information Loss:      │
│   ExperienceLibrary │ δ_sem (online fix)   │ never delete/compress →     │
│                     │                      │ r_M never increases.        │
│                     │                      │ Effectiveness weighting →   │
│                     │                      │ online δ_sem correction.    │
├─────────────────────┼──────────────────────┼─────────────────────────────┤
│ injection.py        │ δ_att (reduce)       │ Dual-Channel Separation:    │
│   build_augmented   │                      │ success/failure in distinct │
│   _prompt           │                      │ sections → reduces          │
│                     │                      │ consistency collapse.       │
│                     │                      │ Structured format → reduces │
│                     │                      │ format parsing ambiguity.   │
├─────────────────────┼──────────────────────┼─────────────────────────────┤
│ refine.py           │ δ_att (reduce)       │ Cross-Agent Critic:         │
│   cross_agent_eval  │                      │ filters noise/trivial       │
│   critic_refine     │                      │ skills that would waste     │
│                     │                      │ attention budget.           │
│                     │                      │ Enrichment → clearer format │
│                     │                      │ → lower parsing ambiguity.  │
├─────────────────────┼──────────────────────┼─────────────────────────────┤
│ response_filter.py  │ δ_att (reduce)       │ Attention Signal Density:   │
│   AIResponseProc    │                      │ removes noise from conv     │
│                     │                      │ history → concentrates      │
│                     │                      │ attention on actionable     │
│                     │                      │ content in future turns.    │
├─────────────────────┼──────────────────────┼─────────────────────────────┤
│ gate.py             │ r_M (preserve)       │ Relevance Gating:           │
│   should_augment    │ δ_att (prevent)      │ only inject when relevant   │
│                     │                      │ experiences exist →         │
│                     │                      │ prevents irrelevant skills  │
│                     │                      │ from consuming attention.   │
└─────────────────────┴──────────────────────┴─────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
WHY COMPRESSION IS HARMFUL (Anti-pattern analysis)
═══════════════════════════════════════════════════════════════════════════════

Prior work (SkillOS, Voyager, MemSkill) reduces δ_sem via Merge/Prune:
    - Fewer candidates → easier to select correctly → δ_sem ↓
    - BUT deleted skills may cover unique task regions → r_M ↑
    - ε_LLM(r_M) ↑ outweighs δ_sem ↓ → NET NEGATIVE

SkillForge's approach:
    - Never delete/compress → r_M frozen (ε_LLM unchanged)
    - Effectiveness weighting → δ_sem corrected online (no deletion needed)
    - Quality gating + dual-channel + structured format → δ_att ↓↓
    - Net result: gap bound tightens without coverage sacrifice

═══════════════════════════════════════════════════════════════════════════════
δ_att DEGRADATION MECHANISMS (5 sources)
═══════════════════════════════════════════════════════════════════════════════

1. Lost in the Middle (Liu et al., 2023):
   - Mid-position content receives 30-50% less attention weight
   - Mitigation: Quality gating ensures only high-value skills are injected,
     reducing total injection volume so all skills stay in attention peaks

2. Format Parsing Ambiguity:
   - Prose-style skills harder to parse than structured format
   - Mitigation: refine.py enforces structured JSON output;
     injection.py uses markdown headings + labeled fields

3. Consistency Collapse:
   - Contradictory skills → LLM silently picks one (23-47% error rate)
   - Mitigation: Dual-channel separation (success ≠ failure section);
     avoidance notes use positive framing ("Instead do X" not "Don't do Y")

4. Negation Priming (Pink Elephant Effect):
   - "Don't do X" primes the LLM to do X
   - Mitigation: refine.py's avoidance_note uses constructive alternatives;
     injection formats emphasize WHAT TO DO, not what to avoid

5. Retrieval Dilution:
   - Near-duplicate skills split probability mass under Boltzmann retrieval
   - Mitigation: Effectiveness weighting naturally downweights redundant
     skills that don't improve scores when injected (self-correcting signal)

═══════════════════════════════════════════════════════════════════════════════
PAPER NARRATIVE (one-paragraph summary)
═══════════════════════════════════════════════════════════════════════════════

SRDP theory (Corollary 15) bounds skill-augmented agent performance by
coverage error ε_LLM(r_M) and retrieval error δ_M. We decompose δ_M into
semantic retrieval error δ_sem (wrong skill selected) and attention
degradation error δ_att (right skill selected but not properly utilized).
Prior work reduces δ_sem via compression (Merge/Prune), but this
simultaneously increases r_M — our experiments confirm this trade-off is
net negative. SkillForge takes a different approach: (1) Zero Information
Loss freezes r_M; (2) Effectiveness-Weighted Retrieval corrects δ_sem
online; (3) Cross-Agent Critic + Dual-Channel Injection reduces δ_att
through quality gating and signal separation. Together, these mechanisms
tighten the gap bound without sacrificing coverage.
"""

# ═══════════════════════════════════════════════════════════════════════════
# Theoretical constants referenced across modules
# ═══════════════════════════════════════════════════════════════════════════

# Effectiveness weighting bounds (δ_sem online correction)
EFFECTIVENESS_WEIGHT_MIN = 0.3   # Floor: never fully suppress a skill (preserves r_M)
EFFECTIVENESS_WEIGHT_MAX = 1.5   # Ceiling: prevents runaway amplification
EFFECTIVENESS_COLD_START = 1.0   # Default weight before sufficient observations
EFFECTIVENESS_MIN_OBS = 2        # Minimum observations before weighting activates

# Quality gating thresholds (δ_att prevention)
QUALITY_SCORE_MIN = 0.3          # Below this, "success" is likely noise
QUALITY_CAUSAL_MIN_LEN = 5       # Minimum causal_lesson length to be non-trivial
QUALITY_HIGH_SCORE = 0.8         # Above this, even unrefined experiences pass

# Relevance gating (r_M preservation + δ_att prevention)
RELEVANCE_THRESHOLD = 0.1        # Low threshold: preserves coverage (r_M)
                                  # while preventing completely irrelevant injection

# AI response filter (attention signal density)
AI_FILTER_MIN_TEXT = 50          # Minimum chars to trigger AI evaluation
AI_FILTER_CONFIDENCE = 0.7      # Below this confidence, keep content (safe default)

# Loop detection (behavioral δ_att — agent stuck = wasted attention budget)
LOOP_EXACT_THRESHOLD = 3         # Identical calls before intervention
LOOP_SAME_TOOL_THRESHOLD = 6     # Same tool calls before intervention


def get_theory_summary() -> str:
    """Return a concise theory summary for inclusion in paper/docs."""
    return (
        "SkillForge grounds its design in the SRDP gap bound: "
        "sup|V*-V^π| ≤ C·[ε(r_M) + δ_sem + δ_att]. "
        "Zero Information Loss freezes r_M; "
        "Effectiveness-Weighted Retrieval corrects δ_sem online; "
        "Cross-Agent Critic + Dual-Channel Injection reduces δ_att "
        "through quality gating and signal separation."
    )


def get_module_mapping() -> dict[str, dict[str, str]]:
    """Return the module → theory mapping as a structured dict.

    Useful for automated documentation generation and paper tables.
    """
    return {
        "experience.py": {
            "theory_target": "r_M (frozen) + δ_sem (online correction)",
            "mechanism": "Zero Information Loss + Effectiveness-Weighted Retrieval",
            "bound_effect": "ε_LLM(r_M) unchanged; δ_sem ↓ via online reweighting",
        },
        "injection.py": {
            "theory_target": "δ_att (reduce)",
            "mechanism": "Dual-Channel Separation + Structured Format",
            "bound_effect": "δ_att ↓ via consistency collapse prevention + format clarity",
        },
        "refine.py": {
            "theory_target": "δ_att (reduce)",
            "mechanism": "Cross-Agent Critic + Forced Enrichment",
            "bound_effect": "δ_att ↓ via noise filtering + format standardization",
        },
        "response_filter.py": {
            "theory_target": "δ_att (reduce)",
            "mechanism": "AI-Driven Attention Signal Density Optimization",
            "bound_effect": "δ_att ↓ via noise removal from conversation history",
        },
        "gate.py": {
            "theory_target": "r_M (preserve) + δ_att (prevent)",
            "mechanism": "Relevance-Based Injection Gating",
            "bound_effect": "Prevents irrelevant skills from consuming attention budget",
        },
    }
