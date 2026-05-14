"""
SkillOS Dialect Framework — token compression for skill/memory presentation.

Adapted from SkillOS (github.com/EvolvingAgentsLabs/skillos) for SkillForge.

SkillOS Key Insight: "Generated tokens are more expensive than input tokens.
Every token an agent outputs costs compute, latency, and money. Storing verbose
prose in memory means future agents read verbose prose — the waste compounds."

We integrate 3 dialect types most relevant to SkillForge:

1. formal-proof: Forces step-by-step symbolic derivation for multi-hop QA
   - SkillOS benchmark: -51.3% tokens, 90/100 accuracy (same as plain)
   - Our use: Convert skill procedures into formal proof notation

2. caveman-prose: Strips filler words, keeps logic
   - SkillOS benchmark: -46-75% tokens, reversible
   - Our use: Compress skill descriptions and memory entries

3. exec-plan: Scenario → minimal execution plan
   - SkillOS benchmark: -70-85% tokens
   - Our use: Compress trajectory steps into minimal action sequences

Additionally, we implement the Hierarchical Skill Taxonomy:
   - 3-level hierarchy: Domain → Family → Skill
   - 4-step lazy loading protocol: -61% routing-phase tokens

SkillOS Benchmark Reference Values:
  | Dialect        | Token Reduction | Quality (Plain → SkillOS) |
  |----------------|-----------------|---------------------------|
  | formal-proof   | -51.3%          | 90 → 90 /100             |
  | system-dynamics| -61.1%          | 100 → 100 /100           |
  | strict-patch   | -97.5%          | 2/2 → 2/2                |
  | caveman-prose  | -46-75%         | reversible                |
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from src.models import Skill


# ============================================================
# Dialect Definitions
# ============================================================

@dataclass
class DialectResult:
    """Result of applying a dialect compression."""
    original_text: str
    compressed_text: str
    dialect_name: str
    original_tokens: int  # Approximate
    compressed_tokens: int  # Approximate
    compression_ratio: float  # 0.0 = no compression, 1.0 = 100% compressed
    preserved_elements: list[str] = field(default_factory=list)  # Elements that were preserved


# ============================================================
# Dialect 1: Formal Proof
# ============================================================

FORMAL_PROOF_SYSTEM_PROMPT = """\
You are a formal proof compiler. Convert the given skill/procedure into \
ONLY formal-proof notation. No English prose — only structured derivation.

### Formal-Proof Grammar:
```
GIVEN:
  P1: [premise/input]
  P2: [premise/input]
DERIVE:
  D1: [statement] [BY rule]
  D2: [statement] [BY rule]
QED: [conclusion/answer]
```
Rules: definition, decomposition, search, extraction, inference, verification, combination.
Use exact values at every step. Output ONLY the proof block."""

FORMAL_PROOF_RENDERER_PROMPT = """\
You are a technical writer. Read the following formal proof notation and \
write a clear, concise answer. Preserve ALL factual values exactly as given.

---
{dialect_output}
---

Write only the final answer, no explanation."""


def compile_formal_proof(skill: Skill) -> str:
    """
    Compile a skill's procedure into formal-proof dialect notation.

    This is the local (non-LLM) version that converts structured skill data
    into formal proof format. For LLM-powered compilation, use compile_with_llm().
    """
    lines = ["GIVEN:"]

    # Premises from skill description and facts
    lines.append(f"  P1: Task requires: {skill.description}")
    for i, fact in enumerate(skill.facts[:3], 2):
        lines.append(f"  P{i}: {fact}")

    lines.append("DERIVE:")
    for i, step in enumerate(skill.procedure, 1):
        rule = "decomposition" if i == 1 else "inference"
        if "search" in step.lower() or "find" in step.lower():
            rule = "search"
        elif "extract" in step.lower() or "identify" in step.lower():
            rule = "extraction"
        elif "combin" in step.lower() or "synthesiz" in step.lower():
            rule = "combination"
        elif "verif" in step.lower() or "check" in step.lower():
            rule = "verification"
        lines.append(f"  D{i}: {step} [BY {rule}]")

    # Constraints as verification steps
    for i, constraint in enumerate(skill.constraints[:2]):
        lines.append(f"  V{i+1}: Ensure: {constraint} [BY verification]")

    lines.append("QED: Apply derived steps to produce answer")

    return "\n".join(lines)


# ============================================================
# Dialect 2: Caveman Prose
# ============================================================

# Words to strip (articles, filler, hedging, passive voice markers)
CAVEMAN_STRIP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "must",
    "that", "which", "who", "whom", "whose",
    "very", "really", "quite", "rather", "somewhat", "fairly",
    "just", "simply", "basically", "essentially", "actually", "literally",
    "in order to", "make sure to", "it is important to",
    "please", "kindly", "always", "never",
    "however", "therefore", "furthermore", "moreover", "additionally",
    "nevertheless", "consequently", "subsequently",
}

CAVEMAN_REPLACEMENTS = {
    "you should": "",
    "make sure": "",
    "in order to": "to",
    "it is important to": "",
    "you need to": "",
    "you must": "",
    "please ensure": "",
    "be sure to": "",
    "take into account": "consider",
    "with respect to": "re:",
    "in the context of": "in",
    "as a result of": "from",
    "for the purpose of": "for",
}


def compress_caveman(text: str, level: str = "full") -> DialectResult:
    """
    Apply caveman-prose dialect compression.

    Levels:
      - lite: Strip articles and filler only (~30% reduction)
      - full: Strip + abbreviate + restructure (~50% reduction)
      - ultra: Maximum compression, terse fragments (~75% reduction)
    """
    original = text
    original_tokens = len(text.split())

    # Phase 1: Apply replacements
    compressed = text
    for phrase, replacement in CAVEMAN_REPLACEMENTS.items():
        compressed = re.sub(re.escape(phrase), replacement, compressed, flags=re.IGNORECASE)

    # Phase 2: Strip filler words
    if level in ("full", "ultra"):
        words = compressed.split()
        filtered = [w for w in words if w.lower().strip(string.punctuation) not in CAVEMAN_STRIP_WORDS]
        compressed = " ".join(filtered)

    # Phase 3: Ultra compression — remove all non-essential words
    if level == "ultra":
        # Keep only nouns, verbs, numbers, and key adjectives
        words = compressed.split()
        # Simple heuristic: keep words > 3 chars or numbers
        filtered = [w for w in words if len(w) > 3 or w[0].isdigit() or w in ("to", "if", "or", "no")]
        compressed = " ".join(filtered)

    # Clean up whitespace
    compressed = re.sub(r"\s+", " ", compressed).strip()
    compressed = re.sub(r"\s+([.,;:!?])", r"\1", compressed)

    compressed_tokens = len(compressed.split())

    ratio = 1.0 - (compressed_tokens / original_tokens) if original_tokens > 0 else 0.0

    return DialectResult(
        original_text=original,
        compressed_text=compressed,
        dialect_name=f"caveman-prose-{level}",
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=ratio,
    )


# ============================================================
# Dialect 3: Exec Plan
# ============================================================

def compress_exec_plan(steps: list[str]) -> DialectResult:
    """
    Apply exec-plan dialect: compress a list of procedure steps into
    minimal action notation.

    Format: [STEP N] action_verb(target) → expected_result
    """
    original = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    original_tokens = len(original.split())

    compressed_lines = []
    for i, step in enumerate(steps, 1):
        # Extract verb and target
        words = step.split()
        if not words:
            continue
        verb = words[0].lower().rstrip(".,;:")
        target = " ".join(words[1:4]) if len(words) > 1 else ""
        result = " ".join(words[-3:]) if len(words) > 4 else ""

        if result and result != target:
            compressed_lines.append(f"[{i}] {verb}({target}) → {result}")
        else:
            compressed_lines.append(f"[{i}] {verb}({target})")

    compressed = "\n".join(compressed_lines)
    compressed_tokens = len(compressed.split())

    ratio = 1.0 - (compressed_tokens / original_tokens) if original_tokens > 0 else 0.0

    return DialectResult(
        original_text=original,
        compressed_text=compressed,
        dialect_name="exec-plan",
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=ratio,
    )


# ============================================================
# Hierarchical Skill Taxonomy (SkillOS 3-level)
# ============================================================

@dataclass
class SkillTaxonomy:
    """
    3-level skill hierarchy: Domain → Family → Skill.

    SkillOS finding: 4-step lazy loading reduces routing-phase
    token consumption by ~61% versus a flat registry.

    Lazy loading protocol:
      Step 1: Load domain index only (e.g., "memory/", "planning/")
      Step 2: On domain match, load family index
      Step 3: On family match, load skill manifest (metadata only)
      Step 4: On skill selection, load full skill definition
    """
    domains: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    # Structure: {domain: {family: [skill_id, ...]}}

    def register(self, skill: Skill, domain: str = "general", family: str = "default") -> None:
        """Register a skill into the taxonomy."""
        if domain not in self.domains:
            self.domains[domain] = {}
        if family not in self.domains[domain]:
            self.domains[domain][family] = []
        self.domains[domain][family].append(skill.skill_id)

    def get_domain_index(self) -> str:
        """Step 1: Return domain-level index (minimal tokens)."""
        lines = ["[DOMAINS]"]
        for domain, families in self.domains.items():
            total_skills = sum(len(skills) for skills in families.values())
            lines.append(f"  {domain}/ ({total_skills} skills, {len(families)} families)")
        return "\n".join(lines)

    def get_family_index(self, domain: str) -> str:
        """Step 2: Return family-level index for a domain."""
        if domain not in self.domains:
            return f"[ERROR] Domain '{domain}' not found"
        lines = [f"[FAMILIES in {domain}/]"]
        for family, skills in self.domains[domain].items():
            lines.append(f"  {family}/ ({len(skills)} skills)")
        return "\n".join(lines)

    def get_skill_manifests(self, domain: str, family: str) -> str:
        """Step 3: Return skill manifests (metadata only) for a family."""
        if domain not in self.domains or family not in self.domains[domain]:
            return f"[ERROR] {domain}/{family} not found"
        lines = [f"[SKILLS in {domain}/{family}/]"]
        for skill_id in self.domains[domain][family]:
            lines.append(f"  - {skill_id}")
        return "\n".join(lines)

    def compute_token_savings(self, total_skills: int) -> dict[str, int]:
        """
        Compute token savings from lazy loading vs flat registry.

        Flat registry: all skills loaded = ~50 tokens/skill × N skills
        Lazy loading: domain index + family index + selected manifests
        """
        tokens_per_skill = 50  # Approximate tokens per skill in flat registry
        flat_tokens = total_skills * tokens_per_skill

        # Lazy loading: domain index (~5 tokens/domain) + family (~10 tokens/family)
        num_domains = len(self.domains)
        num_families = sum(len(f) for f in self.domains.values())
        lazy_step1 = num_domains * 5
        lazy_step2 = num_families * 10
        lazy_step3 = min(5, total_skills) * tokens_per_skill  # Only load selected

        lazy_tokens = lazy_step1 + lazy_step2 + lazy_step3

        return {
            "flat_tokens": flat_tokens,
            "lazy_tokens": lazy_tokens,
            "savings": flat_tokens - lazy_tokens,
            "savings_pct": (flat_tokens - lazy_tokens) / flat_tokens * 100 if flat_tokens > 0 else 0,
        }


# ============================================================
# Integrated Dialect Compiler
# ============================================================

class DialectCompiler:
    """
    Unified dialect compiler that selects and applies the best dialect
    for a given content type.

    Integrates with SkillForge's existing SkillFormatter for
    attention-aware presentation.
    """

    DIALECT_REGISTRY = {
        "formal-proof": {
            "type": "symbolic",
            "reduction": "51-75%",
            "reversible": True,
            "domain": "knowledge, memory",
            "description": "Forces step-by-step reasoning with rule citations",
        },
        "caveman-prose": {
            "type": "lexical",
            "reduction": "46-75%",
            "reversible": True,
            "domain": "memory, knowledge",
            "description": "Strip noise, keep logic",
        },
        "exec-plan": {
            "type": "symbolic",
            "reduction": "70-85%",
            "reversible": True,
            "domain": "orchestration, memory",
            "description": "Scenario → minimal execution plan",
        },
    }

    def __init__(self, llm_client: Any = None) -> None:
        self.llm_client = llm_client

    def compile(self, content: str, dialect: str = "auto", **kwargs) -> DialectResult:
        """
        Compile content using the specified dialect.

        Args:
            content: Text to compress
            dialect: "formal-proof", "caveman-prose", "exec-plan", or "auto"
        """
        if dialect == "auto":
            dialect = self._auto_select(content)

        if dialect == "caveman-prose":
            level = kwargs.get("level", "full")
            return compress_caveman(content, level=level)
        elif dialect == "exec-plan":
            steps = content.split("\n") if isinstance(content, str) else content
            steps = [s.strip() for s in steps if s.strip()]
            # Remove numbering
            steps = [re.sub(r"^\d+\.\s*", "", s) for s in steps]
            return compress_exec_plan(steps)
        elif dialect == "formal-proof":
            return self._compile_formal_proof_text(content)
        else:
            raise ValueError(f"Unknown dialect: {dialect}")

    def compile_skill(self, skill: Skill, dialect: str = "auto") -> DialectResult:
        """Compile a skill using the best dialect for its content."""
        if dialect == "auto":
            # Skills with procedures → exec-plan
            # Skills with constraints → formal-proof
            # General → caveman-prose
            if len(skill.procedure) > 3:
                dialect = "exec-plan"
            elif len(skill.constraints) > 2:
                dialect = "formal-proof"
            else:
                dialect = "caveman-prose"

        if dialect == "formal-proof":
            original = self._skill_to_text(skill)
            compressed = compile_formal_proof(skill)
            original_tokens = len(original.split())
            compressed_tokens = len(compressed.split())
            return DialectResult(
                original_text=original,
                compressed_text=compressed,
                dialect_name="formal-proof",
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                compression_ratio=1.0 - (compressed_tokens / original_tokens) if original_tokens > 0 else 0,
            )
        elif dialect == "exec-plan":
            return compress_exec_plan(skill.procedure)
        else:
            original = self._skill_to_text(skill)
            return compress_caveman(original, level="full")

    def compile_with_llm(self, content: str, dialect: str, task_context: str = "") -> str:
        """
        Use LLM to compile content into a dialect (higher quality but costs tokens).

        This is the SkillOS approach: the LLM IS the interpreter.
        """
        if self.llm_client is None:
            raise ValueError("LLM client required for LLM-powered compilation")

        if dialect == "formal-proof":
            system_prompt = FORMAL_PROOF_SYSTEM_PROMPT
        else:
            system_prompt = f"Compress the following into {dialect} format. Be maximally concise."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        return self.llm_client.chat(messages, temperature=0.0, max_tokens=1024)

    def render_from_dialect(self, dialect_output: str, dialect: str) -> str:
        """
        Render dialect output back to natural language (using LLM).

        This is the SkillOS "egress" step: internal dialect → user prose.
        """
        if self.llm_client is None:
            raise ValueError("LLM client required for rendering")

        if dialect == "formal-proof":
            prompt = FORMAL_PROOF_RENDERER_PROMPT.format(dialect_output=dialect_output)
        else:
            prompt = f"Expand the following compressed notation into clear English:\n\n{dialect_output}"

        messages = [
            {"role": "system", "content": "You are a technical writer. Be concise and precise."},
            {"role": "user", "content": prompt},
        ]
        return self.llm_client.chat(messages, temperature=0.0, max_tokens=512)

    def _auto_select(self, content: str) -> str:
        """Auto-select the best dialect based on content analysis."""
        content_lower = content.lower()
        # If it looks like steps/procedure
        if re.search(r"^\d+\.\s", content, re.MULTILINE):
            return "exec-plan"
        # If it has mathematical/logical content
        if re.search(r"(calculate|prove|derive|theorem|equation)", content_lower):
            return "formal-proof"
        # Default: caveman prose
        return "caveman-prose"

    @staticmethod
    def _skill_to_text(skill: Skill) -> str:
        """Convert skill to full text representation."""
        parts = [f"Skill: {skill.name}", f"Description: {skill.description}"]
        if skill.procedure:
            parts.append("Procedure:")
            for i, step in enumerate(skill.procedure, 1):
                parts.append(f"  {i}. {step}")
        if skill.constraints:
            parts.append("Constraints:")
            for c in skill.constraints:
                parts.append(f"  - {c}")
        if skill.facts:
            parts.append("Facts:")
            for f in skill.facts:
                parts.append(f"  - {f}")
        return "\n".join(parts)

    @staticmethod
    def _compile_formal_proof_text(text: str) -> DialectResult:
        """Compile arbitrary text into formal-proof notation."""
        original_tokens = len(text.split())

        lines = ["GIVEN:"]
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        for i, sent in enumerate(sentences[:3], 1):
            lines.append(f"  P{i}: {sent[:80]}")

        lines.append("DERIVE:")
        for i, sent in enumerate(sentences[3:8], 1):
            lines.append(f"  D{i}: {sent[:80]} [BY inference]")

        if len(sentences) > 8:
            lines.append(f"QED: {sentences[-1][:80]}")
        else:
            lines.append("QED: Conclusion follows from derivation")

        compressed = "\n".join(lines)
        compressed_tokens = len(compressed.split())

        return DialectResult(
            original_text=text,
            compressed_text=compressed,
            dialect_name="formal-proof",
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=1.0 - (compressed_tokens / original_tokens) if original_tokens > 0 else 0,
        )
