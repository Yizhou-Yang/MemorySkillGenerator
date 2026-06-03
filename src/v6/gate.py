"""Applicability Gate + Task Type Classification — no hardcoded keywords."""
from __future__ import annotations
import re
from .experience import ExperienceLibrary, _compute_similarity


def assess_task_complexity(task_desc: str) -> str:
    """simple / moderate / complex based on structural signals (no domain keywords)."""
    desc_lower = task_desc.lower()
    word_count = len(task_desc.split())
    sentence_count = len([s for s in re.split(r'[.!?\n]', task_desc) if s.strip()])

    complex_score = 0
    # Multi-step: multiple independent clauses
    complex_score += len(re.findall(r'\band\b', desc_lower)) // 2  # 2+ "and" = multi-step
    complex_score += len(re.findall(r'(?:then|also|next|after that|finally|additionally)', desc_lower))
    complex_score += len(re.findall(r'(?:all|every|each)\b', desc_lower))
    # Constraints
    complex_score += len(re.findall(r'(?:except|unless|only if|but not|make sure|must)', desc_lower))
    # Length as proxy
    if word_count > 80: complex_score += 2
    elif word_count > 40: complex_score += 1
    if sentence_count >= 4: complex_score += 1

    simple_score = 0
    # Question form
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
                   relevance_threshold: float = 0.15) -> tuple[bool, str]:
    """Three-layer gate: complexity → relevance → historical effectiveness."""
    complexity = assess_task_complexity(task_desc)

    if complexity == "simple":
        return False, f"simple_task (complexity={complexity})"

    candidates = library.retrieve_similar(task_desc, top_k=1)
    if not candidates:
        return False, "no_relevant_experiences"

    best_relevance = _compute_similarity(task_desc, candidates[0].task_desc)
    if best_relevance < relevance_threshold:
        return False, f"low_relevance ({best_relevance:.3f} < {relevance_threshold})"

    effectiveness = library.get_augmentation_effectiveness(complexity)
    if effectiveness < 0.3:
        return False, f"historically_ineffective ({effectiveness:.1%} help rate for {complexity})"

    return True, f"approved (complexity={complexity}, relevance={best_relevance:.3f}, effectiveness={effectiveness:.1%})"


def classify_task_type(task_desc: str, expected: str = "", metadata: dict | None = None) -> str:
    """Auto-classify: agentic / qa / embodied. Uses structural signals, not keyword lists.
    
    Signal priority: metadata > expected structure > description structure.
    """
    metadata = metadata or {}
    desc_lower = task_desc.lower()

    # Signal 1: Metadata (explicit labels from benchmark loader)
    if metadata:
        meta_type = metadata.get("task_type", "").lower()
        # Embodied: physical manipulation task types
        if meta_type and any(kw in meta_type for kw in ("pick", "place", "look_at", "clean", "heat", "cool")):
            return "embodied"
        # Agentic: has interactive environment
        if metadata.get("scenario_path") or metadata.get("tools") or metadata.get("apps"):
            return "agentic"
        bench = metadata.get("benchmark", "").lower()
        if bench in ("gaia2", "swebench"):
            return "agentic"

    # Signal 2: Expected answer structure
    if expected:
        expected_lower = expected.lower()
        # Multi-step action sequence
        if "->" in expected or expected.count("\n") >= 3:
            # Check if actions look physical
            physical_verbs = ("go to", "take", "put", "use", "open", "pick", "move", "clean", "heat")
            if any(v in expected_lower for v in physical_verbs):
                return "embodied"
            return "agentic"
        # Short factual answer
        if len(expected.split()) < 20:
            return "qa"

    # Signal 3: Description structure (no domain-specific keyword lists)
    sentences = [s.strip() for s in re.split(r'[.!?\n]', task_desc) if s.strip()]

    # Imperative = starts with a verb (i.e. NOT pronoun/article/question word)
    _non_imp = re.compile(
        r'^(?:I|You|He|She|It|We|They|The|A|An|This|That|These|Those|My|Your|His|Her|Its|Our|Their|'
        r'What|Who|When|Where|How|Which|Is|Are|Was|Were|Do|Does|Did|'
        r'Can|Could|Would|Should|May|If|Although|Because|Since|After|However|Note)\b', re.I)
    # Cognitive verbs: these are "thinking" imperatives → QA, not action
    _cognitive = re.compile(
        r'^(?:Answer|Summarize|Explain|Describe|Identify|Define|Compare|Analyze|'
        r'Evaluate|Discuss|List|Name|State|Determine|Consider|Based on|Given)\b', re.I)
    imperative_count = sum(1 for s in sentences
                          if s and len(s.split()) >= 3 and not _non_imp.match(s) and not _cognitive.match(s))
    cognitive_count = sum(1 for s in sentences if s and _cognitive.match(s))

    # Question detection
    question_count = sum(1 for s in sentences
                        if s.rstrip().endswith('?')
                        or re.match(r'^(?:what|who|when|where|which|how|is|are|does|did|can|could)\b', s.lower()))

    # Physical action detection: verbs about moving/manipulating objects
    physical_verbs_in_desc = len(re.findall(
        r'\b(?:pick up|put down|go to|move to|take .{1,20} from|place .{1,20} on|examine .{1,20} with)\b', desc_lower))

    # Multi-command detection: multiple distinct instructions
    multi_command = len(re.findall(r'(?:^|\.\s+)[A-Z][a-z]+\b', task_desc))  # capitalized sentence starts

    if physical_verbs_in_desc >= 2: return "embodied"
    if question_count >= 1 and imperative_count == 0: return "qa"
    if cognitive_count >= 1 and imperative_count == 0: return "qa"
    if imperative_count >= 1 or multi_command >= 3: return "agentic"

    # Fallback: long multi-sentence = likely agentic; short = qa
    if len(sentences) >= 3 and len(task_desc.split()) > 50:
        return "agentic"
    return "qa"
