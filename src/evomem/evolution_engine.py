"""
Evolution Engine v2 — Information-theoretic skill evolution.

Core insight: Version is not a counter, it's an ADAPTIVE COMPRESSION CONTROLLER.

Each skill has layered representation:
  Layer 0 (hot):  compressed strategy — injected into prompt (low token cost)
  Layer 1 (warm): detailed patterns + constraints — expanded on demand
  Layer 2 (cold): raw experience traces — never deleted, used for rollback

Learning is embedding-space EMA update, not text concatenation.
Evolution decisions consider:
  1. Novelty: how much new information does this experience carry?
  2. Confidence: how stable is this skill? (high version + low drift = stable)
  3. Info gain: novelty × outcome — is it worth updating?
  4. Update cost: changing a high-confidence skill is expensive

Key difference from v1:
  - v1: decide based on similarity thresholds → text merge
  - v2: decide based on information theory → embedding update + layered compression
"""

from __future__ import annotations

import math
import time
import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from src.models import Skill
from src.evomem.versioned_skill_library import VersionedSkillLibrary, SkillHistory
from src.evomem.skill_embedder import SkillEmbedder
from src.evomem.patch_recorder import PatchRecorder


# ============================================================
# Skill State: Version as Learning State
# ============================================================

@dataclass
class SkillState:
    """
    The learning state of a skill — version is meaningful, not just a counter.
    
    Inspired by:
      - EMA (exponential moving average) from RL value functions
      - Information gain from Bayesian learning
      - Catastrophic forgetting prevention from continual learning
    """
    skill_id: str
    
    # Embedding state (the "knowledge" in vector space)
    centroid: np.ndarray | None = None  # Mean embedding of all experiences
    
    # Learning dynamics
    experience_count: int = 0
    confidence: float = 0.0           # [0, 1] — how stable/reliable this skill is
    learning_rate: float = 1.0        # Decreases with experience (1/sqrt(n))
    retrieval_weight: float = 1.0     # Decays with disuse, recovers on recall
    
    # Performance tracking
    perf_history: list = field(default_factory=list)  # Recent F1 scores
    perf_ema: float = 0.0             # Exponential moving average of performance
    
    # Drift tracking
    drift_history: list = field(default_factory=list)  # Cosine distance per update
    cumulative_drift: float = 0.0
    
    # Layered representation
    layer0_summary: str = ""          # Hot: compressed (for prompt injection)
    layer1_details: str = ""          # Warm: detailed patterns
    layer2_examples: list = field(default_factory=list)  # Cold: raw examples
    
    # Timestamps
    created_at: float = 0.0
    last_used_at: float = 0.0
    last_updated_at: float = 0.0
    
    @property
    def version(self) -> int:
        """Version = experience count (meaningful, not arbitrary)."""
        return self.experience_count
    
    @property
    def maturity(self) -> str:
        """Human-readable maturity level."""
        if self.experience_count <= 1:
            return "nascent"
        elif self.experience_count <= 3:
            return "developing"
        elif self.confidence >= 0.7:
            return "mature"
        else:
            return "unstable"
    
    def compute_learning_rate(self) -> float:
        """Adaptive learning rate: slower for mature skills."""
        return 1.0 / math.sqrt(max(1, self.experience_count))
    
    def compute_confidence(self) -> float:
        """Confidence = f(experience_count, performance_consistency, low_drift)."""
        if not self.perf_history:
            return 0.0
        
        # Factor 1: Experience (more = higher confidence, with diminishing returns)
        exp_factor = 1.0 - math.exp(-self.experience_count / 5.0)
        
        # Factor 2: Performance consistency (low variance = high confidence)
        if len(self.perf_history) >= 2:
            perf_var = np.var(self.perf_history[-5:])
            consistency_factor = 1.0 / (1.0 + 5 * perf_var)
        else:
            consistency_factor = 0.5
        
        # Factor 3: Low drift (stable embedding = high confidence)
        drift_factor = 1.0 / (1.0 + self.cumulative_drift)
        
        return min(1.0, exp_factor * consistency_factor * drift_factor)


# ============================================================
# Evolution Decision (enhanced with info-theoretic reasoning)
# ============================================================

class EvolutionDecision:
    """Evolution decision with information-theoretic context."""

    STRENGTHEN = "strengthen"
    REFINE = "refine"
    BRANCH = "branch"
    RETIRE = "retire"
    ROLLBACK = "rollback"
    CONSOLIDATE = "consolidate"
    CREATE = "create"
    NOOP = "noop"

    def __init__(self, action: str, target_id: str | None = None,
                 similarity: float = 0.0, novelty: float = 0.0,
                 info_gain: float = 0.0, reason: str = ""):
        self.action = action
        self.target_id = target_id
        self.similarity = similarity
        self.novelty = novelty
        self.info_gain = info_gain
        self.reason = reason

    def __repr__(self):
        return (f"EvolutionDecision({self.action}, "
                f"target={self.target_id[:8] if self.target_id else None}, "
                f"sim={self.similarity:.3f}, novelty={self.novelty:.3f}, "
                f"info_gain={self.info_gain:.3f})")


# ============================================================
# Evolution Engine v2
# ============================================================

class SkillEvolutionEngine:
    """
    Information-theoretic skill evolution engine.
    
    Core principle: Every evolution decision is an information trade-off.
      - Update cost: risk of catastrophic forgetting (proportional to confidence)
      - Info gain: novelty × outcome quality
      - Decision: update iff info_gain > update_cost
    
    Version semantics:
      - version = experience_count (meaningful)
      - High version + high confidence = mature skill (low learning rate)
      - High version + low confidence = unstable skill (needs rollback)
      - Low version = nascent skill (high learning rate, explore freely)
    """

    # Information-theoretic thresholds
    NOVELTY_THRESHOLD_CREATE = 0.7     # Very novel → new skill
    NOVELTY_THRESHOLD_BRANCH = 0.4     # Moderately novel → branch
    INFO_GAIN_MIN_UPDATE = 0.05        # Below this → not worth updating
    CONFIDENCE_PROTECT = 0.8           # Above this → resist single-example changes
    DRIFT_ROLLBACK_THRESHOLD = 0.5     # Cumulative drift → rollback
    DECAY_RATE = 0.98                  # Retrieval weight decay per step

    def __init__(
        self,
        library: VersionedSkillLibrary | None = None,
        embedder: SkillEmbedder | None = None,
        recorder: PatchRecorder | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.library = library or VersionedSkillLibrary()
        self.embedder = embedder or SkillEmbedder()
        self.recorder = recorder or PatchRecorder()
        self.config = config or {}

        # Skill states (the real learning state, beyond what library stores)
        self._states: dict[str, SkillState] = {}
        self._step_counter = 0

        # Stats
        self.stats = {
            "decisions": {d: 0 for d in [
                "strengthen", "refine", "branch", "retire",
                "rollback", "consolidate", "create", "noop"
            ]},
            "total_steps": 0,
            "total_info_gained": 0.0,
            "total_drift": 0.0,
        }

    # ================================================================
    # Core: Information-Theoretic Decision
    # ================================================================

    def decide(self, task_desc: str, outcome: float,
               experience_embedding: np.ndarray | None = None) -> EvolutionDecision:
        """
        Decide evolution action using information theory.
        
        Args:
            task_desc: Task description text
            outcome: Quality score [0, 1] — can be F1 (supervised) or self-eval (unsupervised)
            experience_embedding: Pre-computed embedding (optional, computed if None)
        
        Returns:
            EvolutionDecision with info-theoretic reasoning
        """
        self._step_counter += 1
        
        # Compute experience embedding
        if experience_embedding is None:
            experience_embedding = self.embedder.embed(task_desc[:500])
        
        # Decay all retrieval weights (recency bias)
        for state in self._states.values():
            state.retrieval_weight *= self.DECAY_RATE
        
        # Find best matching skill
        results = self.embedder.search(task_desc[:500], top_k=1, threshold=0.01)
        
        if not results:
            # No skills exist yet
            if outcome >= 0.2:
                return EvolutionDecision(
                    EvolutionDecision.CREATE, novelty=1.0, info_gain=outcome,
                    reason="Empty library, creating first skill"
                )
            return EvolutionDecision(EvolutionDecision.NOOP, reason="Empty library + poor outcome")
        
        best_id, best_sim = results[0]
        state = self._states.get(best_id)
        
        if not state:
            # State not tracked yet (legacy skill)
            if outcome >= 0.3:
                return EvolutionDecision(
                    EvolutionDecision.CREATE, novelty=1.0 - best_sim,
                    similarity=best_sim, info_gain=outcome,
                    reason="No state tracked, creating new"
                )
            return EvolutionDecision(EvolutionDecision.NOOP)
        
        # === Information-theoretic quantities ===
        novelty = 1.0 - best_sim  # How different is this from existing knowledge?
        info_gain = novelty * outcome  # Novelty weighted by quality
        update_cost = state.confidence * novelty  # Cost of changing a confident skill
        net_info = info_gain - update_cost  # Net benefit of updating
        
        # Update retrieval weight (this skill was recalled)
        state.retrieval_weight = min(1.0, state.retrieval_weight + 0.2)
        state.last_used_at = time.time()
        
        # === Decision logic based on information balance ===
        
        # Case 1: Very novel → CREATE new skill (don't contaminate existing)
        if novelty >= self.NOVELTY_THRESHOLD_CREATE:
            if outcome >= 0.3:
                return EvolutionDecision(
                    EvolutionDecision.CREATE, similarity=best_sim,
                    novelty=novelty, info_gain=info_gain,
                    reason=f"High novelty ({novelty:.2f}), new skill needed"
                )
            return EvolutionDecision(EvolutionDecision.NOOP,
                                     reason=f"High novelty but poor outcome")
        
        # Case 2: Moderately novel → BRANCH (preserve parent, create variant)
        if novelty >= self.NOVELTY_THRESHOLD_BRANCH:
            if outcome >= 0.4:
                return EvolutionDecision(
                    EvolutionDecision.BRANCH, target_id=best_id,
                    similarity=best_sim, novelty=novelty, info_gain=info_gain,
                    reason=f"Moderate novelty ({novelty:.2f}), branching"
                )
            return EvolutionDecision(EvolutionDecision.NOOP,
                                     reason="Moderate novelty + low outcome")
        
        # Case 3: Low novelty (similar to existing skill)
        # Sub-case 3a: Successful → STRENGTHEN (but respect confidence)
        if outcome >= 0.5:
            if net_info > self.INFO_GAIN_MIN_UPDATE or state.confidence < 0.5:
                return EvolutionDecision(
                    EvolutionDecision.STRENGTHEN, target_id=best_id,
                    similarity=best_sim, novelty=novelty, info_gain=info_gain,
                    reason=f"Low novelty + success, net_info={net_info:.3f}"
                )
            return EvolutionDecision(EvolutionDecision.NOOP,
                                     reason=f"Success but info_gain ({info_gain:.3f}) < cost ({update_cost:.3f})")
        
        # Sub-case 3b: Partial → REFINE (add constraints)
        if outcome >= 0.1:
            return EvolutionDecision(
                EvolutionDecision.REFINE, target_id=best_id,
                similarity=best_sim, novelty=novelty, info_gain=info_gain,
                reason=f"Partial outcome ({outcome:.2f}), refining"
            )
        
        # Sub-case 3c: Failure → check for ROLLBACK
        state.perf_history.append(outcome)
        recent_perfs = state.perf_history[-3:]
        if len(recent_perfs) >= 3 and all(p < 0.2 for p in recent_perfs):
            if state.cumulative_drift > self.DRIFT_ROLLBACK_THRESHOLD:
                return EvolutionDecision(
                    EvolutionDecision.ROLLBACK, target_id=best_id,
                    similarity=best_sim, novelty=novelty,
                    reason=f"3 consecutive failures + high drift ({state.cumulative_drift:.2f})"
                )
            if state.experience_count > 5:
                return EvolutionDecision(
                    EvolutionDecision.RETIRE, target_id=best_id,
                    similarity=best_sim,
                    reason="Persistent failure, skill is not useful"
                )
        
        return EvolutionDecision(EvolutionDecision.NOOP,
                                 reason=f"Failure, watching (streak={len(recent_perfs)})")

    # ================================================================
    # Execute Evolution (with EMA embedding update)
    # ================================================================

    def execute(
        self,
        decision: EvolutionDecision,
        new_skill: Skill,
        task_id: str = "",
        benchmark: str = "",
        outcome: float = 0.0,
        experience_embedding: np.ndarray | None = None,
    ) -> None:
        """Execute evolution with proper learning dynamics."""
        self.stats["decisions"][decision.action] += 1
        self.stats["total_steps"] += 1

        if experience_embedding is None:
            experience_embedding = self.embedder.embed(
                f"{new_skill.name} {new_skill.description}"[:500]
            )

        if decision.action == EvolutionDecision.CREATE:
            self._create(new_skill, decision, task_id, outcome, experience_embedding)
        elif decision.action == EvolutionDecision.STRENGTHEN:
            self._strengthen(new_skill, decision, task_id, benchmark, outcome, experience_embedding)
        elif decision.action == EvolutionDecision.REFINE:
            self._refine(new_skill, decision, task_id, benchmark, outcome)
        elif decision.action == EvolutionDecision.BRANCH:
            self._branch(new_skill, decision, task_id, outcome, experience_embedding)
        elif decision.action == EvolutionDecision.RETIRE:
            self._retire(decision, task_id, benchmark)
        elif decision.action == EvolutionDecision.ROLLBACK:
            self._rollback(decision, task_id, benchmark)

    def _create(self, skill: Skill, decision: EvolutionDecision,
                task_id: str, outcome: float, embedding: np.ndarray):
        """Create new skill with initial learning state."""
        # Create in versioned library
        self.library.create(skill, reason=decision.reason, task_id=task_id, performance=outcome)
        self.embedder.add_to_index(skill.skill_id, f"{skill.name} {skill.description}")

        # Initialize learning state
        state = SkillState(
            skill_id=skill.skill_id,
            centroid=embedding.copy(),
            experience_count=1,
            confidence=0.1,  # Low confidence initially
            learning_rate=1.0,
            perf_history=[outcome],
            perf_ema=outcome,
            layer0_summary=skill.description[:200],
            layer1_details=skill.description,
            layer2_examples=[task_id],
            created_at=time.time(),
            last_used_at=time.time(),
            last_updated_at=time.time(),
        )
        state.confidence = state.compute_confidence()
        self._states[skill.skill_id] = state

        logger.info(f"[Evo:CREATE] '{skill.name}' v1 (outcome={outcome:.2f}, confidence={state.confidence:.2f})")

    def _strengthen(self, new_skill: Skill, decision: EvolutionDecision,
                    task_id: str, benchmark: str, outcome: float, embedding: np.ndarray):
        """Strengthen via EMA embedding update + layered compression."""
        state = self._states.get(decision.target_id)
        if not state or state.centroid is None:
            return

        # === EMA update of centroid (the actual "learning") ===
        alpha = state.compute_learning_rate()
        old_centroid = state.centroid.copy()
        state.centroid = (1 - alpha) * state.centroid + alpha * embedding
        # Normalize
        norm = np.linalg.norm(state.centroid)
        if norm > 0:
            state.centroid = state.centroid / norm

        # Compute drift from this update
        drift = 1.0 - float(np.dot(old_centroid, state.centroid))
        state.drift_history.append(drift)
        state.cumulative_drift = sum(state.drift_history[-10:])

        # Update learning dynamics
        state.experience_count += 1
        state.learning_rate = state.compute_learning_rate()
        state.perf_history.append(outcome)
        state.perf_ema = 0.7 * state.perf_ema + 0.3 * outcome
        state.confidence = state.compute_confidence()
        state.last_updated_at = time.time()
        state.layer2_examples.append(task_id)

        # Update versioned library (text-level merge for retrieval)
        history = self.library.get_history(decision.target_id)
        if history and history.latest:
            current = Skill.model_validate(history.latest_skill)
            old_desc = current.description

            # Layered compression update
            current.description = self._compress_with_new(
                current.description, new_skill.description, state.confidence
            )
            current.source_tasks.append(task_id)
            current.success_rate = state.perf_ema

            # Update layer0 (compressed summary for prompt)
            state.layer0_summary = self._generate_summary(current, state)
            state.layer1_details = current.description

            self.library.evolve(
                decision.target_id, current, change_type="strengthen",
                reason=f"EMA update α={alpha:.3f}, drift={drift:.4f}",
                task_id=task_id, performance=outcome,
            )
            self.embedder.add_to_index(decision.target_id, f"{current.name} {current.description}")

            self.recorder.record_merge(
                skill_id=decision.target_id, skill_name=current.name,
                content_before=old_desc[:200], content_after=current.description[:200],
                merged_with=new_skill.name,
                rationale=f"EMA strengthen: α={alpha:.3f}, drift={drift:.4f}, confidence={state.confidence:.2f}",
                task_id=task_id, benchmark=benchmark,
            )

        self.stats["total_info_gained"] += decision.info_gain
        self.stats["total_drift"] += drift

        logger.info(
            f"[Evo:STRENGTHEN] v{state.experience_count} "
            f"(α={alpha:.3f}, drift={drift:.4f}, conf={state.confidence:.2f})"
        )

    def _refine(self, new_skill: Skill, decision: EvolutionDecision,
                task_id: str, benchmark: str, outcome: float):
        """Refine: add constraints without changing centroid significantly."""
        state = self._states.get(decision.target_id)
        if not state:
            return

        history = self.library.get_history(decision.target_id)
        if not history or not history.latest:
            return

        current = Skill.model_validate(history.latest_skill)
        old_desc = current.description

        # Add constraints (refinement = precision, not direction change)
        for c in new_skill.constraints:
            if c not in current.constraints:
                current.constraints.append(c)
        current.constraints = current.constraints[:8]

        # Track performance but DON'T change centroid much
        state.perf_history.append(outcome)
        state.perf_ema = 0.7 * state.perf_ema + 0.3 * outcome
        state.confidence = state.compute_confidence()
        state.experience_count += 1
        state.last_updated_at = time.time()

        # Update layer1 details with constraints
        constraint_text = "; ".join(current.constraints[-3:])
        state.layer1_details = f"{current.description}\nConstraints: {constraint_text}"

        self.library.evolve(
            decision.target_id, current, change_type="refine",
            reason=decision.reason, task_id=task_id, performance=outcome,
        )

        self.recorder.record_update(
            skill_id=decision.target_id, skill_name=current.name,
            content_before=old_desc[:200], content_after=current.description[:200],
            rationale=f"Refine: added {len(new_skill.constraints)} constraints, outcome={outcome:.2f}",
            task_id=task_id, benchmark=benchmark,
        )

        logger.info(f"[Evo:REFINE] v{state.experience_count} (+{len(new_skill.constraints)} constraints)")

    def _branch(self, new_skill: Skill, decision: EvolutionDecision,
                task_id: str, outcome: float, embedding: np.ndarray):
        """Branch: create a new lineage, preserving parent."""
        new_skill.metadata["branched_from"] = decision.target_id
        self.library.create(
            new_skill, reason=f"Branch from {decision.target_id[:8]} (novelty={decision.novelty:.2f})",
            task_id=task_id, performance=outcome,
        )
        self.embedder.add_to_index(new_skill.skill_id, f"{new_skill.name} {new_skill.description}")

        # New state for branch
        state = SkillState(
            skill_id=new_skill.skill_id,
            centroid=embedding.copy(),
            experience_count=1,
            confidence=0.1,
            learning_rate=1.0,
            perf_history=[outcome],
            perf_ema=outcome,
            layer0_summary=new_skill.description[:200],
            layer1_details=new_skill.description,
            layer2_examples=[task_id],
            created_at=time.time(),
            last_used_at=time.time(),
            last_updated_at=time.time(),
        )
        self._states[new_skill.skill_id] = state

        logger.info(f"[Evo:BRANCH] '{new_skill.name}' from {decision.target_id[:8]}")

    def _retire(self, decision: EvolutionDecision, task_id: str, benchmark: str):
        """Retire: don't delete, just lower retrieval weight to ~0."""
        state = self._states.get(decision.target_id)
        if state:
            state.retrieval_weight = 0.01  # Near-zero but not deleted

        self.library.retire(decision.target_id, reason=decision.reason, task_id=task_id)
        self.embedder.remove_from_index(decision.target_id)

        history = self.library.get_history(decision.target_id)
        if history and history.latest:
            self.recorder.record_delete(
                skill_id=decision.target_id, skill_name=history.latest.name,
                content_before=history.latest.description[:200],
                rationale=decision.reason, task_id=task_id, benchmark=benchmark,
            )
        logger.info(f"[Evo:RETIRE] {decision.target_id[:8]} (data preserved in cold storage)")

    def _rollback(self, decision: EvolutionDecision, task_id: str, benchmark: str):
        """Rollback: restore centroid from best-performing version."""
        state = self._states.get(decision.target_id)
        history = self.library.get_history(decision.target_id)
        if not state or not history or len(history.versions) < 2:
            return

        # Find best-performing version
        best_v = max(history.versions, key=lambda v: v.performance_at_creation)
        if best_v.version == history.current_version:
            return

        # Rollback in library
        self.library.rollback(
            decision.target_id, to_version=best_v.version,
            reason=f"Rolling back: drift={state.cumulative_drift:.2f}"
        )

        # Reset state to earlier point
        state.cumulative_drift = 0.0
        state.drift_history = []
        state.confidence = 0.5  # Reset confidence (we're uncertain now)
        # Re-embed from best version text
        self.embedder.add_to_index(
            decision.target_id,
            f"{best_v.name} {best_v.description}"
        )
        state.centroid = self.embedder.embed(f"{best_v.name} {best_v.description}")

        logger.info(f"[Evo:ROLLBACK] → v{best_v.version} (drift reset, confidence={state.confidence:.2f})")

    # ================================================================
    # Retrieval (version-aware, confidence-weighted)
    # ================================================================

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[str, SkillState, float]]:
        """
        Retrieve skills weighted by similarity × retrieval_weight × confidence.
        
        Returns: [(skill_id, state, effective_score), ...]
        """
        results = self.embedder.search(query[:500], top_k=top_k * 2, threshold=0.05)
        if not results:
            return []

        scored = []
        for skill_id, sim in results:
            state = self._states.get(skill_id)
            if not state:
                scored.append((skill_id, None, sim))
                continue
            # Effective score = similarity × retrieval_weight
            effective = sim * state.retrieval_weight
            scored.append((skill_id, state, effective))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]

    def retrieve_with_context(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve skills with adaptive compression level.
        
        - High confidence skills → return layer0 (summary only, save tokens)
        - Low confidence skills → return layer1 (details, help model get it right)
        - On failure → expand to layer2 (full examples)
        """
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return ""

        lines = []
        for skill_id, state, score in results:
            if not state:
                continue

            history = self.library.get_history(skill_id)
            if not history or history.is_retired:
                continue

            # Adaptive compression: confidence determines detail level
            if state.confidence >= 0.7:
                # Mature skill → just summary (save tokens)
                lines.append(
                    f"[{state.layer0_summary[:150]}] "
                    f"(v{state.version}, conf={state.confidence:.0%})"
                )
            elif state.confidence >= 0.4:
                # Developing → summary + key constraints
                lines.append(f"[Skill v{state.version} ({state.maturity})]")
                lines.append(f"  {state.layer1_details[:250]}")
            else:
                # Low confidence → full details + examples
                lines.append(f"[Skill v{state.version} (UNSTABLE, needs verification)]")
                lines.append(f"  {state.layer1_details[:300]}")
                if state.layer2_examples:
                    lines.append(f"  Examples: {state.layer2_examples[-2:]}")

            lines.append("")

        return "\n".join(lines)

    # ================================================================
    # Helpers
    # ================================================================

    def _compress_with_new(self, existing: str, new: str, confidence: float) -> str:
        """
        Merge descriptions with compression proportional to confidence.
        High confidence → keep existing, barely change.
        Low confidence → incorporate more from new experience.
        """
        # Weight of new info decreases with confidence
        new_weight = 1.0 - confidence

        if new_weight < 0.2:
            # High confidence: only add truly new sentences
            existing_sentences = set(existing.split(". "))
            new_sentences = set(new.split(". "))
            novel = new_sentences - existing_sentences
            if novel:
                addition = ". ".join(list(novel)[:1])  # Add at most 1 new sentence
                return f"{existing}. {addition}"[:600]
            return existing
        else:
            # Low confidence: more aggressive merge
            sentences = set(existing.split(". ")) | set(new.split(". "))
            return ". ".join(list(sentences)[:6])[:600]

    def _generate_summary(self, skill: Skill, state: SkillState) -> str:
        """Generate layer0 compressed summary."""
        # Shorter for mature skills, longer for developing ones
        max_len = 100 if state.confidence > 0.7 else 200
        summary = f"{skill.name}: {skill.description[:max_len]}"
        if skill.constraints:
            summary += f" [{skill.constraints[0][:50]}]"
        return summary

    # ================================================================
    # Stats & Introspection
    # ================================================================

    def get_learning_summary(self) -> dict:
        """Full summary of the learning state."""
        return {
            "total_skills": len(self._states),
            "active_skills": sum(1 for s in self._states.values() if s.retrieval_weight > 0.1),
            "mature_skills": sum(1 for s in self._states.values() if s.confidence >= 0.7),
            "avg_confidence": (
                sum(s.confidence for s in self._states.values()) / len(self._states)
                if self._states else 0
            ),
            "avg_version": (
                sum(s.experience_count for s in self._states.values()) / len(self._states)
                if self._states else 0
            ),
            "total_drift": self.stats["total_drift"],
            "total_info_gained": self.stats["total_info_gained"],
            "decisions": self.stats["decisions"],
        }
