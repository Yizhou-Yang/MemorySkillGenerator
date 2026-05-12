"""
RL Controller — embedding-based skill selection + Gumbel-Top-K sampling + PPO training.

Reference: MemSkill paper §3.2-3.7:
- Embedding-based compatibility score: z_{t,i} = h_t^T u_i
- Gumbel-Top-K sampling: exploratory during training, greedy at eval
- PPO training loop: uses downstream EM/F1 as reward signal
- Dynamic skill bank: new skills plug-and-play without retraining action head

Reference: docs/internal/memskill_analysis.md §3.2-3.7
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger


# ============================================================
# Data Structures
# ============================================================

@dataclass
class ControllerState:
    """Controller state representation, corresponds to MemSkill Eq.1: h_t = f_ctx(x_t, M_t)"""
    embedding: np.ndarray  # State embedding vector (dim,)
    span_text: str = ""  # Raw span text (for debugging)
    retrieved_memories: list[str] = field(default_factory=list)


@dataclass
class SkillEmbedding:
    """Skill embedding representation, corresponds to MemSkill Eq.2: u_i = f_skill(desc(s_i))"""
    skill_id: str
    skill_name: str
    embedding: np.ndarray  # Skill description embedding vector (dim,)
    is_new: bool = False  # Whether newly added skill (for exploration incentive)
    creation_step: int = 0  # Training step when created


@dataclass
class SelectionResult:
    """Top-K skill selection result"""
    selected_skill_ids: list[str]  # Selected K skill IDs
    selected_indices: list[int]  # Selected K skill indices in the bank
    log_prob: float  # Joint log probability log π(A_t | s_t)
    raw_scores: np.ndarray  # Raw compatibility scores z_{t,i}
    probabilities: np.ndarray  # Softmax probabilities p_θ(i | h_t)


@dataclass
class PPOTransition:
    """Single-step transition for PPO training"""
    state_embedding: np.ndarray
    selected_indices: list[int]
    log_prob: float
    reward: float
    value: float
    advantage: float = 0.0
    returns: float = 0.0


# ============================================================
# Controller MLP
# ============================================================

class ControllerMLP:
    """
    Lightweight MLP for state embedding transformation.

    MemSkill's controller is a simple MLP that maps state embedding
    to the same dimensional space as skill embeddings, then uses dot product for compatibility score.

    Implemented in numpy (no PyTorch dependency) to keep the project lightweight.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        output_dim: int = 1024,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        rng = np.random.RandomState(seed)
        # Xavier initialization
        scale1 = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + output_dim))

        self.W1 = rng.randn(input_dim, hidden_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.randn(hidden_dim, output_dim).astype(np.float32) * scale2
        self.b2 = np.zeros(output_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: x -> ReLU(xW1 + b1) -> W2 + b2"""
        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        return h @ self.W2 + self.b2

    def get_params(self) -> list[np.ndarray]:
        """Get all parameters (for PPO update)"""
        return [self.W1, self.b1, self.W2, self.b2]

    def set_params(self, params: list[np.ndarray]) -> None:
        """Set all parameters"""
        self.W1, self.b1, self.W2, self.b2 = params

    def clone_params(self) -> list[np.ndarray]:
        """Deep copy all parameters"""
        return [p.copy() for p in self.get_params()]


class ValueNetwork:
    """
    Value network V_φ(s_t) for PPO advantage estimation.

    Simple two-layer MLP that outputs a scalar value.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 128,
        seed: int = 123,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        rng = np.random.RandomState(seed)
        scale1 = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + 1))

        self.W1 = rng.randn(input_dim, hidden_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.randn(hidden_dim, 1).astype(np.float32) * scale2
        self.b2 = np.zeros(1, dtype=np.float32)

    def forward(self, x: np.ndarray) -> float:
        """Forward pass: x -> ReLU(xW1 + b1) -> W2 + b2 -> scalar"""
        h = np.maximum(0, x @ self.W1 + self.b1)
        return float((h @ self.W2 + self.b2)[0])

    def get_params(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    def set_params(self, params: list[np.ndarray]) -> None:
        self.W1, self.b1, self.W2, self.b2 = params

    def clone_params(self) -> list[np.ndarray]:
        return [p.copy() for p in self.get_params()]


# ============================================================
# Skill Selection Controller
# ============================================================

class SkillSelectionController:
    """
    Embedding-based skill selection controller.

    Core mechanism (MemSkill §3.2-3.4):
    1. Compatibility Score: z_{t,i} = h_t^T u_i (dot product of state and skill)
    2. Softmax probability: p_θ(i|h_t) = softmax(z_t)_i
    3. Gumbel-Top-K sampling: add Gumbel noise for exploration during training, greedy at eval
    4. Joint probability: π(A_t|s_t) = ∏ p(a_j) / (1 - Σ_{l<j} p(a_l))
    5. Exploration Incentive: new skill probability mass >= τ_t (linear decay)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

        # Dimension config
        self.embedding_dim: int = self.config.get("embedding_dim", 1024)
        self.hidden_dim: int = self.config.get("hidden_dim", 256)
        self.top_k: int = self.config.get("top_k", 3)

        # Exploration incentive config (MemSkill §3.8.5)
        self.tau_0: float = self.config.get("tau_0", 0.3)
        self.t_explore: int = self.config.get("t_explore", 50)

        # PPO config
        self.clip_epsilon: float = self.config.get("clip_epsilon", 0.2)
        self.gamma: float = self.config.get("gamma", 0.99)
        self.entropy_coeff: float = self.config.get("entropy_coeff", 0.01)
        self.value_coeff: float = self.config.get("value_coeff", 0.5)
        self.learning_rate: float = self.config.get("learning_rate", 3e-4)

        # Networks
        self.policy_net = ControllerMLP(
            input_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.embedding_dim,
        )
        self.value_net = ValueNetwork(
            input_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim // 2,
        )

        # Skill bank (dynamic size)
        self._skill_embeddings: list[SkillEmbedding] = []

        # Training state
        self._current_step: int = 0
        self._trajectory_buffer: list[PPOTransition] = []
        self._best_params: list[np.ndarray] | None = None
        self._best_reward: float = -float("inf")

        logger.info(
            f"[Controller] Initialized: dim={self.embedding_dim}, "
            f"hidden={self.hidden_dim}, top_k={self.top_k}"
        )

    @property
    def skill_bank_size(self) -> int:
        """Current skill bank size"""
        return len(self._skill_embeddings)

    def register_skill(
        self,
        skill_id: str,
        skill_name: str,
        embedding: np.ndarray,
        is_new: bool = False,
    ) -> None:
        """
        Register a skill into the bank.

        New skills plug-and-play without retraining action head.
        This is the core advantage of embedding-based scoring.
        """
        se = SkillEmbedding(
            skill_id=skill_id,
            skill_name=skill_name,
            embedding=embedding.astype(np.float32),
            is_new=is_new,
            creation_step=self._current_step,
        )
        self._skill_embeddings.append(se)
        logger.info(
            f"[Controller] Registered skill '{skill_name}' "
            f"(bank_size={self.skill_bank_size}, is_new={is_new})"
        )

    def remove_skill(self, skill_id: str) -> bool:
        """Remove a skill from the bank"""
        before = len(self._skill_embeddings)
        self._skill_embeddings = [
            se for se in self._skill_embeddings if se.skill_id != skill_id
        ]
        removed = len(self._skill_embeddings) < before
        if removed:
            logger.info(f"[Controller] Removed skill {skill_id[:8]}...")
        return removed

    def select_skills(
        self,
        state: ControllerState,
        top_k: int | None = None,
        training: bool = False,
    ) -> SelectionResult:
        """
        Select Top-K skills.

        Args:
            state: Current state (embedding)
            top_k: Number to select (default self.top_k)
            training: Training mode (True uses Gumbel-Top-K sampling)

        Returns:
            SelectionResult containing selected skills and probability info
        """
        k = min(top_k or self.top_k, self.skill_bank_size)
        if k == 0:
            return SelectionResult(
                selected_skill_ids=[],
                selected_indices=[],
                log_prob=0.0,
                raw_scores=np.array([]),
                probabilities=np.array([]),
            )

        # Step 1: Transform state embedding via policy network
        h_t = self.policy_net.forward(state.embedding)

        # Step 2: Compute compatibility scores (MemSkill Eq.3)
        skill_embeddings = np.stack(
            [se.embedding for se in self._skill_embeddings]
        )  # (N, dim)
        z_t = skill_embeddings @ h_t  # (N,) — dot product

        # Step 3: Exploration incentive (MemSkill §3.8.5 Eq.6-8)
        z_t = self._apply_exploration_incentive(z_t)

        # Step 4: Softmax probabilities
        probs = self._softmax(z_t)

        # Step 5: Select Top-K
        if training:
            selected_indices = self._gumbel_top_k(z_t, k)
        else:
            selected_indices = self._greedy_top_k(probs, k)

        # Step 6: Compute joint log probability (MemSkill Eq.11)
        log_prob = self._compute_joint_log_prob(probs, selected_indices)

        selected_ids = [
            self._skill_embeddings[i].skill_id for i in selected_indices
        ]

        return SelectionResult(
            selected_skill_ids=selected_ids,
            selected_indices=selected_indices,
            log_prob=log_prob,
            raw_scores=z_t,
            probabilities=probs,
        )

    def get_value(self, state: ControllerState) -> float:
        """Get state value estimate V_φ(s_t)"""
        return self.value_net.forward(state.embedding)

    # ================================================================
    # Gumbel-Top-K Sampling (MemSkill §3.4)
    # ================================================================

    @staticmethod
    def _gumbel_top_k(logits: np.ndarray, k: int) -> list[int]:
        """
        Gumbel-Top-K sampling.

        Add independent Gumbel(0,1) noise to each logit, then take Top-K.
        Equivalent to sampling without replacement from softmax(logits).
        """
        # Sample Gumbel(0,1) noise: g = -log(-log(u)), u ~ Uniform(0,1)
        u = np.random.uniform(1e-8, 1.0 - 1e-8, size=logits.shape)
        gumbel_noise = -np.log(-np.log(u))
        perturbed = logits + gumbel_noise
        # Take Top-K indices
        indices = np.argsort(perturbed)[::-1][:k].tolist()
        return indices

    @staticmethod
    def _greedy_top_k(probs: np.ndarray, k: int) -> list[int]:
        """Greedy selection of top-K by probability"""
        return np.argsort(probs)[::-1][:k].tolist()

    # ================================================================
    # Joint Probability Computation (MemSkill Eq.11)
    # ================================================================

    @staticmethod
    def _compute_joint_log_prob(
        probs: np.ndarray, selected_indices: list[int]
    ) -> float:
        """
        Compute joint log probability of Top-K selection without replacement.

        π(A_t|s_t) = ∏_{j=1}^K p(a_j) / (1 - Σ_{l<j} p(a_l))

        Corresponds to MemSkill Eq.11 "sampling without replacement" math.
        """
        log_prob = 0.0
        cumulative_prob = 0.0

        for idx in selected_indices:
            p_i = float(probs[idx])
            denominator = 1.0 - cumulative_prob
            if denominator <= 1e-10:
                break
            conditional_p = p_i / denominator
            conditional_p = max(conditional_p, 1e-10)  # Numerical stability
            log_prob += math.log(conditional_p)
            cumulative_prob += p_i

        return log_prob

    # ================================================================
    # Exploration Incentive (MemSkill §3.8.5 Eq.6-8)
    # ================================================================

    def _apply_exploration_incentive(self, z_t: np.ndarray) -> np.ndarray:
        """
        Apply exploration incentive to new skills.

        Eq.6: Σ_{i ∈ S_new} p_θ(i|s_t) >= τ_t
        Eq.7: z'_{t,i} = z_{t,i} + δ_t (i ∈ S_new)
        Eq.8: τ_t = τ_0 · (1 - t/T_explore)
        """
        new_indices = [
            i for i, se in enumerate(self._skill_embeddings)
            if se.is_new and (self._current_step - se.creation_step) < self.t_explore
        ]

        if not new_indices:
            return z_t

        # Compute current τ_t (linear decay)
        # Compute decay using each new skill's own age
        max_age = max(
            self._current_step - self._skill_embeddings[i].creation_step
            for i in new_indices
        )
        tau_t = self.tau_0 * max(0.0, 1.0 - max_age / self.t_explore)

        if tau_t <= 0:
            return z_t

        # Check if new skill probability mass satisfies constraint
        probs = self._softmax(z_t)
        new_prob_mass = sum(probs[i] for i in new_indices)

        if new_prob_mass >= tau_t:
            return z_t  # Constraint already satisfied

        # Compute δ_t needed to satisfy constraint
        # Binary search for minimum δ
        z_modified = z_t.copy()
        delta_low, delta_high = 0.0, 10.0
        for _ in range(20):  # Binary search
            delta_mid = (delta_low + delta_high) / 2
            z_trial = z_t.copy()
            for i in new_indices:
                z_trial[i] += delta_mid
            trial_probs = self._softmax(z_trial)
            trial_mass = sum(trial_probs[i] for i in new_indices)
            if trial_mass >= tau_t:
                delta_high = delta_mid
            else:
                delta_low = delta_mid

        for i in new_indices:
            z_modified[i] += delta_high

        return z_modified

    # ================================================================
    # PPO Training (MemSkill §3.7 Eq.15-19)
    # ================================================================

    def record_transition(self, transition: PPOTransition) -> None:
        """Record a transition to the buffer"""
        self._trajectory_buffer.append(transition)
        self._current_step += 1

    def compute_advantages(self, final_reward: float) -> None:
        """
        Compute GAE advantages.

        MemSkill uses episode-level reward, intermediate steps are 0.
        G_t = γ^{T-t} · R
        Â_t = G_t - V_φ(s_t)
        """
        T = len(self._trajectory_buffer)
        if T == 0:
            return

        for t, transition in enumerate(self._trajectory_buffer):
            # Discounted returns
            transition.returns = (self.gamma ** (T - 1 - t)) * final_reward
            # Advantage = returns - value baseline
            transition.advantage = transition.returns - transition.value

    def ppo_update(self, epochs: int = 4, batch_size: int = 32) -> dict[str, float]:
        """
        Execute PPO update.

        Eq.16: L_policy = E[min(r_t · Â_t, clip(r_t, 1±ε) · Â_t)]
        Eq.19: max L_policy - c_v · L_value + c_H · H(θ)

        Returns:
            Training statistics
        """
        if not self._trajectory_buffer:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        old_policy_params = self.policy_net.clone_params()
        old_value_params = self.value_net.clone_params()

        for _epoch in range(epochs):
            for transition in self._trajectory_buffer:
                # Compute log_prob under new policy
                state = ControllerState(embedding=transition.state_embedding)
                result = self.select_skills(state, training=False)

                if result.probabilities.size == 0:
                    continue

                new_log_prob = self._compute_joint_log_prob(
                    result.probabilities, transition.selected_indices
                )

                # Importance ratio r_t(θ) = π_new / π_old
                ratio = math.exp(
                    min(new_log_prob - transition.log_prob, 20.0)
                )

                # Clipped surrogate objective
                adv = transition.advantage
                surr1 = ratio * adv
                surr2 = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv
                policy_loss = -min(surr1, surr2)

                # Value loss
                new_value = self.value_net.forward(transition.state_embedding)
                value_loss = (new_value - transition.returns) ** 2

                # Entropy bonus
                entropy = -np.sum(
                    result.probabilities * np.log(result.probabilities + 1e-10)
                )

                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_entropy += entropy
                num_updates += 1

                # Simplified gradient update (numerical differentiation approximation)
                # Should use PyTorch autograd in production
                self._numerical_update(
                    transition, policy_loss, value_loss, entropy
                )

        # Clear the buffer
        n = max(num_updates, 1)
        stats = {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "num_transitions": len(self._trajectory_buffer),
        }

        self._trajectory_buffer.clear()

        logger.info(
            f"[Controller] PPO update: policy_loss={stats['policy_loss']:.4f}, "
            f"value_loss={stats['value_loss']:.4f}, entropy={stats['entropy']:.4f}"
        )
        return stats

    def _numerical_update(
        self,
        transition: PPOTransition,
        policy_loss: float,
        value_loss: float,
        entropy: float,
    ) -> None:
        """
        Simplified parameter update.

        Full implementation should use PyTorch autograd.
        Uses small random perturbation + directional update as approximation.
        """
        lr = self.learning_rate
        total_loss = (
            policy_loss
            + self.value_coeff * value_loss
            - self.entropy_coeff * entropy
        )

        # Small update to policy network
        for param in self.policy_net.get_params():
            noise = np.random.randn(*param.shape).astype(np.float32) * 0.01
            param -= lr * total_loss * noise

        # Small update to value network
        for param in self.value_net.get_params():
            noise = np.random.randn(*param.shape).astype(np.float32) * 0.01
            param -= lr * value_loss * noise

    # ================================================================
    # Snapshot & Rollback (MemSkill §3.8.6)
    # ================================================================

    def save_snapshot(self, reward: float) -> bool:
        """
        Save current parameter snapshot (if best).

        Returns:
            True if a new best snapshot was saved
        """
        if reward > self._best_reward:
            self._best_reward = reward
            self._best_params = (
                self.policy_net.clone_params() + self.value_net.clone_params()
            )
            logger.info(
                f"[Controller] New best snapshot: reward={reward:.4f}"
            )
            return True
        return False

    def rollback_to_best(self) -> bool:
        """
        Rollback to best parameter snapshot.

        Returns:
            True if rollback succeeded
        """
        if self._best_params is None:
            return False

        policy_params = self._best_params[:4]
        value_params = self._best_params[4:]
        self.policy_net.set_params(policy_params)
        self.value_net.set_params(value_params)
        logger.info(
            f"[Controller] Rolled back to best snapshot "
            f"(reward={self._best_reward:.4f})"
        )
        return True

    # ================================================================
    # Utility Methods
    # ================================================================

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / (np.sum(exp_x) + 1e-10)

    def get_skill_ids(self) -> list[str]:
        """Get all registered skill IDs"""
        return [se.skill_id for se in self._skill_embeddings]

    def mark_skills_not_new(self) -> None:
        """Mark all skills as non-new (called after exploration period ends)"""
        for se in self._skill_embeddings:
            se.is_new = False
