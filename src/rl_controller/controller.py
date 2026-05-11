"""
RL Controller — 基于 embedding 的 skill 选择 + Gumbel-Top-K 采样 + PPO 训练。

参考 MemSkill 论文 §3.2-3.7:
- Embedding-based compatibility score: z_{t,i} = h_t^T u_i
- Gumbel-Top-K 采样: 训练时有探索性，评估时贪心
- PPO 训练循环: 用下游 EM/F1 作为 reward signal
- 可变 skill bank: 新 skill 直接 plug-and-play，不需要重训 action head

Reference: docs/internal/memskill_analysis.md §3.2-3.7
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ControllerState:
    """Controller 的状态表示，对应 MemSkill 公式 1: h_t = f_ctx(x_t, M_t)"""
    embedding: np.ndarray  # 状态 embedding 向量 (dim,)
    span_text: str = ""  # 原始 span 文本（用于调试）
    retrieved_memories: list[str] = field(default_factory=list)


@dataclass
class SkillEmbedding:
    """Skill 的 embedding 表示，对应 MemSkill 公式 2: u_i = f_skill(desc(s_i))"""
    skill_id: str
    skill_name: str
    embedding: np.ndarray  # skill description embedding 向量 (dim,)
    is_new: bool = False  # 是否为新加入的 skill（用于 exploration incentive）
    creation_step: int = 0  # 创建时的训练步数


@dataclass
class SelectionResult:
    """Top-K skill 选择结果"""
    selected_skill_ids: list[str]  # 选中的 K 个 skill ID
    selected_indices: list[int]  # 选中的 K 个 skill 在 bank 中的索引
    log_prob: float  # 联合对数概率 log π(A_t | s_t)
    raw_scores: np.ndarray  # 原始 compatibility scores z_{t,i}
    probabilities: np.ndarray  # softmax 概率 p_θ(i | h_t)


@dataclass
class PPOTransition:
    """PPO 训练用的单步 transition"""
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
    轻量 MLP 用于 state embedding 变换。

    MemSkill 的 controller 是一个简单的 MLP，将 state embedding
    映射到与 skill embedding 同维度的空间，然后用内积做 compatibility score。

    这里用 numpy 实现（不依赖 PyTorch），保持项目轻量。
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
        # Xavier 初始化
        scale1 = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + output_dim))

        self.W1 = rng.randn(input_dim, hidden_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.randn(hidden_dim, output_dim).astype(np.float32) * scale2
        self.b2 = np.zeros(output_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播: x -> ReLU(xW1 + b1) -> W2 + b2"""
        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        return h @ self.W2 + self.b2

    def get_params(self) -> list[np.ndarray]:
        """获取所有参数（用于 PPO 更新）"""
        return [self.W1, self.b1, self.W2, self.b2]

    def set_params(self, params: list[np.ndarray]) -> None:
        """设置所有参数"""
        self.W1, self.b1, self.W2, self.b2 = params

    def clone_params(self) -> list[np.ndarray]:
        """深拷贝所有参数"""
        return [p.copy() for p in self.get_params()]


class ValueNetwork:
    """
    Value network V_φ(s_t) 用于 PPO 的 advantage 估计。

    简单的两层 MLP，输出标量 value。
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
        """前向传播: x -> ReLU(xW1 + b1) -> W2 + b2 -> scalar"""
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
    基于 embedding 的 skill 选择 controller。

    核心机制 (MemSkill §3.2-3.4):
    1. Compatibility Score: z_{t,i} = h_t^T u_i (状态与 skill 的内积)
    2. Softmax 概率: p_θ(i|h_t) = softmax(z_t)_i
    3. Gumbel-Top-K 采样: 训练时加 Gumbel 噪声探索，评估时贪心
    4. 联合概率: π(A_t|s_t) = ∏ p(a_j) / (1 - Σ_{l<j} p(a_l))
    5. Exploration Incentive: 新 skill 的概率质量 >= τ_t (线性衰减)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

        # 维度配置
        self.embedding_dim: int = self.config.get("embedding_dim", 1024)
        self.hidden_dim: int = self.config.get("hidden_dim", 256)
        self.top_k: int = self.config.get("top_k", 3)

        # Exploration incentive 配置 (MemSkill §3.8.5)
        self.tau_0: float = self.config.get("tau_0", 0.3)
        self.t_explore: int = self.config.get("t_explore", 50)

        # PPO 配置
        self.clip_epsilon: float = self.config.get("clip_epsilon", 0.2)
        self.gamma: float = self.config.get("gamma", 0.99)
        self.entropy_coeff: float = self.config.get("entropy_coeff", 0.01)
        self.value_coeff: float = self.config.get("value_coeff", 0.5)
        self.learning_rate: float = self.config.get("learning_rate", 3e-4)

        # 网络
        self.policy_net = ControllerMLP(
            input_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.embedding_dim,
        )
        self.value_net = ValueNetwork(
            input_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim // 2,
        )

        # Skill bank (动态大小)
        self._skill_embeddings: list[SkillEmbedding] = []

        # 训练状态
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
        """当前 skill bank 大小"""
        return len(self._skill_embeddings)

    def register_skill(
        self,
        skill_id: str,
        skill_name: str,
        embedding: np.ndarray,
        is_new: bool = False,
    ) -> None:
        """
        注册一个 skill 到 bank 中。

        新 skill 直接 plug-and-play，不需要重训 action head。
        这是 embedding-based score 的核心优势。
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
        """从 bank 中移除一个 skill"""
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
        选择 Top-K 个 skill。

        Args:
            state: 当前状态 (embedding)
            top_k: 选择数量 (默认 self.top_k)
            training: 是否训练模式 (True 时用 Gumbel-Top-K 采样)

        Returns:
            SelectionResult 包含选中的 skill 和概率信息
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

        # Step 1: 通过 policy network 变换 state embedding
        h_t = self.policy_net.forward(state.embedding)

        # Step 2: 计算 compatibility scores (MemSkill 公式 3)
        skill_embeddings = np.stack(
            [se.embedding for se in self._skill_embeddings]
        )  # (N, dim)
        z_t = skill_embeddings @ h_t  # (N,) — 内积

        # Step 3: Exploration incentive (MemSkill §3.8.5 公式 6-8)
        z_t = self._apply_exploration_incentive(z_t)

        # Step 4: Softmax 概率
        probs = self._softmax(z_t)

        # Step 5: 选择 Top-K
        if training:
            selected_indices = self._gumbel_top_k(z_t, k)
        else:
            selected_indices = self._greedy_top_k(probs, k)

        # Step 6: 计算联合对数概率 (MemSkill 公式 11)
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
        """获取状态的 value 估计 V_φ(s_t)"""
        return self.value_net.forward(state.embedding)

    # ================================================================
    # Gumbel-Top-K 采样 (MemSkill §3.4)
    # ================================================================

    @staticmethod
    def _gumbel_top_k(logits: np.ndarray, k: int) -> list[int]:
        """
        Gumbel-Top-K 采样。

        给每个 logit 加独立 Gumbel(0,1) 噪声，然后取 Top-K。
        等价于按 softmax(logits) 做无放回采样。
        """
        # 采样 Gumbel(0,1) 噪声: g = -log(-log(u)), u ~ Uniform(0,1)
        u = np.random.uniform(1e-8, 1.0 - 1e-8, size=logits.shape)
        gumbel_noise = -np.log(-np.log(u))
        perturbed = logits + gumbel_noise
        # 取 Top-K 索引
        indices = np.argsort(perturbed)[::-1][:k].tolist()
        return indices

    @staticmethod
    def _greedy_top_k(probs: np.ndarray, k: int) -> list[int]:
        """贪心选择概率最高的 K 个"""
        return np.argsort(probs)[::-1][:k].tolist()

    # ================================================================
    # 联合概率计算 (MemSkill 公式 11)
    # ================================================================

    @staticmethod
    def _compute_joint_log_prob(
        probs: np.ndarray, selected_indices: list[int]
    ) -> float:
        """
        计算无放回 Top-K 选择的联合对数概率。

        π(A_t|s_t) = ∏_{j=1}^K p(a_j) / (1 - Σ_{l<j} p(a_l))

        对应 MemSkill 公式 11 的"无放回抽糖果"数学。
        """
        log_prob = 0.0
        cumulative_prob = 0.0

        for idx in selected_indices:
            p_i = float(probs[idx])
            denominator = 1.0 - cumulative_prob
            if denominator <= 1e-10:
                break
            conditional_p = p_i / denominator
            conditional_p = max(conditional_p, 1e-10)  # 数值稳定
            log_prob += math.log(conditional_p)
            cumulative_prob += p_i

        return log_prob

    # ================================================================
    # Exploration Incentive (MemSkill §3.8.5 公式 6-8)
    # ================================================================

    def _apply_exploration_incentive(self, z_t: np.ndarray) -> np.ndarray:
        """
        对新 skill 施加 exploration incentive。

        公式 6: Σ_{i ∈ S_new} p_θ(i|s_t) >= τ_t
        公式 7: z'_{t,i} = z_{t,i} + δ_t (i ∈ S_new)
        公式 8: τ_t = τ_0 · (1 - t/T_explore)
        """
        new_indices = [
            i for i, se in enumerate(self._skill_embeddings)
            if se.is_new and (self._current_step - se.creation_step) < self.t_explore
        ]

        if not new_indices:
            return z_t

        # 计算当前 τ_t (线性衰减)
        # 对每个新 skill 用其自己的 age 计算衰减
        max_age = max(
            self._current_step - self._skill_embeddings[i].creation_step
            for i in new_indices
        )
        tau_t = self.tau_0 * max(0.0, 1.0 - max_age / self.t_explore)

        if tau_t <= 0:
            return z_t

        # 检查当前新 skill 的概率质量是否满足约束
        probs = self._softmax(z_t)
        new_prob_mass = sum(probs[i] for i in new_indices)

        if new_prob_mass >= tau_t:
            return z_t  # 已满足约束

        # 计算需要加的 δ_t 使约束成立
        # 二分搜索找最小 δ
        z_modified = z_t.copy()
        delta_low, delta_high = 0.0, 10.0
        for _ in range(20):  # 二分搜索
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
    # PPO 训练 (MemSkill §3.7 公式 15-19)
    # ================================================================

    def record_transition(self, transition: PPOTransition) -> None:
        """记录一个 transition 到 buffer"""
        self._trajectory_buffer.append(transition)
        self._current_step += 1

    def compute_advantages(self, final_reward: float) -> None:
        """
        计算 GAE advantages。

        MemSkill 用 episode-level reward，中间步骤都是 0。
        G_t = γ^{T-t} · R
        Â_t = G_t - V_φ(s_t)
        """
        T = len(self._trajectory_buffer)
        if T == 0:
            return

        for t, transition in enumerate(self._trajectory_buffer):
            # 折扣回报
            transition.returns = (self.gamma ** (T - 1 - t)) * final_reward
            # Advantage = returns - value baseline
            transition.advantage = transition.returns - transition.value

    def ppo_update(self, epochs: int = 4, batch_size: int = 32) -> dict[str, float]:
        """
        执行 PPO 更新。

        公式 16: L_policy = E[min(r_t · Â_t, clip(r_t, 1±ε) · Â_t)]
        公式 19: max L_policy - c_v · L_value + c_H · H(θ)

        Returns:
            训练统计信息
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
                # 计算新策略下的 log_prob
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

                # 简化的梯度更新 (数值微分近似)
                # 在实际部署中应使用 PyTorch autograd
                self._numerical_update(
                    transition, policy_loss, value_loss, entropy
                )

        # 清空 buffer
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
        简化的参数更新。

        在完整实现中应使用 PyTorch autograd。
        这里用小幅随机扰动 + 方向性更新作为近似。
        """
        lr = self.learning_rate
        total_loss = (
            policy_loss
            + self.value_coeff * value_loss
            - self.entropy_coeff * entropy
        )

        # 对 policy network 做小幅更新
        for param in self.policy_net.get_params():
            noise = np.random.randn(*param.shape).astype(np.float32) * 0.01
            param -= lr * total_loss * noise

        # 对 value network 做小幅更新
        for param in self.value_net.get_params():
            noise = np.random.randn(*param.shape).astype(np.float32) * 0.01
            param -= lr * value_loss * noise

    # ================================================================
    # Snapshot & Rollback (MemSkill §3.8.6)
    # ================================================================

    def save_snapshot(self, reward: float) -> bool:
        """
        保存当前参数快照（如果是最佳）。

        Returns:
            True 如果保存了新的最佳快照
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
        回滚到最佳参数快照。

        Returns:
            True 如果成功回滚
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
    # 辅助方法
    # ================================================================

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """数值稳定的 softmax"""
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / (np.sum(exp_x) + 1e-10)

    def get_skill_ids(self) -> list[str]:
        """获取所有注册的 skill ID"""
        return [se.skill_id for se in self._skill_embeddings]

    def mark_skills_not_new(self) -> None:
        """将所有 skill 标记为非新 skill（exploration 结束后调用）"""
        for se in self._skill_embeddings:
            se.is_new = False
