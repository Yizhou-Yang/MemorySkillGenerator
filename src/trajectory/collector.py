"""
Trajectory collector.

Drives a ReAct agent through benchmark tasks and records the full
thought / action / observation interaction trajectory.

The collector forces multi-step reasoning by:
1. Requiring the agent to decompose the problem first.
2. Providing simulated observations that challenge the agent.
3. Asking the agent to verify before giving a final answer.
4. Injecting deliberate noise (errors, dead-ends, retries) to test
   the denoising capability of different skill induction variants.
This produces rich trajectories (10-20 steps) that differentiate
the three skill induction variants.
"""

from __future__ import annotations

import random
from typing import Any

from loguru import logger

from src.models import StepType, TaskType, Trajectory, TrajectoryStep
from src.utils.llm import LLMClient


class TrajectoryCollector:
    """
    Trajectory collector.

    Drives an agent to execute benchmark tasks and records every
    interaction step.  Forces multi-step reasoning to produce rich
    trajectories that differentiate skill induction variants.
    """

    def __init__(self, llm_client: LLMClient, config: dict[str, Any] | None = None) -> None:
        self.llm_client = llm_client
        self.config = config or {}
        self.max_steps: int = self.config.get("max_steps", 15)

    def collect(
        self,
        task_id: str,
        task_description: str,
        task_type: TaskType = TaskType.OTHER,
        context: str = "",
    ) -> Trajectory:
        """
        Execute a task and collect the trajectory.

        Args:
            task_id: Unique task identifier.
            task_description: Task description text.
            task_type: Task type enum value.
            context: Additional context (e.g. dialogue history).

        Returns:
            The complete interaction trajectory.
        """
        logger.info(f"Starting trajectory collection: task_id={task_id}")

        trajectory = Trajectory(
            task_id=task_id,
            task_description=task_description,
            task_type=task_type,
        )

        system_prompt = self._build_system_prompt()
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]

        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}"})

        messages.append({"role": "user", "content": f"Task:\n{task_description}"})

        min_steps_before_answer = 5  # Force at least 5 rounds of reasoning
        noise_count = 0  # Track how many noise injections we've done

        for step_idx in range(self.max_steps):
            try:
                response = self.llm_client.chat(messages)

                parsed_steps = self._parse_response(response, len(trajectory.steps))
                trajectory.steps.extend(parsed_steps)

                # Only allow finishing after minimum steps
                if self._is_finished(response) and step_idx >= min_steps_before_answer:
                    trajectory.success = True
                    trajectory.final_answer = self._extract_answer(response)
                    logger.info(
                        f"Task completed: task_id={task_id}, "
                        f"steps={trajectory.num_steps}"
                    )
                    break

                messages.append({"role": "assistant", "content": response})

                # Inject noise at steps 2 and 4 to create heavy error patterns
                # This is the KEY differentiator between variants:
                # - traj→skill will include ALL noise in the skill
                # - memory→skill should filter it out during compression
                # - hybrid→skill should partially filter it
                if step_idx in (2, 4) and noise_count < 2:
                    noise_obs = self._inject_noise(task_description, noise_count)
                    messages.append(
                        {"role": "user", "content": f"Observation: {noise_obs}"}
                    )
                    trajectory.steps.append(
                        TrajectoryStep(
                            step_id=len(trajectory.steps),
                            step_type=StepType.ERROR,
                            content=noise_obs,
                        )
                    )
                    # Also add a redundant repetition step (more noise)
                    if noise_count == 0:
                        redundant = (
                            "Let me reconsider... Actually, I think my earlier "
                            "approach might have been on the right track after "
                            "all. Let me re-examine the same evidence again."
                        )
                        trajectory.steps.append(
                            TrajectoryStep(
                                step_id=len(trajectory.steps),
                                step_type=StepType.THOUGHT,
                                content=redundant,
                            )
                        )
                    noise_count += 1
                    continue

                # Generate a challenging observation to force deeper reasoning
                observation = self._generate_observation(
                    response, step_idx, task_description
                )
                messages.append(
                    {"role": "user", "content": f"Observation: {observation}"}
                )
                trajectory.steps.append(
                    TrajectoryStep(
                        step_id=len(trajectory.steps),
                        step_type=StepType.OBSERVATION,
                        content=observation,
                    )
                )

            except Exception as exc:
                logger.error(f"Trajectory collection error (step {step_idx}): {exc}")
                trajectory.steps.append(
                    TrajectoryStep(
                        step_id=len(trajectory.steps),
                        step_type=StepType.ERROR,
                        content=str(exc),
                    )
                )
                break

        if not trajectory.success:
            # Extract answer from last response if possible
            if trajectory.steps:
                last_content = trajectory.steps[-1].content
                if last_content:
                    trajectory.final_answer = last_content[:200]
                    trajectory.success = True
            logger.warning(
                f"Task ended at max steps: task_id={task_id}, "
                f"steps={trajectory.num_steps}"
            )

        return trajectory

    def _build_system_prompt(self) -> str:
        """Build the agent system prompt that forces multi-step reasoning."""
        return (
            "You are a careful task-execution agent that reasons step by step.\n"
            "Follow the ReAct format strictly:\n\n"
            "Thought: [your reasoning about the current state]\n"
            "Action: [one specific action to take]\n\n"
            "IMPORTANT RULES:\n"
            "1. Do NOT give a final answer immediately. First decompose the "
            "problem into sub-questions.\n"
            "2. For each sub-question, state a Thought and an Action.\n"
            "3. After receiving observations, reflect on whether they are "
            "sufficient or if you need more information.\n"
            "4. Only when you have gathered enough evidence across multiple "
            "steps, give your final answer as:\n"
            "   Answer: [your final answer]\n"
            "5. If you make an error, acknowledge it and correct your approach.\n"
            "6. Show your work: explain WHY each step leads to the next."
        )

    def _generate_observation(
        self, response: str, step_idx: int, task_description: str
    ) -> str:
        """
        Generate a simulated observation that challenges the agent.

        Uses a progression of observation types to force rich trajectories:
        - Step 0: Ask for problem decomposition
        - Step 1: Challenge with alternative interpretation
        - Step 2: (noise injection handled separately)
        - Step 3: Ask agent to recover from the error
        - Step 4: Ask for verification / confidence check
        - Step 5: Push toward synthesis and final answer
        - Step 6+: Push toward final answer
        """
        if step_idx == 0:
            return (
                "Good start. Before proceeding, please decompose this problem "
                "into 2-3 specific sub-questions that need to be answered. "
                "For each sub-question, identify what information you need."
            )
        elif step_idx == 1:
            return (
                "I see your decomposition. Now work through each sub-question "
                "one at a time. For the first sub-question, what specific "
                "evidence or reasoning supports your conclusion? "
                "Consider if there might be an alternative interpretation."
            )
        elif step_idx == 3:
            return (
                "The previous approach had an issue. Please reconsider your "
                "strategy. What went wrong? Identify the error and try a "
                "different approach. Sometimes the first intuition is wrong."
            )
        elif step_idx == 4:
            return (
                "Good recovery. Now address the remaining sub-questions. "
                "Also consider: is there any information that contradicts "
                "your current reasoning? Double-check your key assumptions "
                "and verify each step of your logic."
            )
        elif step_idx == 5:
            return (
                "You have gathered evidence for the sub-questions. "
                "Now synthesise your findings: how do the answers to the "
                "sub-questions combine to answer the original question? "
                "State your confidence level (high/medium/low) and why. "
                "If confident, give your final Answer."
            )
        else:
            return (
                "Please provide your final answer now. "
                "Format: Answer: [your answer]"
            )

    def _inject_noise(self, task_description: str, noise_count: int = 0) -> str:
        """
        Inject deliberate noise into the trajectory.

        Different noise types at different injection points:
        - noise_count=0 (step 2): logical error / wrong assumption
        - noise_count=1 (step 4): dead-end / distractor confusion

        This creates error/dead-end patterns that test the denoising
        capability of different skill induction variants:
        - traj_to_skill will include this noise in the skill (bad)
        - memory_to_skill should filter it out during compression (good)
        - hybrid_to_skill should partially filter it (medium)
        """
        if noise_count == 0:
            # First noise: logical error
            templates = [
                (
                    "ERROR: The previous reasoning contains a logical flaw. "
                    "You assumed a connection that doesn't exist in the given "
                    "information. Specifically, you may have confused correlation "
                    "with causation. Please re-examine your evidence carefully "
                    "and identify which assumption was incorrect."
                ),
                (
                    "CORRECTION: An earlier step produced an incorrect "
                    "intermediate result. This is a common pitfall — the error "
                    "propagated through subsequent reasoning. Go back to the "
                    "point of error and redo the calculation or inference."
                ),
            ]
        else:
            # Second noise: dead-end / distractor
            templates = [
                (
                    "WARNING: Your approach is heading toward a dead end. "
                    "The information you're relying on may be from a distractor "
                    "paragraph, not a supporting fact. Step back and reconsider "
                    "which sources are actually relevant to the question. "
                    "HINT: Try a completely different decomposition strategy."
                ),
                (
                    "RETRY NEEDED: Your intermediate conclusion appears to "
                    "contradict known facts. This often happens when key details "
                    "are overlooked. Re-read the relevant context and look for "
                    "details you may have missed on the first pass. "
                    "Consider whether you're solving the right sub-problem."
                ),
            ]
        return random.choice(templates)

    def _parse_response(
        self, response: str, base_step_id: int
    ) -> list[TrajectoryStep]:
        """Parse an agent response into a list of trajectory steps."""
        parsed_steps: list[TrajectoryStep] = []
        current_id = base_step_id

        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("Thought:"):
                parsed_steps.append(
                    TrajectoryStep(
                        step_id=current_id,
                        step_type=StepType.THOUGHT,
                        content=line[len("Thought:"):].strip(),
                    )
                )
                current_id += 1
            elif line.startswith("Action:"):
                parsed_steps.append(
                    TrajectoryStep(
                        step_id=current_id,
                        step_type=StepType.ACTION,
                        content=line[len("Action:"):].strip(),
                    )
                )
                current_id += 1
            elif line.startswith("Answer:"):
                parsed_steps.append(
                    TrajectoryStep(
                        step_id=current_id,
                        step_type=StepType.ACTION,
                        content=f"Final Answer: {line[len('Answer:'):].strip()}",
                    )
                )
                current_id += 1

        # If no structured steps were parsed, treat the whole response as a thought
        if not parsed_steps:
            parsed_steps.append(
                TrajectoryStep(
                    step_id=base_step_id,
                    step_type=StepType.THOUGHT,
                    content=response.strip(),
                )
            )

        return parsed_steps

    def _is_finished(self, response: str) -> bool:
        """Check whether the agent has finished the task."""
        return "Answer:" in response

    def _extract_answer(self, response: str) -> str:
        """Extract the final answer from the response."""
        for line in response.split("\n"):
            if line.strip().startswith("Answer:"):
                return line.strip()[len("Answer:"):].strip()
        return ""
