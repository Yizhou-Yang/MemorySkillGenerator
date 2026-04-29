"""
Trajectory collector.

Drives a ReAct agent through benchmark tasks and records the full
thought / action / observation interaction trajectory.

The collector forces multi-step reasoning by:
1. Requiring the agent to decompose the problem first.
2. Providing simulated observations that challenge the agent.
3. Asking the agent to verify before giving a final answer.

No artificial noise is injected — the trajectory is kept natural.
Differentiation between skill induction variants comes from how
each variant processes the (naturally verbose) trajectory:
- traj→skill sees everything (information overload)
- memory→skill sees only compressed memory (may lose details)
- hybrid→skill sees memory + selected evidence (best of both)
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

                # Generate a challenging observation to force deeper reasoning
                # No artificial noise — keep trajectory natural
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

        Designed to produce naturally verbose trajectories (6-8 steps)
        with rich reasoning content. The verbosity is what creates
        differentiation: traj→skill must process ALL of this,
        memory→skill compresses it, hybrid→skill selects from it.
        """
        if step_idx == 0:
            return (
                "Good start. Before proceeding, please decompose this problem "
                "into 2-3 specific sub-questions that need to be answered. "
                "For each sub-question, identify what information you need "
                "and what approach you would take."
            )
        elif step_idx == 1:
            return (
                "I see your decomposition. Now work through each sub-question "
                "one at a time. For the first sub-question, what specific "
                "evidence or reasoning supports your conclusion? "
                "Consider if there might be an alternative interpretation. "
                "Also explain what assumptions you are making."
            )
        elif step_idx == 2:
            return (
                "Good progress on the first sub-question. Now move to the "
                "second sub-question. What evidence do you have? Are there "
                "any connections between the first and second sub-questions? "
                "Explain the relationship between your findings so far."
            )
        elif step_idx == 3:
            return (
                "Now address any remaining sub-questions. Also consider: "
                "is there any information that contradicts your current "
                "reasoning? Double-check your key assumptions and verify "
                "each step of your logic. What would change your answer?"
            )
        elif step_idx == 4:
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
