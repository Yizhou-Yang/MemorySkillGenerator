"""
Trajectory collector.

Drives a ReAct agent through benchmark tasks and records the full
thought / action / observation interaction trajectory.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.models import StepType, TaskType, Trajectory, TrajectoryStep
from src.utils.llm import LLMClient


class TrajectoryCollector:
    """
    Trajectory collector.

    Drives an agent to execute benchmark tasks and records every
    interaction step.  Uses a simple ReAct loop in the MVP phase.
    """

    def __init__(self, llm_client: LLMClient, config: dict[str, Any] | None = None) -> None:
        self.llm_client = llm_client
        self.config = config or {}
        self.max_steps: int = self.config.get("max_steps", 50)

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

        for step_idx in range(self.max_steps):
            try:
                # Agent generates thought + action
                response = self.llm_client.chat(messages)

                # Parse response into trajectory steps
                parsed_steps = self._parse_response(response, step_idx)
                trajectory.steps.extend(parsed_steps)

                # Check whether the task is finished
                if self._is_finished(response):
                    trajectory.success = True
                    trajectory.final_answer = self._extract_answer(response)
                    logger.info(
                        f"Task completed: task_id={task_id}, "
                        f"steps={trajectory.num_steps}"
                    )
                    break

                # Append response to conversation history
                messages.append({"role": "assistant", "content": response})

                # Simulate environment feedback (simplified in MVP)
                observation = self._get_observation(response)
                if observation:
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
            logger.warning(
                f"Task not completed: task_id={task_id}, "
                f"steps={trajectory.num_steps}"
            )

        return trajectory

    def _build_system_prompt(self) -> str:
        """Build the agent system prompt."""
        return (
            "You are a task-execution agent. "
            "Follow the ReAct format step by step:\n\n"
            "Thought: [your reasoning]\n"
            "Action: [the action to take]\n"
            "Answer: [your final answer, if ready]\n\n"
            "Rules:\n"
            "1. Execute only one action at a time.\n"
            "2. Analyse the observation carefully before deciding the next step.\n"
            "3. When you are confident in the answer, prefix it with 'Answer:'."
        )

    def _parse_response(
        self, response: str, base_step_id: int
    ) -> list[TrajectoryStep]:
        """Parse an agent response into a list of trajectory steps."""
        parsed_steps: list[TrajectoryStep] = []
        current_id = base_step_id

        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("Thought:"):
                parsed_steps.append(
                    TrajectoryStep(
                        step_id=current_id,
                        step_type=StepType.THOUGHT,
                        content=line[len("Thought:") :].strip(),
                    )
                )
                current_id += 1
            elif line.startswith("Action:"):
                parsed_steps.append(
                    TrajectoryStep(
                        step_id=current_id,
                        step_type=StepType.ACTION,
                        content=line[len("Action:") :].strip(),
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
                return line.strip()[len("Answer:") :].strip()
        return ""

    def _get_observation(self, response: str) -> str:
        """
        Obtain an environment observation.

        Returns an empty string in the MVP phase.
        Will return real observations once a tool-execution environment
        (e.g. LangGraph tools) is integrated.
        """
        # TODO: Integrate a real tool-execution environment (e.g. LangGraph tools)
        return ""
