"""
Gaia2 Adapter — translates Gaia2/ARE trace format into SkillForge generic format.
"""

from __future__ import annotations

import json
from typing import Any

from src.adapters import BaseAdapter, GenericAction, GenericTask


class Gaia2Adapter(BaseAdapter):
    """Adapter for Meta ARE (Gaia2) benchmark traces."""

    def parse_task(self, scenario_path: str, trace_path: str) -> GenericTask:
        task_desc = self.get_task_description(scenario_path)
        oracle = self.extract_oracle(scenario_path)
        agent = self.extract_tool_calls(trace_path)

        with open(scenario_path) as f:
            scenario = json.load(f)
        scenario_id = scenario.get('metadata', {}).get('definition', {}).get('scenario_id', '')

        return GenericTask(
            task_id=scenario_id or scenario_path,
            task_description=task_desc,
            oracle_actions=oracle,
            agent_actions=agent,
        )

    def evaluate(self, task: GenericTask) -> dict[str, float]:
        """Compute ER, Precision, F1."""
        def sig(action: GenericAction) -> str:
            return f"{action.tool}:{json.dumps(action.args, sort_keys=True)}"

        o_sigs = [sig(a) for a in task.oracle_actions]
        a_sigs = [sig(a) for a in task.agent_actions]

        matched = 0
        used = set()
        for os_ in o_sigs:
            for j, as_ in enumerate(a_sigs):
                if j not in used and os_ == as_:
                    matched += 1
                    used.add(j)
                    break

        er = matched / len(o_sigs) if o_sigs else 0
        prec = matched / len(a_sigs) if a_sigs else 0
        f1 = 2 * prec * er / (prec + er) if (prec + er) > 0 else 0

        return {"er": er, "precision": prec, "f1": f1,
                "oracle_n": len(o_sigs), "agent_n": len(a_sigs), "matched": matched}

    def extract_tool_calls(self, trace_path: str) -> list[GenericAction]:
        with open(trace_path) as f:
            trace = json.load(f)

        actions = []
        for i, ev in enumerate(trace.get('completed_events', [])):
            if ev.get('event_type') != 'AGENT':
                continue
            act = ev.get('action', {})
            func = act.get('function', '')
            if func in ('send_message_to_user', 'send_message_to_agent'):
                continue

            tool = f"{act.get('app', '')}__{func}"
            args = act.get('args', [])
            arg_dict = {a['name']: a.get('value', '') for a in args} if isinstance(args, list) else {}

            actions.append(GenericAction(
                tool=tool, args=arg_dict, step_index=i,
            ))
        return actions

    def extract_oracle(self, scenario_path: str) -> list[GenericAction]:
        with open(scenario_path) as f:
            scenario = json.load(f)

        actions = []
        for i, ev in enumerate(scenario.get('events', [])):
            if ev.get('event_type') != 'AGENT':
                continue
            act = ev.get('action', {})
            tool = f"{act.get('app', '')}__{act.get('function', '')}"
            args = act.get('args', [])
            arg_dict = {a['name']: a.get('value', '') for a in args} if isinstance(args, list) else {}

            actions.append(GenericAction(
                tool=tool, args=arg_dict, step_index=i,
            ))
        return actions

    def get_task_description(self, scenario_path: str) -> str:
        with open(scenario_path) as f:
            scenario = json.load(f)

        for ev in scenario.get('events', []):
            if ev.get('event_type') == 'USER':
                args = ev.get('action', {}).get('args', [])
                for a in args:
                    if a.get('name') == 'content':
                        return a.get('value', '')
        return ""
