"""ALFWorld environment wrapper.

Provides a clean reset/step interface over alfworld.agents.environment.alfred_tw_env.AlfredTWEnv,
hiding the batch dimension (always batch_size=1 for our serial eval).

Usage:
    env = AlfworldEnv(split="valid_unseen", max_steps=50)
    print(env.num_games)  # 134

    obs, info = env.reset(game_idx=0)
    print(obs)              # natural-language room description + task
    print(info["task"])     # e.g. "clean some ladle and put it in diningtable."
    print(info["actions"])  # list of admissible commands

    obs, info = env.step("go to cabinet 1")
    print(info["won"])       # True only when task succeeds
    print(info["done"])      # True if max_steps reached or won
"""

from __future__ import annotations

import os
import re
from typing import Any

# Default ALFWORLD_DATA path (override via env var)
DEFAULT_ALFWORLD_DATA = "/root/workspace/SkillForge/.venv_alfworld/data"
if "ALFWORLD_DATA" not in os.environ:
    os.environ["ALFWORLD_DATA"] = DEFAULT_ALFWORLD_DATA


def _build_alfworld_config(split: str) -> dict:
    """Minimal AlfredTWEnv config (alfworld 0.4.2 schema)."""
    return {
        "env": {
            "type": "AlfredTWEnv",
            "regen_game_files": False,
            "domain_randomization": False,
            "task_types": [1, 2, 3, 4, 5, 6],
            "expert_timeout_steps": 150,
            "expert_type": "handcoded",
            "goal_desc_human_anns_prob": 0.0,
            "hybrid": {"start_eps": 100000, "thor_prob": 0.5, "eval_mode": "tw"},
            "thor": {
                "screen_width": 300, "screen_height": 300,
                "smooth_nav": False, "save_frames_to_disk": False,
                "save_frames_path": "./videos/",
            },
        },
        "dataset": {
            "data_path": "$ALFWORLD_DATA/json_2.1.1/train",
            "eval_id_data_path": "$ALFWORLD_DATA/json_2.1.1/valid_seen",
            "eval_ood_data_path": "$ALFWORLD_DATA/json_2.1.1/valid_unseen",
            "num_train_games": -1,
            "num_eval_games": -1,
        },
        "logic": {
            "domain": "$ALFWORLD_DATA/logic/alfred.pddl",
            "grammar": "$ALFWORLD_DATA/logic/alfred.twl2",
        },
        "general": {
            "random_seed": 42,
            "training_method": "dqn",
            "training": {"batch_size": 1},
            "evaluate": {"batch_size": 1, "env": {"type": "AlfredTWEnv"}},
        },
        "rl": {
            "action_space": "admissible",
            "training": {"max_nb_steps_per_episode": 50},
        },
    }


# Map alfworld split name → AlfredTWEnv train_eval kwarg
SPLIT_MAP = {
    "train": "train",
    "valid_seen": "eval_in_distribution",      # 140 games
    "valid_unseen": "eval_out_of_distribution",  # 134 games
}


def extract_task_description(initial_obs: str) -> str:
    """Pull out the 'Your task is to: ...' line from the alfworld initial obs."""
    m = re.search(r"Your task is to:\s*(.+)", initial_obs)
    if m:
        return m.group(1).strip().rstrip(".")
    return ""


def extract_room_state(initial_obs: str) -> str:
    """Strip the welcome banner and task line, leaving the room description."""
    lines = []
    for ln in initial_obs.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("-=") or s.startswith("Your task"):
            continue
        lines.append(s)
    return " ".join(lines)


class AlfworldEnv:
    """Single-task ALFWorld wrapper. Always batch_size=1.

    NOTE: A single AlfworldEnv instance should be reset()-then-finished one
    game at a time. Construct once, then call `env.reset(game_idx=k)` and
    `env.step(action)` repeatedly for each game.
    """

    def __init__(
        self,
        split: str = "valid_unseen",
        max_steps: int = 50,
        max_traj_steps_override: int | None = None,
    ):
        if split not in SPLIT_MAP:
            raise ValueError(f"Unknown split {split!r}; valid: {list(SPLIT_MAP)}")
        self.split = split
        self.max_steps = max_steps

        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv  # type: ignore

        cfg = _build_alfworld_config(split)
        if max_traj_steps_override:
            cfg["rl"]["training"]["max_nb_steps_per_episode"] = max_traj_steps_override
        self._cfg = cfg
        self._env_type = AlfredTWEnv(cfg, train_eval=SPLIT_MAP[split])
        self.num_games = self._env_type.num_games
        self._batch_env = self._env_type.init_env(batch_size=1)

        # internal state for the current episode
        self._step_count = 0
        self._task: str = ""
        self._won: bool = False

    # ---------------------------------------------------------------- API

    def list_tasks(self) -> list[str]:
        """Return the list of game files (ordered, length == num_games).

        Each is an absolute path to a .tw-pddl game file. The 'task' is the
        natural-language goal which we extract on reset.
        """
        return list(self._env_type.game_files)

    def reset(self, game_idx: int | None = None) -> tuple[str, dict[str, Any]]:
        """Reset the env. If game_idx is given, jump to that game.

        Note: alfworld's TextworldBatchGymEnv consumes games in a round-robin
        from the gamefiles list. To pick a specific game we re-init with a
        single-element list.
        """
        if game_idx is not None:
            target = self._env_type.game_files[game_idx]
            # rebuild a 1-element env pointed at this specific game
            self._batch_env = self._env_type.init_env(batch_size=1)
            # Override gamefiles so reset() picks our target
            try:
                self._batch_env.gamefiles = [target]
            except AttributeError:
                # fall back: textworld stores gamefiles as a property
                pass

        obs, info = self._batch_env.reset()
        ob = obs[0]
        admissible = info.get("admissible_commands", [[]])[0] if info.get("admissible_commands") else []
        won_arr = info.get("won", [False])
        won = bool(won_arr[0]) if won_arr is not None else False

        self._step_count = 0
        self._task = extract_task_description(ob)
        self._won = won

        return ob, {
            "task": self._task,
            "room": extract_room_state(ob),
            "actions": list(admissible),
            "won": won,
            "done": won,
            "step": 0,
            "raw_info_keys": list(info.keys()),
        }

    def step(self, action: str) -> tuple[str, dict[str, Any]]:
        """Take one action. Returns (obs, info)."""
        self._step_count += 1
        try:
            obs, scores, dones, info = self._batch_env.step([action])
        except Exception as exc:
            # Some malformed actions can crash textworld; treat as no-op + done
            return f"[ENV ERROR: {exc}]", {
                "task": self._task,
                "actions": [],
                "won": self._won,
                "done": True,
                "step": self._step_count,
                "score": 0,
                "error": str(exc),
            }

        ob = obs[0]
        admissible = info.get("admissible_commands", [[]])[0] if info.get("admissible_commands") else []
        won_arr = info.get("won", [False])
        won = bool(won_arr[0]) if won_arr is not None else False
        self._won = won
        done_native = bool(dones[0]) if dones is not None else False
        done = won or done_native or self._step_count >= self.max_steps

        return ob, {
            "task": self._task,
            "actions": list(admissible),
            "won": won,
            "done": done,
            "step": self._step_count,
            "score": float(scores[0]) if scores is not None else 0.0,
        }

    @property
    def task(self) -> str:
        return self._task

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def won(self) -> bool:
        return self._won


def task_type_from_gamefile(gamefile: str) -> str:
    """Classify a gamefile path into one of 6 ALFWorld task types.

    Used by `run_alfworld_eval.py` for stratified sampling.
    """
    g = gamefile.lower()
    # alfworld trial directory names encode the task type
    if "pick_and_place_simple" in g:
        return "pick_place"
    if "pick_two_obj_and_place" in g:
        return "pick_two"
    if "look_at_obj_in_light" in g:
        return "examine"
    if "pick_clean_then_place" in g or "pick_clean" in g:
        return "clean"
    if "pick_heat_then_place" in g or "pick_heat" in g:
        return "heat"
    if "pick_cool_then_place" in g or "pick_cool" in g:
        return "cool"
    return "other"
