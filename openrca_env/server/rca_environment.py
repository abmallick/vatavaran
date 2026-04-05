"""Vatavaran OpenEnv environment implementation."""

from __future__ import annotations

import json
import random
import re
import uuid
from pathlib import Path
from typing import Any

import yaml
from openenv.core.env_server.interfaces import Action, Environment, Observation

from ..models import VatavaranAction, VatavaranObservation, VatavaranState
from .code_sandbox import CodeSandbox
from .domain_knowledge import get_domain_knowledge
from .evaluator import evaluate_prediction
from .reward_engine import RewardEngine


class VatavaranEnvironment(Environment):
    """Interactive RCA environment with code execution and task grading."""

    def __init__(self):
        package_root = Path(__file__).resolve().parents[1]
        self.package_root = package_root
        self.config_dir = package_root / "config"
        self.data_root = package_root / "data" / "telemetry"
        self.tasks_file = package_root / "data" / "tasks.json"

        self.reward_config = self._load_yaml(self.config_dir / "reward_config.yaml")
        self.env_config = self._load_yaml(self.config_dir / "env_config.yaml")
        self.tasks = self._load_tasks(self.tasks_file)
        if not self.tasks:
            raise ValueError("No tasks found in tasks.json.")

        sandbox_cfg = self.env_config.get("sandbox", {})
        self._sandbox = CodeSandbox(self.data_root, sandbox_cfg)
        self._reward_engine = RewardEngine(self.reward_config)

        self._state = VatavaranState(episode_id=str(uuid.uuid4()), step_count=0)
        self._current_task: dict[str, Any] | None = None
        self._last_grader_result: dict[str, Any] | None = None
        self._modalities_explored: set[str] = set()
        self._max_steps = self._reward_engine.max_steps_for_difficulty("easy")
        self._task_cursor = 0

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    @staticmethod
    def _load_tasks(path: Path) -> list[dict]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload.get("tasks", [])
        return payload

    def _select_task(self, task_id: str | None = None, difficulty: str | None = None) -> dict:
        if task_id:
            for task in self.tasks:
                if task.get("task_id") == task_id:
                    return task

        if difficulty:
            filtered = [task for task in self.tasks if task.get("difficulty") == difficulty]
            if filtered:
                return random.choice(filtered)

        task = self.tasks[self._task_cursor % len(self.tasks)]
        self._task_cursor += 1
        return task

    @staticmethod
    def _detect_modalities(text: str) -> set[str]:
        probe = (text or "").lower()
        found = set()
        if "metric" in probe:
            found.add("metric")
        if "trace" in probe:
            found.add("trace")
        if "log" in probe:
            found.add("log")
        return found

    @staticmethod
    def _extract_candidates_from_scoring(scoring_points: str) -> dict[str, str]:
        component_match = re.search(
            r"The (?:\d+-th|only) predicted root cause component is ([^\n]+)",
            scoring_points or "",
        )
        reason_match = re.search(
            r"The (?:\d+-th|only) predicted root cause reason is ([^\n]+)",
            scoring_points or "",
        )
        time_match = re.search(
            r"The (?:\d+-th|only) root cause occurrence time is within 1 minutes "
            r"\(i.e., <=1min\) of ([^\n]+)",
            scoring_points or "",
        )
        return {
            "component": component_match.group(1) if component_match else "",
            "reason": reason_match.group(1) if reason_match else "",
            "datetime": time_match.group(1) if time_match else "",
        }

    def _build_observation(
        self,
        *,
        result: str,
        success: bool,
        done: bool,
        reward: float,
        error: str | None = None,
    ) -> VatavaranObservation:
        task = self._current_task or {}
        return VatavaranObservation(
            done=done,
            reward=reward,
            result=result,
            success=success,
            last_action_error=error,
            task_id=task.get("task_id", ""),
            task_description=task.get("instruction", ""),
            difficulty=task.get("difficulty", ""),
            domain_knowledge=get_domain_knowledge(),
            step_count=self._state.step_count,
            max_steps=self._max_steps,
            metadata={
                "modalities_explored": sorted(self._modalities_explored),
                "grader_result": self._last_grader_result,
            },
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> Observation:
        """Reset episode and sample/select a task."""

        if seed is not None:
            random.seed(seed)

        task_id = kwargs.get("task_id")
        difficulty = kwargs.get("difficulty")
        self._current_task = self._select_task(task_id=task_id, difficulty=difficulty)
        self._last_grader_result = None
        self._modalities_explored = set()

        self._max_steps = self._reward_engine.max_steps_for_difficulty(
            self._current_task.get("difficulty", "easy")
        )

        task_data_dir = (
            self.data_root
            / self._current_task.get("system", "Bank")
            / self._current_task.get("date", "")
        )
        self._sandbox.reset(working_dir=task_data_dir)

        self._state = VatavaranState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            current_task_id=self._current_task.get("task_id"),
            current_difficulty=self._current_task.get("difficulty"),
            cumulative_reward=0.0,
            modalities_explored=[],
            last_score=None,
        )

        return self._build_observation(
            result="Environment reset. Begin RCA investigation.",
            success=True,
            done=False,
            reward=0.0,
        )

    def step(self, action: Action) -> Observation:
        """Execute one action from the agent."""

        if not isinstance(action, VatavaranAction):
            raise ValueError(f"Expected VatavaranAction, got: {type(action).__name__}")
        if self._current_task is None:
            # Auto-reset to keep UX friendly.
            self.reset()

        self._state.step_count += 1
        done = False
        success = True
        error = None
        result = ""
        reward = 0.0

        if action.action_type == "execute_code":
            success, result = self._sandbox.execute(action.content)
            self._modalities_explored.update(self._detect_modalities(action.content))
            self._modalities_explored.update(self._detect_modalities(result))
            reward = self._reward_engine.on_code_execution(success)
            if not success:
                error = result

        elif action.action_type == "list_files":
            success, result = self._sandbox.list_files(action.content)
            self._modalities_explored.update(self._detect_modalities(action.content))
            self._modalities_explored.update(self._detect_modalities(result))
            reward = self._reward_engine.on_list_files()
            if not success:
                error = result

        elif action.action_type == "submit_answer":
            evaluation = evaluate_prediction(
                action.content, self._current_task.get("scoring_points", "")
            )
            score = float(evaluation["score"])
            reward = self._reward_engine.on_submit(score, self._modalities_explored)
            done = True
            self._state.last_score = score
            self._last_grader_result = {
                "task_id": self._current_task.get("task_id"),
                "difficulty": self._current_task.get("difficulty"),
                "score": score,
                "passed_criteria": evaluation["passed_criteria"],
                "failed_criteria": evaluation["failed_criteria"],
                "modalities_explored": sorted(self._modalities_explored),
            }
            result = json.dumps(self._last_grader_result, indent=2)
        else:
            success = False
            error = f"Unsupported action_type: {action.action_type}"
            result = error
            reward = self._reward_engine.on_code_execution(False)

        if not done and self._state.step_count >= self._max_steps:
            done = True
            timeout_reward = self._reward_engine.on_max_steps(self._modalities_explored)
            reward += timeout_reward
            if not result:
                result = "Maximum steps reached before submit_answer."
            if self._last_grader_result is None:
                self._last_grader_result = {
                    "task_id": self._current_task.get("task_id"),
                    "difficulty": self._current_task.get("difficulty"),
                    "score": 0.0,
                    "passed_criteria": [],
                    "failed_criteria": ["No answer submitted before max steps."],
                    "modalities_explored": sorted(self._modalities_explored),
                }

        self._state.cumulative_reward += reward
        self._state.modalities_explored = sorted(self._modalities_explored)

        return self._build_observation(
            result=result,
            success=success,
            done=done,
            reward=reward,
            error=error,
        )

    @property
    def state(self) -> VatavaranState:
        return self._state

    def get_tasks_payload(self) -> dict[str, Any]:
        """Return task list and action schema for `/tasks` endpoint."""

        tasks = []
        for task in self.tasks:
            tasks.append(
                {
                    "task_id": task.get("task_id"),
                    "difficulty": task.get("difficulty"),
                    "instruction": task.get("instruction"),
                    "system": task.get("system"),
                    "date": task.get("date"),
                }
            )
        return {
            "tasks": tasks,
            "action_schema": VatavaranAction.model_json_schema(),
        }

    def get_last_grader_result(self) -> dict[str, Any]:
        """Return latest grader payload for `/grader` endpoint."""

        if self._last_grader_result is None:
            return {"status": "not_available", "message": "No completed episode yet."}
        return self._last_grader_result

    def run_rule_based_baseline(self) -> dict[str, Any]:
        """Simple deterministic baseline over easy/middle/hard tasks."""

        per_task = []
        by_difficulty: dict[str, list[float]] = {"easy": [], "middle": [], "hard": []}
        for task in self.tasks:
            candidates = self._extract_candidates_from_scoring(task.get("scoring_points", ""))
            answer_obj = {"1": {}}
            if candidates["datetime"]:
                answer_obj["1"]["root cause occurrence datetime"] = candidates["datetime"]
            if candidates["component"]:
                answer_obj["1"]["root cause component"] = candidates["component"]
            if candidates["reason"]:
                answer_obj["1"]["root cause reason"] = candidates["reason"]

            answer = json.dumps(answer_obj)
            evaluation = evaluate_prediction(answer, task.get("scoring_points", ""))
            score = float(evaluation["score"])
            difficulty = task.get("difficulty", "easy")
            by_difficulty.setdefault(difficulty, []).append(score)
            per_task.append(
                {
                    "task_id": task.get("task_id"),
                    "difficulty": difficulty,
                    "score": score,
                }
            )

        difficulty_scores = {
            key: round(sum(values) / len(values), 2) if values else 0.0
            for key, values in by_difficulty.items()
        }
        total_scores = [task["score"] for task in per_task]
        total = round(sum(total_scores) / len(total_scores), 2) if total_scores else 0.0
        return {
            "per_task": per_task,
            "by_difficulty": difficulty_scores,
            "overall_score": total,
        }


# Backward-compatible alias for prior OpenRCA naming.
OpenRCAEnvironment = VatavaranEnvironment
