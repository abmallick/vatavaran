"""
Vatavaran Incident Response Environment — core Environment implementation.

Simulates production incidents in a microservice architecture.
The agent must diagnose root causes and apply correct remediations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ActionType, AlertInfo, IncidentAction, IncidentObservation
except ImportError:
    from models import ActionType, AlertInfo, IncidentAction, IncidentObservation

from .graders import grade_episode
from .simulation import (
    InfraSimulation,
    build_healthy_infrastructure,
    inject_task_easy,
    inject_task_hard,
    inject_task_medium,
)
from .tasks import ALL_TASKS, TaskDefinition

MAX_STEPS = 25

FAULT_INJECTORS = {
    "service_outage": inject_task_easy,
    "db_pool_exhaustion": inject_task_medium,
    "cascading_failure": inject_task_hard,
}


class IncidentResponseEnvironment(Environment):
    """
    An SRE incident response environment where an agent must diagnose and
    resolve production incidents across a simulated microservice architecture.

    Supports 3 tasks of increasing difficulty:
      - service_outage (easy): Single service OOM crash
      - db_pool_exhaustion (medium): Database config misconfiguration
      - cascading_failure (hard): Memory-leak-triggered cascade
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._sim: InfraSimulation | None = None
        self._task: TaskDefinition | None = None
        self._action_history: list[dict[str, Any]] = []
        self._done = False
        self._cumulative_reward = 0.0
        self._base_time = datetime(2026, 3, 27, 2, 0, 0)
        self._incident_resolved = False
        self._grader_result: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "service_outage", **kwargs: Any) -> IncidentObservation:
        """Reset environment for the specified task."""
        if task_id not in ALL_TASKS:
            task_id = "service_outage"

        self._task = ALL_TASKS[task_id]
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._action_history = []
        self._done = False
        self._cumulative_reward = 0.0
        self._incident_resolved = False
        self._grader_result = None

        services = build_healthy_infrastructure(self._base_time)
        injector = FAULT_INJECTORS[task_id]
        injector(services, self._base_time)
        self._sim = InfraSimulation(services)

        alert = AlertInfo(
            alert_id=f"INC-{uuid4().hex[:8].upper()}",
            severity=self._task.alert_severity,
            title=self._task.alert_title,
            description=self._task.alert_description,
            triggered_at=self._base_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            affected_services=self._task.affected_services,
        )

        return IncidentObservation(
            alert=alert,
            message=(
                f"You are the on-call SRE. An incident has been triggered.\n"
                f"Task: {self._task.name} ({self._task.difficulty})\n"
                f"Investigate and resolve the incident. You have {MAX_STEPS} steps."
            ),
            data=None,
            available_actions=[e.value for e in ActionType],
            services=self._sim.service_names(),
            step_count=0,
            max_steps=MAX_STEPS,
            incident_resolved=False,
            task_id=task_id,
            done=False,
            reward=0.0,
        )

    def step(self, action: IncidentAction) -> IncidentObservation:  # type: ignore[override]
        """Process one agent action and return the resulting observation."""
        if self._done or self._sim is None or self._task is None:
            return self._terminal_observation("Episode already ended. Call reset() to start a new one.")

        self._state.step_count += 1
        action_type = action.action_type.value if isinstance(action.action_type, ActionType) else str(action.action_type)
        target = action.target_service
        params = action.parameters

        action_record = {
            "action_type": action_type,
            "target_service": target,
            "parameters": params,
        }
        self._action_history.append(action_record)

        # Process the action in the simulation
        data, message = self._sim.process_action(action_type, target, params)

        # Compute step reward
        step_reward = self._compute_step_reward(action_type, target, params, data)
        self._cumulative_reward += step_reward

        # Check for resolution
        if action_type == "resolve":
            self._done = True
            self._incident_resolved = True
            self._grader_result = grade_episode(self._task, self._action_history, True)
            return IncidentObservation(
                message=f"Incident marked as resolved.\n\nGrader feedback:\n{self._grader_result['feedback']}",
                data={"grader": self._grader_result},
                available_actions=[],
                services=self._sim.service_names(),
                step_count=self._state.step_count,
                max_steps=MAX_STEPS,
                incident_resolved=True,
                task_id=self._task.task_id,
                done=True,
                reward=step_reward,
            )

        # Check step limit
        if self._state.step_count >= MAX_STEPS:
            self._done = True
            self._grader_result = grade_episode(self._task, self._action_history, False)
            return IncidentObservation(
                message=f"Step limit reached ({MAX_STEPS}). Episode ended.\n\nGrader feedback:\n{self._grader_result['feedback']}",
                data={"action_result": data, "grader": self._grader_result},
                available_actions=[],
                services=self._sim.service_names(),
                step_count=self._state.step_count,
                max_steps=MAX_STEPS,
                incident_resolved=False,
                task_id=self._task.task_id,
                done=True,
                reward=step_reward,
            )

        return IncidentObservation(
            message=message,
            data=data,
            available_actions=[e.value for e in ActionType],
            services=self._sim.service_names(),
            step_count=self._state.step_count,
            max_steps=MAX_STEPS,
            incident_resolved=False,
            task_id=self._task.task_id,
            done=False,
            reward=step_reward,
        )

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------

    def _compute_step_reward(
        self, action_type: str, target: str | None, params: dict | None, data: dict
    ) -> float:
        """
        Compute per-step reward providing continuous signal.

        Diagnostic actions on relevant services yield positive reward.
        Correct fix actions yield large positive reward.
        Irrelevant or harmful actions yield small negative reward.
        """
        task = self._task
        if task is None:
            return 0.0

        reward = 0.0
        diagnostic_actions = {"check_logs", "check_metrics", "check_service_status", "check_dependencies"}
        remediation_actions = {"restart_service", "scale_service", "rollback_deploy", "update_config"}

        if action_type in diagnostic_actions:
            if target and target in task.relevant_investigation_targets:
                reward += 0.03
                if target == task.root_cause_service:
                    reward += 0.05
            elif target is None and action_type == "check_service_status":
                reward += 0.02
            else:
                reward -= 0.01

        elif action_type in remediation_actions:
            is_primary = (
                action_type == task.fix_action_type
                and target == task.fix_target_service
            )
            is_alternate = any(
                action_type == af["action_type"]
                and target == af.get("target_service", task.fix_target_service)
                for af in task.alternate_fixes
            )

            if is_primary:
                if task.fix_parameters and _params_close(params, task.fix_parameters):
                    reward += 0.20
                elif task.fix_parameters:
                    reward += 0.10
                else:
                    reward += 0.20
            elif is_alternate:
                reward += 0.12
            elif target == task.fix_target_service:
                reward += 0.03
            else:
                reward -= 0.02

        elif action_type == "escalate":
            reward += 0.01

        elif action_type == "resolve":
            grader = grade_episode(task, self._action_history, True)
            reward += grader["score"] * 0.3

        if "error" in data and action_type not in {"resolve"}:
            reward -= 0.02

        return round(reward, 4)

    def _terminal_observation(self, msg: str) -> IncidentObservation:
        return IncidentObservation(
            message=msg,
            data=None,
            available_actions=[],
            services=[],
            step_count=self._state.step_count,
            max_steps=MAX_STEPS,
            incident_resolved=self._incident_resolved,
            task_id=self._task.task_id if self._task else "",
            done=True,
            reward=0.0,
        )

    # ------------------------------------------------------------------
    # Extra accessors for the API endpoints
    # ------------------------------------------------------------------

    def get_grader_result(self) -> dict[str, Any] | None:
        """Return the last grading result, or compute it if episode ended."""
        if self._grader_result:
            return self._grader_result
        if self._task and self._action_history:
            return grade_episode(self._task, self._action_history, self._incident_resolved)
        return None

    def get_action_history(self) -> list[dict[str, Any]]:
        return list(self._action_history)


def _params_close(actual: dict | None, expected: dict) -> bool:
    if not actual:
        return False
    for k, v in expected.items():
        if k not in actual:
            return False
        try:
            if isinstance(v, (int, float)):
                if abs(float(actual[k]) - float(v)) > float(v) * 0.2:
                    return False
        except (ValueError, TypeError):
            return False
    return True
