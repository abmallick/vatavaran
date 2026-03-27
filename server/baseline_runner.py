"""
Deterministic rule-based baseline that runs locally against the environment.
Used by the /baseline endpoint to produce reproducible scores.
"""

from __future__ import annotations

from typing import Any

from .environment import IncidentResponseEnvironment
from .tasks import ALL_TASKS

try:
    from ..models import IncidentAction
except ImportError:
    from models import IncidentAction


# Rule-based action sequences for each task (deterministic, no LLM needed)
BASELINE_STRATEGIES: dict[str, list[dict[str, Any]]] = {
    "service_outage": [
        {"action_type": "check_service_status", "target_service": None},
        {"action_type": "check_logs", "target_service": "api-gateway"},
        {"action_type": "restart_service", "target_service": "api-gateway"},
        {"action_type": "resolve"},
    ],
    "db_pool_exhaustion": [
        {"action_type": "check_service_status", "target_service": None},
        {"action_type": "check_metrics", "target_service": "user-service"},
        {"action_type": "check_metrics", "target_service": "database"},
        {"action_type": "check_logs", "target_service": "database"},
        {"action_type": "check_dependencies", "target_service": "database"},
        {"action_type": "update_config", "target_service": "database", "parameters": {"pool_size": 100}},
        {"action_type": "resolve"},
    ],
    "cascading_failure": [
        {"action_type": "check_service_status", "target_service": None},
        {"action_type": "check_logs", "target_service": "api-gateway"},
        {"action_type": "check_metrics", "target_service": "auth-service"},
        {"action_type": "check_logs", "target_service": "auth-service"},
        {"action_type": "check_logs", "target_service": "cache-service"},
        {"action_type": "check_metrics", "target_service": "cache-service"},
        {"action_type": "check_dependencies", "target_service": "cache-service"},
        {"action_type": "rollback_deploy", "target_service": "cache-service"},
        {"action_type": "resolve"},
    ],
}


def run_baseline_single_task(task_id: str) -> dict[str, Any]:
    """Run the baseline strategy for a single task and return the result."""
    env = IncidentResponseEnvironment()
    obs = env.reset(task_id=task_id)

    strategy = BASELINE_STRATEGIES[task_id]
    trajectory: list[dict[str, Any]] = []
    total_reward = 0.0

    for action_spec in strategy:
        action = IncidentAction(
            action_type=action_spec["action_type"],
            target_service=action_spec.get("target_service"),
            parameters=action_spec.get("parameters"),
        )
        obs = env.step(action)
        total_reward += obs.reward
        trajectory.append({
            "action": action_spec,
            "reward": obs.reward,
            "done": obs.done,
        })
        if obs.done:
            break

    grader_result = env.get_grader_result()
    return {
        "task_id": task_id,
        "task_name": ALL_TASKS[task_id].name,
        "difficulty": ALL_TASKS[task_id].difficulty,
        "steps_taken": len(trajectory),
        "cumulative_reward": round(total_reward, 4),
        "grader_score": grader_result["score"] if grader_result else 0.0,
        "grader_breakdown": grader_result.get("breakdown") if grader_result else {},
        "grader_feedback": grader_result.get("feedback", "") if grader_result else "",
    }


def run_baseline_all_tasks() -> list[dict[str, Any]]:
    """Run baseline for all tasks and return results."""
    results = []
    for task_id in ALL_TASKS:
        result = run_baseline_single_task(task_id)
        results.append(result)
    return results
