"""
Grading logic for the Incident Response Environment.

Computes a 0.0–1.0 score based on:
  - Diagnostic progress (0.0–0.35): Did the agent investigate relevant services?
  - Root-cause identification (0.0–0.20): Did actions demonstrate understanding of the cause?
  - Remediation (0.0–0.35): Did the agent apply the correct fix?
  - Efficiency (0.0–0.10): How close to optimal step count?
"""

from __future__ import annotations

from typing import Any

from .tasks import TaskDefinition


def grade_episode(task: TaskDefinition, action_history: list[dict[str, Any]], incident_resolved: bool) -> dict[str, Any]:
    """
    Grade a completed episode.

    Returns a dict with:
      - score: float (0.0–1.0)
      - breakdown: dict of component scores
      - feedback: str explanation
    """
    diagnostic_score = _score_diagnostics(task, action_history)
    rootcause_score = _score_rootcause(task, action_history)
    remediation_score = _score_remediation(task, action_history, incident_resolved)
    efficiency_score = _score_efficiency(task, action_history)

    total = round(
        diagnostic_score + rootcause_score + remediation_score + efficiency_score, 4
    )
    total = min(1.0, max(0.0, total))

    breakdown = {
        "diagnostic": round(diagnostic_score, 4),
        "root_cause": round(rootcause_score, 4),
        "remediation": round(remediation_score, 4),
        "efficiency": round(efficiency_score, 4),
    }

    feedback = _build_feedback(task, breakdown, action_history, incident_resolved)

    return {"score": total, "breakdown": breakdown, "feedback": feedback}


def _score_diagnostics(task: TaskDefinition, history: list[dict[str, Any]]) -> float:
    """Up to 0.35 for investigating relevant services."""
    max_score = 0.35
    if not task.relevant_investigation_targets:
        return max_score

    diagnostic_actions = {"check_logs", "check_metrics", "check_service_status", "check_dependencies"}
    investigated = set()
    for action in history:
        if action.get("action_type") in diagnostic_actions:
            target = action.get("target_service")
            if target and target in task.relevant_investigation_targets:
                investigated.add(target)
            elif target is None and action.get("action_type") == "check_service_status":
                investigated.add("__overview__")

    coverage = len(investigated) / len(task.relevant_investigation_targets)
    score = max_score * min(1.0, coverage)

    # Bonus for checking root cause service directly
    root_checked = any(
        a.get("target_service") == task.root_cause_service
        and a.get("action_type") in diagnostic_actions
        for a in history
    )
    if root_checked:
        score = min(max_score, score + 0.05)

    return min(max_score, score)


def _score_rootcause(task: TaskDefinition, history: list[dict[str, Any]]) -> float:
    """Up to 0.20 for demonstrating root-cause understanding."""
    max_score = 0.20

    # Check if agent investigated the root cause service specifically
    diagnostic_actions = {"check_logs", "check_metrics", "check_dependencies"}
    root_investigated = False
    root_deep_investigated = False
    for action in history:
        if action.get("target_service") == task.root_cause_service:
            if action.get("action_type") in diagnostic_actions:
                root_investigated = True
            if action.get("action_type") == "check_logs":
                root_deep_investigated = True

    if not root_investigated:
        return 0.0

    score = 0.10
    if root_deep_investigated:
        score = 0.15

    # Full score if the fix targets the correct service
    fix_targets_root = any(
        a.get("target_service") == task.fix_target_service
        and a.get("action_type") in {task.fix_action_type}
        | {af["action_type"] for af in task.alternate_fixes}
        for a in history
    )
    if fix_targets_root:
        score = max_score

    return min(max_score, score)


def _score_remediation(task: TaskDefinition, history: list[dict[str, Any]], resolved: bool) -> float:
    """Up to 0.35 for applying the correct fix."""
    max_score = 0.35

    # Check primary fix
    primary_fix_applied = any(
        a.get("action_type") == task.fix_action_type
        and a.get("target_service") == task.fix_target_service
        for a in history
    )

    if primary_fix_applied:
        # Check parameters if required
        if task.fix_parameters:
            correct_params = any(
                a.get("action_type") == task.fix_action_type
                and a.get("target_service") == task.fix_target_service
                and _params_match(a.get("parameters"), task.fix_parameters)
                for a in history
            )
            return max_score if correct_params else max_score * 0.7
        return max_score

    # Check alternate fixes
    alt_fix_applied = any(
        a.get("action_type") == af["action_type"]
        and a.get("target_service") == af.get("target_service", task.fix_target_service)
        for a in history
        for af in task.alternate_fixes
    )
    if alt_fix_applied:
        return max_score * 0.8

    # Partial credit for targeting right service with a remediation action
    remediation_actions = {"restart_service", "scale_service", "rollback_deploy", "update_config"}
    right_service = any(
        a.get("target_service") == task.fix_target_service
        and a.get("action_type") in remediation_actions
        for a in history
    )
    if right_service:
        return max_score * 0.4

    return 0.0


def _score_efficiency(task: TaskDefinition, history: list[dict[str, Any]]) -> float:
    """Up to 0.10 for being close to optimal step count."""
    max_score = 0.10
    n_steps = len(history)
    optimal = task.optimal_steps

    if n_steps == 0:
        return 0.0
    if n_steps <= optimal:
        return max_score
    if n_steps <= optimal * 2:
        ratio = 1.0 - (n_steps - optimal) / optimal
        return max_score * max(0.0, ratio)
    return 0.0


def _params_match(actual: dict | None, expected: dict) -> bool:
    """Check if the actual parameters satisfy the expected fix parameters."""
    if not actual:
        return False
    for key, val in expected.items():
        if key not in actual:
            return False
        try:
            if isinstance(val, (int, float)):
                if abs(float(actual[key]) - float(val)) > float(val) * 0.2:
                    return False
            elif str(actual[key]) != str(val):
                return False
        except (ValueError, TypeError):
            return False
    return True


def _build_feedback(
    task: TaskDefinition,
    breakdown: dict[str, float],
    history: list[dict[str, Any]],
    resolved: bool,
) -> str:
    lines = [f"Task: {task.name} ({task.difficulty})"]
    lines.append(f"Total score: {sum(breakdown.values()):.2f}/1.00")
    lines.append(f"Steps taken: {len(history)} (optimal: {task.optimal_steps})")
    lines.append("")
    lines.append("Component scores:")
    lines.append(f"  Diagnostic investigation: {breakdown['diagnostic']:.2f}/0.35")
    lines.append(f"  Root-cause identification: {breakdown['root_cause']:.2f}/0.20")
    lines.append(f"  Remediation:               {breakdown['remediation']:.2f}/0.35")
    lines.append(f"  Efficiency:                {breakdown['efficiency']:.2f}/0.10")

    if breakdown["diagnostic"] < 0.15:
        lines.append("\nTip: Investigate more services before attempting a fix.")
    if breakdown["root_cause"] < 0.10:
        lines.append(f"\nTip: The root cause was in {task.root_cause_service}. Check its logs and metrics.")
    if breakdown["remediation"] < 0.15:
        lines.append(f"\nTip: The correct fix was '{task.fix_action_type}' on {task.fix_target_service}.")

    return "\n".join(lines)
