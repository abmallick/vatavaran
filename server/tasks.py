"""
Task definitions for the Incident Response Environment.

Each task represents a production incident with defined fault injection,
optimal resolution paths, and grading criteria.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskDefinition:
    task_id: str
    name: str
    difficulty: str  # easy, medium, hard
    description: str
    alert_title: str
    alert_severity: str
    alert_description: str
    affected_services: list[str]
    root_cause_service: str
    root_cause_description: str

    # Optimal resolution path (ordered list of action types + targets)
    optimal_actions: list[dict[str, Any]] = field(default_factory=list)
    optimal_steps: int = 2

    # Services that are relevant to investigate
    relevant_investigation_targets: list[str] = field(default_factory=list)
    # The key fix action
    fix_action_type: str = ""
    fix_target_service: str = ""
    fix_parameters: dict[str, Any] | None = None
    # Alternate acceptable fix actions
    alternate_fixes: list[dict[str, Any]] = field(default_factory=list)


TASK_EASY = TaskDefinition(
    task_id="service_outage",
    name="Service Outage Recovery",
    difficulty="easy",
    description=(
        "The api-gateway is completely down, returning no responses. All user traffic "
        "is affected. Diagnose why the gateway is down and bring it back online."
    ),
    alert_title="CRITICAL: api-gateway unreachable",
    alert_severity="critical",
    alert_description=(
        "All api-gateway replicas stopped responding at 02:10 UTC. Health checks failing. "
        "HTTP 503 returned for 100% of requests. Immediate action required."
    ),
    affected_services=["api-gateway"],
    root_cause_service="api-gateway",
    root_cause_description="api-gateway crashed due to OutOfMemoryError (OOM killed)",
    optimal_actions=[
        {"action_type": "check_logs", "target_service": "api-gateway"},
        {"action_type": "restart_service", "target_service": "api-gateway"},
    ],
    optimal_steps=2,
    relevant_investigation_targets=["api-gateway"],
    fix_action_type="restart_service",
    fix_target_service="api-gateway",
    alternate_fixes=[
        {"action_type": "scale_service", "target_service": "api-gateway"},
    ],
)


TASK_MEDIUM = TaskDefinition(
    task_id="db_pool_exhaustion",
    name="Database Connection Pool Exhaustion",
    difficulty="medium",
    description=(
        "Multiple services are experiencing high latency and timeouts. Users report "
        "slow page loads and occasional errors. The issue started roughly an hour ago."
    ),
    alert_title="HIGH LATENCY: user-service, order-service SLO breach",
    alert_severity="warning",
    alert_description=(
        "user-service and order-service p99 latency exceeds 8000ms (SLO: 500ms). "
        "Error rate elevated at 15%. api-gateway reporting upstream timeouts. "
        "Issue onset: ~60 minutes ago."
    ),
    affected_services=["user-service", "order-service", "api-gateway"],
    root_cause_service="database",
    root_cause_description="Database connection pool reduced from 100 to 10 by automated config push",
    optimal_actions=[
        {"action_type": "check_metrics", "target_service": "user-service"},
        {"action_type": "check_metrics", "target_service": "database"},
        {"action_type": "check_dependencies", "target_service": "database"},
        {"action_type": "update_config", "target_service": "database", "parameters": {"pool_size": 100}},
    ],
    optimal_steps=4,
    relevant_investigation_targets=["user-service", "order-service", "database", "api-gateway"],
    fix_action_type="update_config",
    fix_target_service="database",
    fix_parameters={"pool_size": 100},
    alternate_fixes=[
        {"action_type": "rollback_deploy", "target_service": "database"},
        {"action_type": "update_config", "target_service": "database"},
    ],
)


TASK_HARD = TaskDefinition(
    task_id="cascading_failure",
    name="Cascading Failure Investigation",
    difficulty="hard",
    description=(
        "Multiple alerts firing across the stack. The api-gateway reports 28% error rate, "
        "auth-service is failing token validation, and the system is in a degraded state "
        "affecting all users. Auto-scaling has not resolved the issue."
    ),
    alert_title="CRITICAL: Cascading failures across multiple services",
    alert_severity="critical",
    alert_description=(
        "Correlated alerts: api-gateway 28% error rate, auth-service 35% error rate, "
        "cache-service unhealthy (97% memory). Auto-scaling api-gateway from 4→8 replicas "
        "did not resolve. Multiple SLO breaches. Issue has been escalating for ~2 hours."
    ),
    affected_services=["api-gateway", "auth-service", "cache-service", "database"],
    root_cause_service="cache-service",
    root_cause_description=(
        "Deploy v1.6.0 to cache-service introduced a memory leak in the session partition module. "
        "As cache degrades, auth-service falls back to DB for token validation, overwhelming the DB "
        "and causing auth failures that cascade to api-gateway."
    ),
    optimal_actions=[
        {"action_type": "check_service_status", "target_service": None},
        {"action_type": "check_logs", "target_service": "api-gateway"},
        {"action_type": "check_logs", "target_service": "auth-service"},
        {"action_type": "check_logs", "target_service": "cache-service"},
        {"action_type": "rollback_deploy", "target_service": "cache-service"},
    ],
    optimal_steps=5,
    relevant_investigation_targets=[
        "api-gateway", "auth-service", "cache-service", "database",
    ],
    fix_action_type="rollback_deploy",
    fix_target_service="cache-service",
    alternate_fixes=[
        {"action_type": "restart_service", "target_service": "cache-service"},
    ],
)


ALL_TASKS: dict[str, TaskDefinition] = {
    TASK_EASY.task_id: TASK_EASY,
    TASK_MEDIUM.task_id: TASK_MEDIUM,
    TASK_HARD.task_id: TASK_HARD,
}
