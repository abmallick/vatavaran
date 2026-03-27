"""
Pydantic models for the Vatavaran Incident Response Environment.

Defines typed Action, Observation schemas for SRE incident triage.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class ActionType(str, Enum):
    CHECK_LOGS = "check_logs"
    CHECK_METRICS = "check_metrics"
    CHECK_SERVICE_STATUS = "check_service_status"
    CHECK_DEPENDENCIES = "check_dependencies"
    RESTART_SERVICE = "restart_service"
    SCALE_SERVICE = "scale_service"
    ROLLBACK_DEPLOY = "rollback_deploy"
    UPDATE_CONFIG = "update_config"
    ESCALATE = "escalate"
    RESOLVE = "resolve"


class IncidentAction(Action):
    """An SRE on-call action taken to diagnose or remediate a production incident."""

    action_type: ActionType = Field(
        ...,
        description=(
            "Type of action: check_logs, check_metrics, check_service_status, "
            "check_dependencies, restart_service, scale_service, rollback_deploy, "
            "update_config, escalate, resolve"
        ),
    )
    target_service: Optional[str] = Field(
        default=None,
        description="Service to act on (e.g. 'api-gateway', 'database'). Required for most actions.",
    )
    parameters: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional parameters. E.g. {'pool_size': 100} for update_config, {'replicas': 3} for scale_service.",
    )


class AlertInfo(Observation):
    """Details of the initial incident alert."""

    alert_id: str = Field(default="", description="Unique alert identifier")
    severity: str = Field(default="", description="Alert severity: critical, warning, info")
    title: str = Field(default="", description="Short alert title")
    description: str = Field(default="", description="Detailed alert description")
    triggered_at: str = Field(default="", description="When the alert fired")
    affected_services: list[str] = Field(default_factory=list, description="Services mentioned in the alert")


class IncidentObservation(Observation):
    """What the on-call agent observes after taking an action."""

    alert: Optional[AlertInfo] = Field(default=None, description="Initial incident alert (shown on reset)")
    message: str = Field(default="", description="System feedback about the last action")
    data: Optional[dict[str, Any]] = Field(default=None, description="Structured response data from the action")
    available_actions: list[str] = Field(
        default_factory=lambda: [e.value for e in ActionType],
        description="Action types the agent can take",
    )
    services: list[str] = Field(default_factory=list, description="All services in the infrastructure")
    step_count: int = Field(default=0, description="Current step number")
    max_steps: int = Field(default=25, description="Maximum steps allowed per episode")
    incident_resolved: bool = Field(default=False, description="Whether the incident has been resolved")
    task_id: str = Field(default="", description="Current task identifier")
