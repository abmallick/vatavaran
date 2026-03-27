"""Vatavaran Incident Response Environment Client."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import IncidentAction, IncidentObservation


class IncidentResponseEnv(
    EnvClient[IncidentAction, IncidentObservation, State]
):
    """
    Client for the Vatavaran Incident Response Environment.

    Example:
        >>> async with IncidentResponseEnv(base_url="http://localhost:8000") as client:
        ...     result = await client.reset()
        ...     result = await client.step(IncidentAction(
        ...         action_type="check_logs",
        ...         target_service="api-gateway",
        ...     ))
    """

    def _step_payload(self, action: IncidentAction) -> Dict:
        payload: Dict[str, Any] = {"action_type": action.action_type.value}
        if action.target_service:
            payload["target_service"] = action.target_service
        if action.parameters:
            payload["parameters"] = action.parameters
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[IncidentObservation]:
        obs_data = payload.get("observation", {})
        observation = IncidentObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
