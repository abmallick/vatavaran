"""Async EnvClient for the Vatavaran environment."""

from __future__ import annotations

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import VatavaranAction, VatavaranObservation, VatavaranState


class VatavaranEnv(EnvClient[VatavaranAction, VatavaranObservation, VatavaranState]):
    """Client wrapper for interacting with the Vatavaran environment server."""

    def _step_payload(self, action: VatavaranAction) -> dict:
        return {
            "action_type": action.action_type,
            "content": action.content,
        }

    def _parse_result(self, payload: dict) -> StepResult[VatavaranObservation]:
        observation = VatavaranObservation(**payload.get("observation", {}))
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict) -> VatavaranState:
        return VatavaranState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            current_task_id=payload.get("current_task_id"),
            current_difficulty=payload.get("current_difficulty"),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            modalities_explored=payload.get("modalities_explored", []),
            last_score=payload.get("last_score"),
        )


# Backward-compatible alias for prior OpenRCA naming.
OpenrcaEnvEnv = VatavaranEnv
