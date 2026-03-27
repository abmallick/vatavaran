"""
FastAPI application for the Vatavaran Incident Response Environment.

Exposes the standard OpenEnv endpoints plus the hackathon-required
/baseline, /grader, and /tasks endpoints.
"""

from __future__ import annotations

import os
import traceback
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv-core is required. Install with: pip install openenv-core"
    ) from e

try:
    from ..models import IncidentAction, IncidentObservation
    from .environment import IncidentResponseEnvironment
except ImportError:
    from models import IncidentAction, IncidentObservation
    from server.environment import IncidentResponseEnvironment

from .tasks import ALL_TASKS

# Create the OpenEnv app (provides /reset, /step, /state, /ws, /health, /schema)
app = create_app(
    IncidentResponseEnvironment,
    IncidentAction,
    IncidentObservation,
    env_name="vatavaran-incident-response",
    max_concurrent_envs=4,
)


# ---------------------------------------------------------------------------
# Hackathon-required extra endpoints
# ---------------------------------------------------------------------------

@app.get("/tasks")
async def list_tasks() -> JSONResponse:
    """Return list of tasks and the action schema."""
    tasks = []
    for t in ALL_TASKS.values():
        tasks.append({
            "task_id": t.task_id,
            "name": t.name,
            "difficulty": t.difficulty,
            "description": t.description,
        })

    action_schema = IncidentAction.model_json_schema()

    return JSONResponse({
        "tasks": tasks,
        "action_schema": action_schema,
    })


@app.post("/grader")
async def run_grader(body: dict[str, Any] | None = None) -> JSONResponse:
    """
    Run the grader on a completed episode.

    Accepts a JSON body with:
      - task_id: str
      - action_history: list[dict]
      - incident_resolved: bool (optional, default True)

    Returns the grading result with score, breakdown, and feedback.
    """
    if body is None:
        raise HTTPException(status_code=400, detail="Request body required")

    task_id = body.get("task_id")
    if not task_id or task_id not in ALL_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_id. Available: {list(ALL_TASKS.keys())}",
        )

    action_history = body.get("action_history", [])
    if not action_history:
        raise HTTPException(status_code=400, detail="action_history is required and must be non-empty")

    resolved = body.get("incident_resolved", True)

    from .graders import grade_episode

    task = ALL_TASKS[task_id]
    result = grade_episode(task, action_history, resolved)
    return JSONResponse({"task_id": task_id, **result})


@app.post("/baseline")
async def run_baseline() -> JSONResponse:
    """
    Trigger the baseline inference script and return scores for all 3 tasks.

    This runs a deterministic rule-based baseline (no LLM required) to
    produce reproducible scores.
    """
    from .baseline_runner import run_baseline_all_tasks

    try:
        results = run_baseline_all_tasks()
        return JSONResponse({"baseline_results": results})
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Baseline failed: {str(e)}")


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
