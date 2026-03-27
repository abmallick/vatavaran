---
title: Vatavaran — Incident Response Environment
emoji: 🔥
colorFrom: red
colorTo: orange
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Vatavaran: Incident Response Agent Environment

**Vatavaran** (Hindi: वातावरण, "environment") is an OpenEnv-compatible environment that simulates production incident response in a microservice architecture. AI agents act as on-call SRE engineers, diagnosing root causes and applying remediations across a realistic infrastructure of 8 interconnected services.

## Why Incident Response?

Incident response is one of the most expensive and high-stakes tasks in modern software operations:

- **$100B+ annual cost** across the tech industry in engineering time spent on incidents
- **Every tech company** from startups to FAANG runs 24/7 on-call rotations
- **High cognitive load**: engineers must reason across distributed systems under time pressure
- **Directly trainable**: clear success criteria, rich partial-progress signals, natural difficulty progression

No existing OpenEnv environment covers this domain. Vatavaran fills that gap, providing a rigorous benchmark for evaluating how well AI agents can perform real-world operational reasoning.

## Tasks

| Task ID | Name | Difficulty | Description |
|---------|------|-----------|-------------|
| `service_outage` | Service Outage Recovery | Easy | The api-gateway is down (OOM killed). Diagnose from logs and restart. |
| `db_pool_exhaustion` | DB Connection Pool Exhaustion | Medium | Multiple services slow due to a bad config push reducing DB pool from 100→10. Trace through metrics and fix config. |
| `cascading_failure` | Cascading Failure Investigation | Hard | Memory leak in cache-service triggers auth failures cascading to api-gateway. Trace the dependency chain and rollback the bad deploy. |

### Difficulty Progression

- **Easy**: Single service, single action fix. Tests basic log reading.
- **Medium**: Multi-service impact, red herrings (unrelated deploys). Tests metric correlation and config management.
- **Hard**: 4-service cascade with the root cause 3 hops away from the symptoms. Tests systematic dependency tracing under ambiguity.

## Infrastructure

The simulated architecture includes 8 services with realistic metrics, logs, configs, and deployment history:

```
┌──────────────┐
│  api-gateway  │──→ auth-service ──→ cache-service
│  (entry)      │──→ user-service ──→ database
│               │──→ order-service ──→ payment-service
└──────────────┘                   ──→ message-queue
```

Each service exposes: status, CPU/memory usage, latency percentiles, error rates, request rates, replica count, configuration, deployment history, and timestamped logs.

## Action Space

```python
class IncidentAction(Action):
    action_type: ActionType   # see below
    target_service: str | None
    parameters: dict | None
```

| Action | Description | Requires `target_service` |
|--------|-------------|:------------------------:|
| `check_logs` | View recent log entries for a service | Yes |
| `check_metrics` | View CPU, memory, latency, error rate | Yes |
| `check_service_status` | Get status of one or all services | Optional |
| `check_dependencies` | View dependency graph, config, deploys | Optional |
| `restart_service` | Restart a service | Yes |
| `scale_service` | Change replica count | Yes |
| `rollback_deploy` | Rollback the most recent deployment | Yes |
| `update_config` | Update service configuration parameters | Yes |
| `escalate` | Escalate to senior on-call | No |
| `resolve` | Mark the incident as resolved | No |

## Observation Space

```python
class IncidentObservation(Observation):
    alert: AlertInfo | None      # Incident alert details (shown on reset)
    message: str                 # Feedback about the last action
    data: dict | None            # Structured response data
    available_actions: list[str] # Available action types
    services: list[str]          # All service names
    step_count: int              # Current step
    max_steps: int               # Step limit (25)
    incident_resolved: bool      # Whether resolved
    task_id: str                 # Current task
```

## Reward Design

Rewards provide **continuous signal** throughout the episode (not just sparse end-of-episode):

| Component | Max | Description |
|-----------|-----|-------------|
| Diagnostic investigation | 0.35 | Investigating relevant services (bonus for root-cause service) |
| Root-cause identification | 0.20 | Demonstrating understanding via targeted investigation |
| Remediation | 0.35 | Applying the correct fix action with correct parameters |
| Efficiency | 0.10 | Fewer steps = higher score (optimal vs actual) |

**Per-step signals**: +0.03–0.08 for useful diagnostic actions, +0.12–0.20 for correct remediations, −0.01–0.02 for irrelevant actions. This enables gradient-based RL training, not just sparse binary rewards.

## Baseline Scores

Deterministic rule-based baseline (no LLM):

| Task | Score | Steps |
|------|-------|-------|
| service_outage (easy) | ~0.85 | 4 |
| db_pool_exhaustion (medium) | ~0.90 | 7 |
| cascading_failure (hard) | ~0.90 | 9 |

These scores are reproducible via the `/baseline` endpoint or `python baseline/inference.py --rule-based`.

## Setup & Usage

### Install

```bash
pip install openenv-core[core]
# or
uv sync
```

### Run Server Locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
docker build -t vatavaran:latest -f server/Dockerfile .
docker run -p 8000:8000 vatavaran:latest
```

### Run Baseline

```bash
# Rule-based (deterministic, no API key needed):
python baseline/inference.py --base-url http://localhost:8000 --rule-based

# LLM-based (requires OPENAI_API_KEY):
export OPENAI_API_KEY=sk-...
python baseline/inference.py --base-url http://localhost:8000
```

### Use as Client

```python
from vatavaran import IncidentAction, IncidentResponseEnv

async with IncidentResponseEnv(base_url="http://localhost:8000") as env:
    result = await env.reset()
    print(result.observation.alert.title)

    result = await env.step(IncidentAction(
        action_type="check_logs",
        target_service="api-gateway",
    ))
    print(result.observation.data)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment (body: `{"task_id": "..."}`) |
| `/step` | POST | Take an action |
| `/state` | GET | Current episode state |
| `/health` | GET | Health check |
| `/tasks` | GET | List tasks and action schema |
| `/grader` | POST | Grade a completed episode |
| `/baseline` | POST | Run baseline and return scores |
| `/schema` | GET | Action/Observation JSON schemas |
| `/ws` | WS | WebSocket for persistent sessions |
| `/web` | GET | Interactive web UI |

## Project Structure

```
vatavaran/
├── openenv.yaml              # OpenEnv manifest
├── pyproject.toml             # Package metadata
├── README.md                  # This file
├── __init__.py                # Package exports
├── models.py                  # Action/Observation Pydantic models
├── client.py                  # EnvClient implementation
├── server/
│   ├── app.py                 # FastAPI application + extra endpoints
│   ├── environment.py         # Core IncidentResponseEnvironment
│   ├── simulation.py          # Infrastructure simulation engine
│   ├── tasks.py               # Task definitions (easy/medium/hard)
│   ├── graders.py             # Grading logic (0.0–1.0)
│   ├── baseline_runner.py     # Deterministic baseline for /baseline
│   ├── Dockerfile             # Container image
│   └── requirements.txt       # Server dependencies
└── baseline/
    └── inference.py           # Standalone inference script (LLM + rule-based)
```

## License

BSD-3-Clause
