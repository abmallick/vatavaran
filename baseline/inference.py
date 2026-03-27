#!/usr/bin/env python3
"""
Baseline inference script for the Vatavaran Incident Response Environment.

Uses the OpenAI API to run a language model agent against all 3 tasks.
Reads API credentials from OPENAI_API_KEY environment variable.
Falls back to a deterministic rule-based agent if no API key is set.

Usage:
    export OPENAI_API_KEY=sk-...
    python baseline/inference.py --base-url http://localhost:8000

    # Or run with rule-based fallback (no API key needed):
    python baseline/inference.py --base-url http://localhost:8000 --rule-based
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import requests

TASK_IDS = ["service_outage", "db_pool_exhaustion", "cascading_failure"]
MAX_STEPS = 20


def get_tasks(base_url: str) -> list[dict]:
    resp = requests.get(f"{base_url}/tasks", timeout=30)
    resp.raise_for_status()
    return resp.json()["tasks"]


def env_reset(base_url: str, task_id: str) -> dict:
    resp = requests.post(f"{base_url}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(base_url: str, action: dict) -> dict:
    resp = requests.post(f"{base_url}/step", json=action, timeout=30)
    resp.raise_for_status()
    return resp.json()


def run_grader(base_url: str, task_id: str, action_history: list[dict]) -> dict:
    resp = requests.post(
        f"{base_url}/grader",
        json={
            "task_id": task_id,
            "action_history": action_history,
            "incident_resolved": True,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# LLM-based agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert SRE on-call engineer. You are responding to a production incident.

Your goal is to:
1. Diagnose the root cause of the incident by examining logs, metrics, and service status
2. Apply the correct remediation action
3. Resolve the incident

Available actions (respond with EXACTLY one JSON object per turn):
- {"action_type": "check_logs", "target_service": "<service-name>"}
- {"action_type": "check_metrics", "target_service": "<service-name>"}
- {"action_type": "check_service_status", "target_service": null}  (overview of all services)
- {"action_type": "check_service_status", "target_service": "<service-name>"}
- {"action_type": "check_dependencies", "target_service": "<service-name>"}
- {"action_type": "restart_service", "target_service": "<service-name>"}
- {"action_type": "scale_service", "target_service": "<service-name>", "parameters": {"replicas": <int>}}
- {"action_type": "rollback_deploy", "target_service": "<service-name>"}
- {"action_type": "update_config", "target_service": "<service-name>", "parameters": {"<key>": <value>}}
- {"action_type": "escalate"}
- {"action_type": "resolve"}

Strategy:
1. First check service status to see which services are affected
2. Check logs and metrics of affected services to understand the issue
3. Trace the root cause through the dependency chain
4. Apply the targeted fix (restart, rollback, config update)
5. Call resolve when the fix has been applied

IMPORTANT: Respond ONLY with a single valid JSON object. No explanation, no markdown."""


def run_llm_agent(base_url: str, task_id: str, model: str = "gpt-4o-mini") -> dict[str, Any]:
    """Run a single task using the OpenAI API."""
    from openai import OpenAI

    client = OpenAI()

    obs = env_reset(base_url, task_id)
    obs_data = obs.get("observation", obs)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"INCIDENT ALERT:\n{json.dumps(obs_data, indent=2)}\n\nBegin diagnosis. Respond with a JSON action."},
    ]

    action_history = []
    total_reward = 0.0

    for step_num in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=300,
        )

        content = response.choices[0].message.content.strip()
        try:
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            action = json.loads(content)
        except json.JSONDecodeError:
            action = {"action_type": "check_service_status", "target_service": None}

        action_history.append(action)
        result = env_step(base_url, action)
        obs_data = result.get("observation", result)
        reward = result.get("reward", 0.0)
        done = result.get("done", False)
        total_reward += reward

        messages.append({"role": "assistant", "content": json.dumps(action)})
        messages.append({"role": "user", "content": f"Observation:\n{json.dumps(obs_data, indent=2)}\n\nNext action?"})

        print(f"  Step {step_num + 1}: {action.get('action_type')} -> {action.get('target_service', 'N/A')} (reward: {reward:.4f})")

        if done:
            break

    grader = run_grader(base_url, task_id, action_history)
    return {
        "task_id": task_id,
        "steps": len(action_history),
        "cumulative_reward": round(total_reward, 4),
        "grader_score": grader.get("score", 0.0),
        "grader_breakdown": grader.get("breakdown", {}),
    }


# ---------------------------------------------------------------------------
# Rule-based agent (deterministic fallback)
# ---------------------------------------------------------------------------

RULE_STRATEGIES = {
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


def run_rule_agent_via_server(base_url: str) -> list[dict[str, Any]]:
    """Run the rule-based baseline by calling the /baseline endpoint."""
    resp = requests.post(f"{base_url}/baseline", timeout=60)
    resp.raise_for_status()
    raw = resp.json()["baseline_results"]
    results = []
    for r in raw:
        print(f"  {r['task_id']}: score={r['grader_score']:.4f}, steps={r['steps_taken']}")
        results.append({
            "task_id": r["task_id"],
            "steps": r["steps_taken"],
            "cumulative_reward": r["cumulative_reward"],
            "grader_score": r["grader_score"],
            "grader_breakdown": r.get("grader_breakdown", {}),
        })
    return results


def run_rule_agent_local() -> list[dict[str, Any]]:
    """Run the rule-based baseline locally (no server needed)."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from server.baseline_runner import run_baseline_all_tasks

    raw = run_baseline_all_tasks()
    results = []
    for r in raw:
        print(f"  {r['task_id']}: score={r['grader_score']:.4f}, steps={r['steps_taken']}")
        results.append({
            "task_id": r["task_id"],
            "steps": r["steps_taken"],
            "cumulative_reward": r["cumulative_reward"],
            "grader_score": r["grader_score"],
            "grader_breakdown": r.get("grader_breakdown", {}),
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Vatavaran Baseline Inference")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Environment server URL")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--rule-based", action="store_true", help="Use deterministic rule-based agent instead of LLM")
    parser.add_argument("--local", action="store_true", help="Run locally without a server (rule-based only)")
    args = parser.parse_args()

    use_llm = not args.rule_based and not args.local and os.environ.get("OPENAI_API_KEY")

    if args.local:
        print("Running rule-based baseline locally (no server)\n")
        all_results = run_rule_agent_local()
    elif use_llm:
        print(f"Running LLM baseline with model: {args.model}")
        print(f"Server: {args.base_url}\n")
        tasks = get_tasks(args.base_url)
        print(f"Found {len(tasks)} tasks:")
        for t in tasks:
            print(f"  - {t['task_id']}: {t['name']} ({t['difficulty']})")
        print()
        all_results = []
        for task_id in TASK_IDS:
            print(f"--- Task: {task_id} ---")
            result = run_llm_agent(args.base_url, task_id, model=args.model)
            all_results.append(result)
            print(f"  Score: {result['grader_score']:.4f}")
            print()
    else:
        print("Running rule-based baseline via server")
        print(f"Server: {args.base_url}\n")
        all_results = run_rule_agent_via_server(args.base_url)

    print("\n" + "=" * 60)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 60)
    for r in all_results:
        print(f"  {r['task_id']:25s}  score={r['grader_score']:.4f}  steps={r['steps']:2d}  reward={r['cumulative_reward']:.4f}")

    avg_score = sum(r["grader_score"] for r in all_results) / len(all_results)
    print(f"\n  Average grader score: {avg_score:.4f}")
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    main()
