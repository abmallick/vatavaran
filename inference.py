"""
Vatavaran OpenEnv baseline inference script.

This script follows the required stdout contract:
- [START]
- [STEP]
- [END]
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import textwrap
from typing import Any, Optional

from openai import OpenAI

from vatavaran import VatavaranAction, VatavaranEnv

IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = os.getenv("RCA_BENCHMARK", "vatavaran")
MAX_STEPS = int(os.getenv("RCA_MAX_STEPS", "3"))
EPISODE_COUNT = int(os.getenv("RCA_EPISODE_COUNT", "3"))
RCA_BASE_URL = os.getenv("RCA_BASE_URL")
RCA_USE_BASE_URL = (os.getenv("RCA_USE_BASE_URL") or "false").lower() == "true"
DIFFICULTY_ORDER = ["easy", "middle", "hard"]


def _clean_action(action: str, limit: int = 140) -> str:
    action = action.replace("\n", " ").strip()
    if len(action) <= limit:
        return action
    return action[: limit - 3] + "..."


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={_clean_action(action)} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]):
    rewards_str = ",".join(f"{value:.2f}" for value in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def _parse_code_observation(text: str) -> dict[str, str]:
    # Sandbox code emits a dict-like string; this parser keeps fallback lightweight.
    result = {"component": "", "reason": "", "datetime": ""}
    component_match = re.search(r"'component':\s*'([^']+)'", text)
    reason_match = re.search(r"'reason':\s*'([^']+)'", text)
    dt_match = re.search(r"'datetime':\s*'([^']+)'", text)
    if component_match:
        result["component"] = component_match.group(1)
    if reason_match:
        result["reason"] = reason_match.group(1)
    if dt_match:
        result["datetime"] = dt_match.group(1)
    return result


def _deterministic_answer(task_description: str, extracted: dict[str, str]) -> str:
    wants_time = "datetime" in task_description or "occurrence datetime" in task_description
    wants_reason = "reason" in task_description
    wants_component = "component" in task_description

    answer_obj: dict[str, Any] = {"1": {}}
    if wants_time and extracted.get("datetime"):
        answer_obj["1"]["root cause occurrence datetime"] = extracted["datetime"]
    if wants_component and extracted.get("component"):
        answer_obj["1"]["root cause component"] = extracted["component"]
    if wants_reason and extracted.get("reason"):
        answer_obj["1"]["root cause reason"] = extracted["reason"]
    return json.dumps(answer_obj)


def _llm_answer(
    client: OpenAI,
    task_description: str,
    extracted: dict[str, str],
) -> str:
    prompt = textwrap.dedent(
        f"""
        You are formatting the final answer for a root cause analysis task.
        Task description:
        {task_description}

        Candidate extracted evidence:
        {json.dumps(extracted)}

        Return JSON only with top-level key "1".
        Include only requested fields from:
        - root cause occurrence datetime
        - root cause component
        - root cause reason
        """
    ).strip()
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Respond with valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=220,
        stream=False,
    )
    raw = (completion.choices[0].message.content or "").strip()
    if "```json" in raw:
        match = re.search(r"```json\s*(.*?)\s*```", raw, re.S)
        if match:
            raw = match.group(1).strip()
    # Validate parse to avoid malformed submit payload.
    json.loads(raw)
    return raw


def _score_from_submit_result(result_text: str, fallback: float) -> float:
    try:
        payload = json.loads(result_text)
        score = float(payload.get("score", fallback))
    except Exception:
        score = fallback
    return min(max(score, 0.0), 1.0)


async def _build_env() -> VatavaranEnv:
    if RCA_USE_BASE_URL and RCA_BASE_URL:
        return VatavaranEnv(base_url=RCA_BASE_URL)
    if IMAGE_NAME:
        return await VatavaranEnv.from_docker_image(IMAGE_NAME)
    # Local fallback for development runs.
    return VatavaranEnv(base_url="http://localhost:8000")


def _safe_reward(value: float | None) -> float:
    return min(max(float(value or 0.0), 0.0), 1.0)


async def _run_episode(client: OpenAI, episode_idx: int) -> tuple[bool, float]:
    env = await _build_env()
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    task_name = f"episode_{episode_idx}"
    try:
        difficulty = DIFFICULTY_ORDER[(episode_idx - 1) % len(DIFFICULTY_ORDER)]
        reset_result = await env.reset(difficulty=difficulty, seed=episode_idx)
        task_name = reset_result.observation.task_id or task_name
        task_description = reset_result.observation.task_description

        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        # Step 1: discover local files for this task.
        step_result = await env.step(VatavaranAction(action_type="list_files", content="."))
        rewards.append(_safe_reward(step_result.reward))
        steps_taken = 1
        log_step(
            step=1,
            action="list_files(.)",
            reward=rewards[-1],
            done=bool(step_result.done),
            error=step_result.observation.last_action_error,
        )
        if step_result.done or MAX_STEPS <= 1:
            score = _score_from_submit_result(
                step_result.observation.result, min(max(rewards[-1], 0.0), 1.0)
            )
            success = score >= 0.5
            return success, score

        # Step 2: run deterministic code to estimate root cause tuple.
        analysis_code = textwrap.dedent(
            """
            import pandas as pd
            from datetime import datetime

            df = pd.read_csv("metric/metric_container.csv")
            peak = df.loc[df["value"].idxmax()]
            reason_map = {
                "OSLinux-CPU_CPU_CPUCpuUtil": "high CPU usage",
                "OSLinux-MEM_MEM_MemUsedPercent": "high memory usage",
                "OSLinux-NET_NET_NicLatency": "network latency",
            }
            candidate = {
                "component": str(peak["cmdb_id"]),
                "reason": reason_map.get(str(peak["kpi_name"]), "high CPU usage"),
                "datetime": datetime.fromtimestamp(int(peak["timestamp"])).strftime("%Y-%m-%d %H:%M:%S"),
            }
            candidate
            """
        ).strip()

        step_result = await env.step(
            VatavaranAction(action_type="execute_code", content=analysis_code)
        )
        rewards.append(_safe_reward(step_result.reward))
        steps_taken = 2
        log_step(
            step=2,
            action="execute_code(metric_peak_detector)",
            reward=rewards[-1],
            done=bool(step_result.done),
            error=step_result.observation.last_action_error,
        )
        if step_result.done or MAX_STEPS <= 2:
            score = _score_from_submit_result(
                step_result.observation.result, min(max(rewards[-1], 0.0), 1.0)
            )
            success = score >= 0.5
            return success, score

        extracted = _parse_code_observation(step_result.observation.result)
        submit_payload = _deterministic_answer(task_description, extracted)
        if API_KEY:
            try:
                submit_payload = _llm_answer(client, task_description, extracted)
            except Exception:
                # Keep deterministic fallback on model errors.
                submit_payload = _deterministic_answer(task_description, extracted)

        # Step 3: submit diagnosis.
        step_result = await env.step(
            VatavaranAction(action_type="submit_answer", content=submit_payload)
        )
        rewards.append(_safe_reward(step_result.reward))
        steps_taken = 3
        log_step(
            step=3,
            action=f"submit_answer({submit_payload})",
            reward=rewards[-1],
            done=bool(step_result.done),
            error=step_result.observation.last_action_error,
        )

        score = _score_from_submit_result(
            step_result.observation.result,
            min(max(step_result.reward or 0.0, 0.0), 1.0),
        )
        success = score >= 0.5
        return success, score
    finally:
        try:
            await env.close()
        except Exception:
            pass
        log_end(
            success=success,
            steps=steps_taken,
            score=min(max(score, 0.0), 1.0),
            rewards=rewards,
        )


async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "missing-key")
    for i in range(1, EPISODE_COUNT + 1):
        await _run_episode(client, i)


if __name__ == "__main__":
    asyncio.run(main())
