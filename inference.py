"""
Vatavaran OpenEnv inference script (task-scoped LLM agent loop).

MANDATORY / environment variables
- API_BASE_URL     LLM API base URL (default: OpenAI https://api.openai.com/v1).
- MODEL_NAME       Model id for inference (default: gpt-4o-mini).
- OPENAI_API_KEY or HF_TOKEN or API_KEY  API key for the LLM.

Optional
- IMAGE_NAME / LOCAL_IMAGE_NAME  Docker image for from_docker_image().
- RCA_BASE_URL + RCA_USE_BASE_URL=true  Remote env server URL.
- RCA_BENCHMARK    Label for [START] line (default: vatavaran).
- RCA_MAX_STEPS    Upper bound on steps per episode (default: 32), capped by env max_steps.
- RCA_SEED         Optional int passed to reset(seed=...).
- RCA_TASK_IDS     Comma-separated task ids (default: Bank_00003,Bank_00006,Bank_00009).
- RCA_MESSAGE_TIMEOUT_S  Max seconds to wait for each env WebSocket reply (default: 300).
- RCA_WS_PING_INTERVAL   WebSocket keepalive ping interval in seconds (default: 60). Set to
                         "none" to disable client pings (can avoid keepalive timeouts on very
                         long server-side work; less ideal through proxies).
- RCA_WS_PING_TIMEOUT    Seconds to wait for pong after a ping (default: 120).

STDOUT: [START], one [STEP] per env.step(), then [END] (always, including on error).
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

IMAGE_NAME = os.getenv("IMAGE_NAME", "vatavaran") or os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = os.getenv("RCA_BENCHMARK", "vatavaran")
DEFAULT_TASK_IDS = ("Bank_00003", "Bank_00006", "Bank_00009")


def _parse_task_ids() -> list[str]:
    raw = os.getenv("RCA_TASK_IDS")
    if raw is not None and raw.strip():
        return [part.strip() for part in raw.split(",") if part.strip()]
    return list(DEFAULT_TASK_IDS)


RCA_BASE_URL = os.getenv("RCA_BASE_URL")
RCA_USE_BASE_URL = (os.getenv("RCA_USE_BASE_URL") or "false").lower() == "true"
RCA_MAX_STEPS = int(os.getenv("RCA_MAX_STEPS", "32"))
RCA_SEED = os.getenv("RCA_SEED")


def _parse_message_timeout_s() -> float:
    return float(os.getenv("RCA_MESSAGE_TIMEOUT_S", "300"))


def _parse_ws_ping_interval() -> float | None:
    env = os.getenv("RCA_WS_PING_INTERVAL")
    if env is None:
        return 60.0
    raw = env.strip().lower()
    if raw in ("none", "off", "disable"):
        return None
    return float(raw)


def _parse_ws_ping_timeout() -> float | None:
    env = os.getenv("RCA_WS_PING_TIMEOUT")
    if env is None:
        return 120.0
    raw = env.strip().lower()
    if raw == "none":
        return None
    return float(raw)

TEMPERATURE = float(os.getenv("RCA_TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("RCA_MAX_TOKENS", "2048"))
SUCCESS_SCORE_THRESHOLD = 0.5

_JSON_PARSE_RETRIES = 3

ALLOWED_ACTIONS = frozenset({"list_files", "execute_code", "submit_answer"})

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an agent solving a root cause analysis (RCA) task in a simulated environment.

    PATH RULES (critical): The sandbox process starts **inside** the incident-day telemetry directory
    (it already contains `metric/`, `trace/`, `log/`). For `list_files`, `content` must be a path
    **relative to that directory** — e.g. `"."`, `"metric"`, `"trace"`, `"log"`, or
    `"metric/metric_app.csv"`. Wrong: any path starting with `data/`, or repeating
    `.../telemetry/<date>/...` — that double-joins paths and fails. For `execute_code`, open and
    read files with the same relative paths (cwd is already the telemetry day folder).

    Each turn you must choose exactly one action by replying with a single JSON object only
    (no markdown fences, no extra text), with this shape:
    {"action_type":"<type>","content":"<string>"}

    action_type must be one of:
    - list_files — content is the path to list (e.g. "." or "metric").
    - execute_code — content is Python source to run in the task sandbox.
    - submit_answer — content is a JSON string for the final diagnosis (format required by the task).

    Use the conversation history: previous assistant messages are your past actions; user messages
    labeled "Environment result" are the outcomes of those actions (like tool returns).
    """
).strip()


def _clean_action(action: str, limit: int = 140) -> str:
    return action
    # action = action.replace("\n", " ").strip()
    # if len(action) <= limit:
    #     return action
    # return action[: limit - 3] + "..."


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={_clean_action(action)} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{value:.2f}" for value in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def _extract_json_object(text: str) -> str:
    raw = (text or "").strip()
    if "```" in raw:
        fence = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.S)
        if fence:
            raw = fence.group(1).strip()
    return raw


def _parse_action_json(text: str) -> VatavaranAction:
    raw = _extract_json_object(text)
    data: Any = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    action_type = data.get("action_type")
    content = data.get("content", "")
    if action_type not in ALLOWED_ACTIONS:
        raise ValueError(f"Invalid action_type: {action_type!r}")
    if not isinstance(content, str):
        content = json.dumps(content) if content is not None else ""
    return VatavaranAction(action_type=action_type, content=content)


def _score_from_submit_result(result_text: str, fallback: float) -> float:
    try:
        payload = json.loads(result_text)
        score = float(payload.get("score", fallback))
    except Exception:
        score = fallback
    return min(max(score, 0.0), 1.0)


def _env_ws_kwargs() -> dict[str, Any]:
    return {
        "message_timeout_s": _parse_message_timeout_s(),
        "ws_ping_interval": _parse_ws_ping_interval(),
        "ws_ping_timeout": _parse_ws_ping_timeout(),
    }


async def _build_env() -> VatavaranEnv:
    ws_kw = _env_ws_kwargs()
    if RCA_USE_BASE_URL and RCA_BASE_URL:
        return VatavaranEnv(base_url=RCA_BASE_URL, **ws_kw)
    if IMAGE_NAME:
        return await VatavaranEnv.from_docker_image(IMAGE_NAME, **ws_kw)
    return VatavaranEnv(base_url="http://localhost:8000", **ws_kw)


def _safe_reward(value: float | None) -> float:
    return min(max(float(value or 0.0), 0.0), 1.0)


def _initial_user_message(
    task_id: str,
    task_description: str,
    domain_knowledge: str,
    max_steps: int,
) -> str:
    return textwrap.dedent(
        f"""
        Task id: {task_id}

        Task description:
        {task_description}

        Path rules: Your first message after reset includes the sandbox working directory. Use only
        relative paths for list_files and file I/O in execute_code (".", "metric", "trace", "log").
        Never use repo-style paths like data/.../telemetry/...

        Domain knowledge (reference):
        {domain_knowledge}

        Max steps for this episode: {max_steps}

        Respond with your first action as JSON only: {{"action_type":"...","content":"..."}}
        """
    ).strip()


def _env_result_user_message(obs: Any, reward: float, done: bool) -> str:
    err = obs.last_action_error if obs.last_action_error else "null"
    body = textwrap.dedent(
        f"""
        Environment result:
        success: {str(obs.success).lower()}
        reward: {reward:.4f}
        done: {str(done).lower()}
        last_action_error: {err}

        Result (main output):
        {obs.result}
        """
    ).strip()
    return body


def get_model_action(client: OpenAI, messages: list[dict[str, str]]) -> str:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    return (completion.choices[0].message.content or "").strip()


async def _run_episode(client: OpenAI, env: VatavaranEnv, task_id: str) -> None:
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    seed: int | None = None
    if RCA_SEED is not None and str(RCA_SEED).strip() != "":
        try:
            seed = int(RCA_SEED)
        except ValueError:
            seed = None

    reset_kwargs: dict[str, Any] = {"task_id": task_id}
    if seed is not None:
        reset_kwargs["seed"] = seed

    try:
        reset_result = await env.reset(**reset_kwargs)
        obs = reset_result.observation
        task_name = obs.task_id or task_id
        env_max = max(1, int(obs.max_steps or RCA_MAX_STEPS))
        step_limit = min(env_max, RCA_MAX_STEPS)

        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _initial_user_message(
                    task_name,
                    obs.task_description,
                    obs.domain_knowledge,
                    step_limit,
                ),
            },
        ]

        if reset_result.done:
            score = _score_from_submit_result(obs.result, 0.0)
            success = score >= SUCCESS_SCORE_THRESHOLD
            return

        for step in range(1, step_limit + 1):
            parse_failures = 0
            action: VatavaranAction | None = None
            assistant_text = ""

            while action is None and parse_failures < _JSON_PARSE_RETRIES:
                try:
                    assistant_text = get_model_action(client, messages)
                    action = _parse_action_json(assistant_text)
                except Exception as exc:
                    parse_failures += 1
                    messages.append({"role": "assistant", "content": assistant_text or "(empty)"})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Invalid or missing JSON action. Reply with a single JSON object only: "
                                '{"action_type":"list_files"|"execute_code"|"submit_answer","content":"..."} '
                                f"Parse error: {exc}"
                            ),
                        }
                    )

            if action is None:
                action = VatavaranAction(action_type="list_files", content=".")

            messages.append({"role": "assistant", "content": assistant_text or json.dumps(action.model_dump())})

            step_result = await env.step(action)
            obs = step_result.observation
            reward = _safe_reward(step_result.reward)
            done = bool(step_result.done)
            err = obs.last_action_error

            rewards.append(reward)
            steps_taken = step

            action_str = json.dumps({"action_type": action.action_type, "content": action.content})
            log_step(step=step, action=action_str, reward=reward, done=done, error=err)

            messages.append(
                {"role": "user", "content": _env_result_user_message(obs, reward, done)}
            )

            if done:
                score = _score_from_submit_result(
                    obs.result,
                    min(max(reward, 0.0), 1.0),
                )
                success = score >= SUCCESS_SCORE_THRESHOLD
                break

        if steps_taken > 0 and not success:
            score = _score_from_submit_result(
                obs.result,
                min(max(rewards[-1], 0.0), 1.0) if rewards else 0.0,
            )

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=min(max(score, 0.0), 1.0),
            rewards=rewards,
        )


async def main() -> None:
    task_ids = _parse_task_ids()
    if not task_ids:
        print("[ERROR] At least one task id is required (RCA_TASK_IDS or default list).", flush=True)
        raise SystemExit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "missing-key")
    env = await _build_env()

    try:
        for task_id in task_ids:
            await _run_episode(client, env, task_id)
    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
