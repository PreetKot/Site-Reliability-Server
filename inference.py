"""
inference.py  —  OpenEnv hackathon inference script.

MUST be named inference.py and placed in the root directory.
Reads OPENAI_API_KEY, API_BASE_URL, MODEL_NAME, HF_TOKEN from environment.

The agent uses a structured SRE reasoning prompt.  No guardrails — the LLM
reasons from observations alone, making scores reproducible and honest.
"""

import argparse
import json
import os
import signal
import time

import requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

_ = HF_TOKEN  # read but not used directly (may be required by the HF Space runtime)

ENV_BASE_URL = "http://localhost:7860"
TEMPERATURE = 0.0
MAX_TOKENS = 600
SEED = 42
TASKS = ["easy", "medium", "hard", "expert"]

FALLBACK_ACTION = json.dumps(
    {
        "action_type": "CHECK_LOGS",
        "target_service": "api-gateway",
        "config_key": None,
        "config_value": None,
        "reason": "Fallback: checking gateway logs to understand current state",
    }
)

SYSTEM_PROMPT = """\
You are an expert Site Reliability Engineer (SRE) responding to a live production incident.
At each step you receive a full system observation. You must:

1. Read ALL provided fields: metrics, active_alerts, logs, current_config, deploy_history, service_graph.
2. Reason about the most likely root cause — check which service is worst, what the logs say,
   and whether deploy_history shows a recent config change that correlates with degradation.
3. Choose exactly ONE action from:
   CHECK_LOGS, INSPECT_SERVICE, RESTART_SERVICE, SCALE_UP, SCALE_DOWN,
   ROLLBACK, UPDATE_CONFIG, SILENCE_ALERT
4. Target exactly ONE service: api-gateway, auth-service, user-service, order-service, db-proxy, cache-service
5. If using UPDATE_CONFIG, specify config_key and config_value from current_config.
6. Provide a concise reason explaining your diagnosis.

Key SRE heuristics:
- High error rates that span multiple dependent services → trace the dependency graph upstream.
- db_timeout in current_config below 1000ms is nearly always the root cause of cascading timeouts.
- SCALE_UP reduces CPU/latency but doesn't fix error-rate bugs — use it for traffic spikes only.
- RESTART_SERVICE clears memory leaks and resets transient errors.
- ROLLBACK reverts the latest deployment — useful when deploy_history shows a recent bad change.
- After fixing a service, SILENCE_ALERT on that service to clean up and earn a bonus.

Respond ONLY in valid JSON. No other text. Schema:
{"action_type": "...", "target_service": "...", "config_key": null, "config_value": null, "reason": "..."}"""


def _timeout_handler(signum, frame):
    _ = (signum, frame)
    print("TIMEOUT: 19-minute limit reached. Saving partial scores.")
    raise TimeoutError()


signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(19 * 60)

client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)


def call_env(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{ENV_BASE_URL}{path}"
    response = requests.request(method, url, json=body, timeout=30)
    response.raise_for_status()
    return response.json()


def parse_action(text: str) -> dict:
    import re

    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return json.loads(FALLBACK_ACTION)


def format_observation(obs: dict, step: int) -> str:
    metrics = obs.get("metrics", {})
    config = obs.get("current_config", {})
    alerts = [
        f"{a['service']}:{a['metric']}={a['current']:.2f} [{a['severity']}]"
        for a in obs.get("active_alerts", [])
        if not a.get("silenced", False)
    ]
    logs = [
        f"[{entry['severity']}] {entry['service']}: {entry['message']}"
        for entry in obs.get("logs", [])[-6:]
    ]
    deploys = [
        f"  {d['deploy_id']} @ {d['timestamp'][:16]} → {d['service']}: {d['changes']}"
        for d in obs.get("deploy_history", [])
    ]
    health = obs.get("health_summary", {}).get("per_service", {})

    health_str = "  " + "\n  ".join(
        f"{svc}: {score:.2f}" for svc, score in sorted(health.items(), key=lambda x: x[1])
    )

    return f"""Step {step}/{obs.get('max_steps')} | Task: {obs.get('task_id')}

HEALTH SCORES (0=critical, 1=perfect):
{health_str}

METRICS:
  CPU%:       {json.dumps({k: round(v, 1) for k, v in metrics.get('cpu_pct', {}).items()})}
  Memory%:    {json.dumps({k: round(v, 1) for k, v in metrics.get('mem_pct', {}).items()})}
  Error rate: {json.dumps({k: round(v, 4) for k, v in metrics.get('error_rate', {}).items()})}
  Latency ms: {json.dumps({k: round(v, 0) for k, v in metrics.get('latency_ms', {}).items()})}

ACTIVE ALERTS: {', '.join(alerts) or 'none'}

CURRENT CONFIG: {json.dumps(config)}

RECENT LOGS:
{chr(10).join(logs) or '  (none)'}

DEPLOY HISTORY (newest last):
{chr(10).join(deploys) or '  (none)'}"""


def run_task(task_id: str) -> dict:
    print("\n" + "=" * 55)
    print(f"  Running task: {task_id}")
    print("=" * 55)

    obs = call_env("POST", "/reset", {"task_id": task_id})
    max_steps = obs.get("max_steps", 15)
    done = False
    step = 0
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    while not done and step < max_steps:
        step += 1
        user_content = format_observation(obs, step)

        # Append user observation to rolling conversation (keeps context across steps)
        messages.append({"role": "user", "content": user_content})

        response_text = FALLBACK_ACTION
        for attempt in range(3):
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    seed=SEED,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or FALLBACK_ACTION
                break
            except Exception as exc:
                print(f"  Attempt {attempt + 1} failed: {exc}")
                if attempt == 2:
                    response_text = FALLBACK_ACTION

        action_dict = parse_action(response_text)
        # Append assistant response to maintain conversation continuity
        messages.append({"role": "assistant", "content": response_text})

        print(
            f"  Step {step:02d}: {action_dict.get('action_type'):<18} → {action_dict.get('target_service')}"
        )

        result = call_env("POST", "/step", action_dict)
        obs = result.get("observation", obs)
        done = result.get("done", False)
        if result.get("info", {}).get("silence_bonus"):
            print(f"          ↳ SILENCE_ALERT cleanup bonus earned on {action_dict.get('target_service')}")

        if done:
            print("  Episode complete.")
            break

    state = call_env("GET", "/state")
    grader_result = call_env("POST", "/grader", state)
    score = grader_result.get("score", 0.0)

    print(f"  Final score: {score:.4f}")
    return {
        "task_id": task_id,
        "score": score,
        "steps": step,
        "breakdown": grader_result.get("breakdown", {}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", action="store_true", help="Output JSON only")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=TASKS,
        choices=TASKS,
        help="Which tasks to run (default: all)",
    )
    args = parser.parse_args()

    scores: dict[str, dict] = {}
    start = time.time()

    try:
        for task_id in args.tasks:
            scores[task_id] = run_task(task_id)
    except TimeoutError:
        print("Inference timed out, partial scores saved.")
    finally:
        signal.alarm(0)

    mean_score = round(
        sum(result["score"] for result in scores.values()) / max(len(scores), 1), 4
    )
    output = {
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "seed": SEED,
        "scores": scores,
        "mean_score": mean_score,
        "total_time_s": round(time.time() - start, 1),
    }

    with open("baseline_scores.json", "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    if args.output_json:
        print(json.dumps(output))
    else:
        print("\n" + "=" * 55)
        print("BASELINE RESULTS")
        print("=" * 55)
        for task_id, result in scores.items():
            print(f"  {task_id:<10}: {result['score']:.4f}  ({result['steps']} steps)")
        print(f"  {'MEAN':<10}: {mean_score:.4f}")
        print("\nSaved to baseline_scores.json")


if __name__ == "__main__":
    main()
