import argparse
import inspect
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import wandb
from accelerate import Accelerator
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import GRPOConfig, GRPOTrainer
from urllib3.util.retry import Retry

ACTION_TYPES = {
    "CHECK_LOGS",
    "INSPECT_SERVICE",
    "DRAIN_TRAFFIC",
    "RESTART_SERVICE",
    "SCALE_UP",
    "SCALE_DOWN",
    "ROLLBACK",
    "UPDATE_CONFIG",
    "SILENCE_ALERT",
    "ACKNOWLEDGE_PAGERDUTY",
    "SEND_SLACK_MESSAGE",
    "RESOLVE_PAGERDUTY",
}

INFRA_ACTIONS = {
    "CHECK_LOGS",
    "INSPECT_SERVICE",
    "DRAIN_TRAFFIC",
    "RESTART_SERVICE",
    "SCALE_UP",
    "SCALE_DOWN",
    "ROLLBACK",
    "UPDATE_CONFIG",
    "SILENCE_ALERT",
}

ENTERPRISE_ACTIONS = {
    "ACKNOWLEDGE_PAGERDUTY",
    "SEND_SLACK_MESSAGE",
    "RESOLVE_PAGERDUTY",
}

VALID_SERVICES = {
    "api-gateway",
    "auth-service",
    "user-service",
    "order-service",
    "db-proxy",
    "cache-service",
}


@dataclass
class EnvStepResult:
    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_http_session(total_retries: int = 5, backoff_factor: float = 0.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[408, 429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def reset_env(session: requests.Session, env_url: str, timeout: float) -> dict[str, Any]:
    response = session.post(
        f"{env_url.rstrip('/')}/reset",
        json={"task_id": "enterprise"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Invalid /reset response format")
    return payload


def step_env(
    session: requests.Session,
    env_url: str,
    action: dict[str, Any],
    timeout: float,
) -> EnvStepResult:
    response = session.post(
        f"{env_url.rstrip('/')}/step",
        json=action,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Invalid /step response format")

    reward_payload = payload.get("reward", {})
    reward_value = 0.0
    if isinstance(reward_payload, dict):
        reward_value = float(reward_payload.get("step_reward", 0.0))
    elif isinstance(reward_payload, (float, int)):
        reward_value = float(reward_payload)

    observation = payload.get("observation", {})
    info = payload.get("info", {})

    return EnvStepResult(
        observation=observation if isinstance(observation, dict) else {},
        reward=reward_value,
        done=bool(payload.get("done", False)),
        info=info if isinstance(info, dict) else {},
    )


def extract_json_object(text: str) -> str | None:
    if not text:
        return None

    fenced_match = re.search(r"```(?:json)?\\s*(\{.*?\})\\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_match:
        return fenced_match.group(1).strip()

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1].strip()
    return None


def fallback_action() -> dict[str, Any]:
    return {"action_type": "CHECK_LOGS", "target_service": "user-service"}


def parse_action_output(text: str) -> dict[str, Any]:
    json_blob = extract_json_object(text)
    if json_blob is None:
        return fallback_action()

    try:
        parsed = json.loads(json_blob)
    except json.JSONDecodeError:
        return fallback_action()

    if not isinstance(parsed, dict):
        return fallback_action()

    action_type = str(parsed.get("action_type", "CHECK_LOGS")).upper()
    target_service = str(parsed.get("target_service", "user-service"))

    if action_type not in ACTION_TYPES:
        action_type = "CHECK_LOGS"
    if target_service not in VALID_SERVICES:
        target_service = "user-service"

    action: dict[str, Any] = {
        "action_type": action_type,
        "target_service": target_service,
    }

    for optional_key in [
        "config_key",
        "config_value",
        "reason",
        "incident_id",
        "channel_name",
        "message_text",
        "params",
    ]:
        if optional_key in parsed:
            action[optional_key] = parsed[optional_key]

    return action


def _is_json_object_response(text: str) -> bool:
    json_blob = extract_json_object(text)
    if json_blob is None:
        return False
    try:
        parsed = json.loads(json_blob)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict)


def _validate_action_payload(payload: dict[str, Any]) -> bool:
    action_type = payload.get("action_type")
    target_service = payload.get("target_service")
    if action_type not in ACTION_TYPES:
        return False
    if target_service not in VALID_SERVICES:
        return False

    if action_type == "UPDATE_CONFIG":
        if payload.get("config_key") is None or payload.get("config_value") is None:
            return False

    if action_type in {"ACKNOWLEDGE_PAGERDUTY", "RESOLVE_PAGERDUTY"}:
        if not payload.get("incident_id") and not payload.get("params", {}).get("incident_id"):
            return False

    if action_type == "SEND_SLACK_MESSAGE":
        channel = payload.get("channel_name") or payload.get("params", {}).get("channel_name")
        message = payload.get("message_text") or payload.get("params", {}).get("message_text")
        if not channel or not message:
            return False

    return True


def _protocol_adherence_scores(actions: list[dict[str, Any]]) -> list[float]:
    scores: list[float] = []
    is_acknowledged = False
    is_notified = False
    infra_done = False
    is_resolved = False

    for payload in actions:
        action_type = payload.get("action_type")
        reward = 0.0

        if action_type in INFRA_ACTIONS and not is_acknowledged:
            reward -= 0.2

        if action_type == "ACKNOWLEDGE_PAGERDUTY":
            if not is_acknowledged and not is_notified and not infra_done and not is_resolved:
                is_acknowledged = True
                reward += 0.25
            else:
                reward -= 0.15
        elif action_type == "SEND_SLACK_MESSAGE":
            if is_acknowledged and not is_notified and not is_resolved:
                is_notified = True
                reward += 0.25
            else:
                reward -= 0.2
        elif action_type == "RESOLVE_PAGERDUTY":
            if is_acknowledged and is_notified and infra_done and not is_resolved:
                is_resolved = True
                reward += 0.3
            else:
                reward -= 0.3
        elif action_type in INFRA_ACTIONS:
            if is_acknowledged and is_notified and not is_resolved and not infra_done:
                infra_done = True
                reward += 0.2

        scores.append(round(max(-1.0, min(1.0, reward)), 4))

    return scores


def make_format_validity_reward_function():
    def format_validity_reward(prompts: list[str], completions: list[Any], **_: Any) -> list[float]:
        _ = prompts
        rewards: list[float] = []
        for completion in completions:
            completion_text = _completion_to_text(completion)
            rewards.append(0.2 if _is_json_object_response(completion_text) else -0.2)
        return rewards

    return format_validity_reward


def make_action_validity_reward_function():
    def action_validity_reward(prompts: list[str], completions: list[Any], **_: Any) -> list[float]:
        _ = prompts
        rewards: list[float] = []
        for completion in completions:
            completion_text = _completion_to_text(completion)
            payload = parse_action_output(completion_text)
            rewards.append(0.3 if _validate_action_payload(payload) else -0.3)
        return rewards

    return action_validity_reward


def make_protocol_adherence_reward_function():
    def protocol_adherence_reward(prompts: list[str], completions: list[Any], **_: Any) -> list[float]:
        _ = prompts
        actions: list[dict[str, Any]] = []
        for completion in completions:
            completion_text = _completion_to_text(completion)
            actions.append(parse_action_output(completion_text))
        return _protocol_adherence_scores(actions)

    return protocol_adherence_reward


def build_prompt(observation: dict[str, Any]) -> str:
    compact_obs = json.dumps(observation, separators=(",", ":"), ensure_ascii=True)
    return (
        "You are an on-call SRE agent for an enterprise workflow incident. "
        "Return ONLY one JSON object and no markdown. "
        "Required keys: action_type, target_service. "
        "Allowed action_type values: CHECK_LOGS, INSPECT_SERVICE, DRAIN_TRAFFIC, RESTART_SERVICE, "
        "SCALE_UP, SCALE_DOWN, ROLLBACK, UPDATE_CONFIG, SILENCE_ALERT, "
        "ACKNOWLEDGE_PAGERDUTY, SEND_SLACK_MESSAGE, RESOLVE_PAGERDUTY.\\n"
        "Observation:\\n"
        f"{compact_obs}\\n"
        "Action JSON:"
    )


def init_model(args: argparse.Namespace):
    """Load a 4-bit quantised base model and wrap it with LoRA adapters.

    Uses standard HuggingFace transformers + PEFT — no Unsloth dependency,
    so there are no Unsloth/TRL version-mismatch issues.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Prepare for gradient checkpointing with k-bit training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def _completion_to_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, list):
        chunks: list[str] = []
        for part in item:
            if isinstance(part, dict):
                chunks.append(str(part.get("content", "")))
            else:
                chunks.append(str(part))
        return "\n".join(chunks)
    if isinstance(item, dict):
        return str(item.get("content", ""))
    return str(item)


def make_env_reward_function(
    session: requests.Session,
    env_url: str,
    timeout: float,
):
    def env_reward_func(prompts: list[str], completions: list[Any], **_: Any) -> list[float]:
        _ = prompts
        rewards: list[float] = []
        episode_active = False
        episode_done = False

        for completion in completions:
            completion_text = _completion_to_text(completion)
            action_payload = parse_action_output(completion_text)

            if episode_done:
                rewards.append(-0.05)
                continue

            try:
                if not episode_active:
                    reset_env(session, env_url, timeout)
                    episode_active = True
                step_result = step_env(session, env_url, action_payload, timeout)
                rewards.append(step_result.reward)
                if step_result.done:
                    episode_done = True
            except Exception:
                rewards.append(-1.0)
                episode_done = True
        return rewards

    return env_reward_func


def build_prompt_dataset(
    session: requests.Session,
    env_url: str,
    timeout: float,
    n_prompts: int = 20,
) -> Dataset:
    tasks = ["easy", "medium", "hard", "expert", "enterprise"]
    prompts: list[str] = []
    print(f"Building prompt dataset ({n_prompts} prompts across {tasks})...")
    for i in range(n_prompts):
        task = tasks[i % len(tasks)]
        try:
            resp = session.post(
                f"{env_url.rstrip('/')}/reset",
                json={"task_id": task},
                timeout=timeout,
            )
            resp.raise_for_status()
            obs = resp.json()
            prompts.append(build_prompt(obs))
        except Exception as exc:
            print(f"  Warning: reset failed for task={task}: {exc}")
            prompts.append(
                "You are an on-call SRE agent. "
                "Return ONE JSON object only. "
                'Example: {"action_type": "CHECK_LOGS", "target_service": "api-gateway"}'
            )
    print(f"Dataset ready: {len(prompts)} prompts")
    return Dataset.from_dict({"prompt": prompts})


def save_training_curves(log_history: list[dict], output_path: str = ".") -> None:
    reward_steps: list[int] = []
    reward_vals: list[float] = []
    loss_steps: list[int] = []
    loss_vals: list[float] = []

    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        if "reward" in entry and entry["reward"] is not None:
            reward_steps.append(step)
            reward_vals.append(float(entry["reward"]))
        if "loss" in entry and entry["loss"] is not None:
            loss_steps.append(step)
            loss_vals.append(float(entry["loss"]))

    fig_r, ax_r = plt.subplots(figsize=(9, 5))
    if reward_vals:
        ax_r.plot(reward_steps, reward_vals, color="green", linewidth=1.5, label="step_reward")
        if len(reward_vals) >= 3:
            z = np.polyfit(reward_steps, reward_vals, 1)
            p = np.poly1d(z)
            ax_r.plot(
                reward_steps, p(reward_steps),
                "r--", linewidth=1.2,
                label=f"trend (slope={z[0]:+.5f})",
            )
        ax_r.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax_r.set_xlabel("Step")
        ax_r.set_ylabel("Mean Reward")
        start_r = reward_vals[0] if reward_vals else 0
        end_r = reward_vals[-1] if reward_vals else 0
        ax_r.set_title(f"Reward Curve  (start={start_r:+.3f} → end={end_r:+.3f})")
        ax_r.legend()
        ax_r.grid(True, alpha=0.3)
    else:
        ax_r.text(0.5, 0.5, "No reward data recorded", ha="center", va="center")
    fig_r.tight_layout()
    reward_out = os.path.join(output_path, "reward_curve.png")
    fig_r.savefig(reward_out, dpi=150, bbox_inches="tight")
    plt.close(fig_r)
    print(f"Saved {reward_out}")

    fig_l, ax_l = plt.subplots(figsize=(9, 5))
    if loss_vals:
        ax_l.plot(loss_steps, loss_vals, color="steelblue", linewidth=1.5, label="grpo_loss")
        ax_l.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax_l.set_xlabel("Step")
        ax_l.set_ylabel("GRPO Loss")
        start_l = loss_vals[0] if loss_vals else 0
        end_l = loss_vals[-1] if loss_vals else 0
        ax_l.set_title(f"Loss Curve  (start={start_l:+.3f} → end={end_l:+.3f})")
        ax_l.legend()
        ax_l.grid(True, alpha=0.3)
    else:
        ax_l.text(0.5, 0.5, "No loss data recorded", ha="center", va="center")
    fig_l.tight_layout()
    loss_out = os.path.join(output_path, "loss_curve.png")
    fig_l.savefig(loss_out, dpi=150, bbox_inches="tight")
    plt.close(fig_l)
    print(f"Saved {loss_out}")


def evaluate_agent(
    session: requests.Session,
    model,
    tokenizer,
    env_url: str,
    timeout: float,
    max_steps: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    label: str = "eval",
) -> float:
    """Run one full episode and return mean step reward — used for before/after comparison."""
    device = next(model.parameters()).device
    model.eval()

    try:
        observation = session.post(
            f"{env_url.rstrip('/')}/reset",
            json={"task_id": "easy"},
            timeout=timeout,
        ).json()
    except Exception:
        return 0.0

    rewards: list[float] = []
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    for _ in range(max_steps):
        prompt = build_prompt(observation)
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(device)
        try:
            with torch.no_grad():
                out = model.generate(**inputs, **generation_kwargs)
            text = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
        except Exception:
            text = json.dumps(fallback_action())
        action = parse_action_output(text)
        try:
            result = step_env(session, env_url, action, timeout)
            rewards.append(result.reward)
            observation = result.observation
            if result.done:
                break
        except Exception:
            break

    model.train()
    mean = sum(rewards) / max(len(rewards), 1)
    print(f"[{label}] steps={len(rewards)} mean_reward={mean:+.4f}")
    return mean


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    os.environ["WANDB_MODE"] = "disabled"

    session = build_http_session()
    model, tokenizer = init_model(args)

    train_dataset = build_prompt_dataset(
        session, args.env_url, args.request_timeout, n_prompts=args.dataset_size
    )

    print("\n=== Pre-training evaluation ===")
    pre_score = evaluate_agent(
        session, model, tokenizer,
        args.env_url, args.request_timeout,
        args.max_steps, args.max_new_tokens,
        args.temperature, args.top_p,
        label="before_training",
    )

    env_reward_fn = make_env_reward_function(
        session=session,
        env_url=args.env_url,
        timeout=args.request_timeout,
    )
    format_reward_fn = make_format_validity_reward_function()
    action_reward_fn = make_action_validity_reward_function()
    protocol_reward_fn = make_protocol_adherence_reward_function()

    # Conditionally disable Dr. GRPO (added in trl>=0.12) to use the standard
    # GRPO loss that doesn't require old_logps / ref_logps.
    _grpo_fields = set(inspect.signature(GRPOConfig.__init__).parameters)
    _extra_config: dict[str, Any] = {}
    if "use_dr_grpo" in _grpo_fields:
        _extra_config["use_dr_grpo"] = False

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        num_train_epochs=args.epochs,
        report_to="none",
        logging_steps=1,
        save_steps=999999,
        **_extra_config,
    )

    # Support both TRL ≤0.11 (tokenizer=) and TRL ≥0.12 (processing_class=)
    _trainer_sig = set(inspect.signature(GRPOTrainer.__init__).parameters)
    _tok_kwarg = (
        {"processing_class": tokenizer}
        if "processing_class" in _trainer_sig
        else {"tokenizer": tokenizer}
    )

    grpo_trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            env_reward_fn,
            format_reward_fn,
            action_reward_fn,
            protocol_reward_fn,
        ],
        args=grpo_config,
        train_dataset=train_dataset,
        **_tok_kwarg,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n=== GRPO Training ===")
    grpo_trainer.train()

    grpo_trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n=== Saving training curves ===")
    save_training_curves(grpo_trainer.state.log_history, output_path=".")

    print("\n=== Post-training evaluation ===")
    post_score = evaluate_agent(
        session, model, tokenizer,
        args.env_url, args.request_timeout,
        args.max_steps, args.max_new_tokens,
        args.temperature, args.top_p,
        label="after_training",
    )

    delta = post_score - pre_score
    print(f"\n=== Training Summary ===")
    print(f"  Pre-training  mean reward : {pre_score:+.4f}")
    print(f"  Post-training mean reward : {post_score:+.4f}")
    print(f"  Improvement               : {delta:+.4f}")
    print(f"  Model saved to            : {args.output_dir}")
    print(f"  Curves saved to           : reward_curve.png, loss_curve.png")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GRPO training for SRE OpenEnv")
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model ID. Default: Qwen/Qwen2.5-1.5B-Instruct (fits Colab T4).",
    )
    parser.add_argument("--env_url", type=str, default="http://localhost:7860")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Full training epochs (3 shows a reward trend on Colab T4).")
    parser.add_argument("--max_steps", type=int, default=15,
                        help="Max env steps per evaluation episode.")
    parser.add_argument("--dataset_size", type=int, default=20,
                        help="Number of prompts built from live env resets.")
    parser.add_argument("--output_dir", type=str, default="./artifacts/grpo-sre")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--request_timeout", type=float, default=20.0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_prompt_length", type=int, default=1024,
                        help="Truncate prompts to this length to fit Colab T4 VRAM.")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_generations", type=int, default=4,
                        help="Completions per prompt for advantage estimation. Min 4.")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    return parser


if __name__ == "__main__":
    cli_args = build_arg_parser().parse_args()
    train(cli_args)
