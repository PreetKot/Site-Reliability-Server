from contextlib import asynccontextmanager
from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from env.data_generator import generate_all_scenarios
from env.environment import SREEnvironment
from env.models import Action, EpisodeState

env = SREEnvironment()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ = app
    scenarios_root = Path("scenarios")
    required_tasks = ["easy", "medium", "hard"]
    should_generate = False

    for task in required_tasks:
        task_dir = scenarios_root / task
        if not task_dir.exists() or not any(task_dir.glob("*.json")):
            should_generate = True
            break

    if should_generate:
        generate_all_scenarios()
    yield


app = FastAPI(
    title="Cloud Chaos SRE",
    description="OpenEnv environment simulating SRE incident response",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "env": "cloud-chaos-sre", "version": "1.0.0"}


@app.post("/reset")
def reset(body: dict):
    task_id = body.get("task_id", "easy")
    scenario_id = body.get("scenario_id", None)
    obs = env.reset(task_id=task_id, scenario_id=scenario_id)
    return obs.model_dump()


@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return env.state().model_dump()


@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "id": "easy",
                "name": "The Detective",
                "difficulty": "easy",
                "description": "Identify the root-cause microservice in a cascading failure",
                "max_steps": 15,
                "action_schema": Action.model_json_schema(),
            },
            {
                "id": "medium",
                "name": "The First Responder",
                "difficulty": "medium",
                "description": "Restore all system health metrics during a traffic spike",
                "max_steps": 15,
                "action_schema": Action.model_json_schema(),
            },
            {
                "id": "hard",
                "name": "The Architect",
                "difficulty": "hard",
                "description": "Diagnose and fix a hidden database timeout misconfiguration",
                "max_steps": 20,
                "action_schema": Action.model_json_schema(),
            },
        ]
    }


@app.post("/grader")
def grader(state: EpisodeState):
    env._state = state
    score, breakdown = env.grade()
    return {"task_id": state.task_id, "score": score, "breakdown": breakdown}


@app.post("/baseline")
def baseline():
    import json
    import subprocess
    import time

    inference_path = Path(__file__).with_name("inference.py")
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(inference_path), "--output-json"],
            capture_output=True,
            text=True,
            timeout=19 * 60,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "error": "inference_timeout",
            "message": "inference.py exceeded 19 minute timeout",
            "stdout_tail": (exc.stdout or "")[-1200:],
            "stderr_tail": (exc.stderr or "")[-1200:],
            "total_time_s": round(time.time() - start, 1),
        }

    elapsed = round(time.time() - start, 1)

    if result.returncode != 0:
        return {
            "ok": False,
            "error": "inference_failed",
            "returncode": result.returncode,
            "stdout_tail": (result.stdout or "")[-1200:],
            "stderr_tail": (result.stderr or "")[-1200:],
            "total_time_s": elapsed,
        }

    stdout_lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    candidate = stdout_lines[-1] if stdout_lines else ""

    try:
        loaded = json.loads(candidate)
    except Exception:
        return {
            "ok": False,
            "error": "invalid_inference_json",
            "message": "Could not parse JSON output from inference.py",
            "stdout_tail": (result.stdout or "")[-1200:],
            "stderr_tail": (result.stderr or "")[-1200:],
            "total_time_s": elapsed,
        }

    if not isinstance(loaded, dict):
        loaded = {"result": loaded}

    scores: dict[str, object] = dict(loaded)
    scores["ok"] = True
    scores["total_time_s"] = elapsed
    return scores
