import json
import random
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

SERVICES = [
    "api-gateway",
    "auth-service",
    "user-service",
    "order-service",
    "db-proxy",
    "cache-service",
]

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"

BASELINE_STATE = {
    "api-gateway": {"cpu_pct": 40.0, "mem_pct": 50.0, "error_rate": 0.001, "latency_ms": 110.0},
    "auth-service": {"cpu_pct": 38.0, "mem_pct": 45.0, "error_rate": 0.001, "latency_ms": 95.0},
    "user-service": {"cpu_pct": 42.0, "mem_pct": 52.0, "error_rate": 0.001, "latency_ms": 120.0},
    "order-service": {"cpu_pct": 44.0, "mem_pct": 54.0, "error_rate": 0.001, "latency_ms": 130.0},
    "db-proxy": {"cpu_pct": 36.0, "mem_pct": 42.0, "error_rate": 0.001, "latency_ms": 50.0},
    "cache-service": {"cpu_pct": 28.0, "mem_pct": 32.0, "error_rate": 0.001, "latency_ms": 12.0},
}

BASE_CONFIG = {
    "db_timeout": 5000,
    "pool_size": 10,
    "replica_count": 1,
    "ttl": 300,
}


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _base_deploy_history(service: str, changes_second: dict) -> list[dict]:
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return [
        {
            "deploy_id": "d001",
            "timestamp": _iso(base),
            "service": service,
            "changes": {"pool_size": 10},
        },
        {
            "deploy_id": "d002",
            "timestamp": _iso(base + timedelta(hours=1)),
            "service": service,
            "changes": changes_second,
        },
    ]


# ---------------------------------------------------------------------------
# Easy — The Detective
# Rotate root cause across ALL 6 services (not just db-proxy).
# Add red-herring noise on 1-2 other services so the signal isn't trivial.
# ---------------------------------------------------------------------------

def _make_easy(index: int) -> dict:
    state = deepcopy(BASELINE_STATE)
    config = deepcopy(BASE_CONFIG)
    root = random.choice(SERVICES)

    # Root service is clearly degraded
    state[root]["cpu_pct"] = round(random.uniform(82.0, 97.0), 2)
    state[root]["mem_pct"] = round(random.uniform(78.0, 94.0), 2)
    state[root]["error_rate"] = round(random.uniform(0.20, 0.55), 4)
    state[root]["latency_ms"] = round(random.uniform(1200.0, 2500.0), 2)

    # 1-2 red-herring services with mild degradation (not root cause)
    red_herrings = [s for s in SERVICES if s != root]
    random.shuffle(red_herrings)
    for svc in red_herrings[: random.randint(1, 2)]:
        state[svc]["error_rate"] = round(
            min(0.09, state[svc]["error_rate"] + random.uniform(0.02, 0.07)), 4
        )
        state[svc]["latency_ms"] = round(
            min(700.0, state[svc]["latency_ms"] + random.uniform(80.0, 400.0)), 2
        )

    return {
        "scenario_id": f"easy-{index:03d}",
        "task_id": "easy",
        "initial_state": state,
        "initial_config": config,
        "deploy_history": _base_deploy_history(root, {"pool_size": random.choice([5, 6, 7])}),
        "ground_truth": {
            "root_cause_service": root,
            "correct_action": "RESTART_SERVICE",
            "correct_config_key": None,
            "correct_config_value": None,
        },
    }


# ---------------------------------------------------------------------------
# Medium — The First Responder
# Traffic spike on api-gateway, order-service, or auth-service.
# Always degrade 3-4 metrics on the hotspot.
# Add secondary pressure to cache-service when api-gateway is the hotspot.
# ---------------------------------------------------------------------------

def _make_medium(index: int) -> dict:
    state = deepcopy(BASELINE_STATE)
    config = deepcopy(BASE_CONFIG)
    hotspot = random.choice(["api-gateway", "order-service", "auth-service"])

    # Always degrade all four metrics on the hotspot for a genuine spike
    state[hotspot]["cpu_pct"] = round(random.uniform(88.0, 97.0), 2)
    state[hotspot]["mem_pct"] = round(random.uniform(85.0, 96.0), 2)
    state[hotspot]["error_rate"] = round(random.uniform(0.06, 0.22), 4)
    state[hotspot]["latency_ms"] = round(random.uniform(900.0, 1800.0), 2)

    # Secondary pressure — makes brute-force SCALE_UP of one service insufficient
    if hotspot == "api-gateway":
        state["cache-service"]["latency_ms"] = round(random.uniform(180.0, 450.0), 2)
        state["cache-service"]["error_rate"] = round(random.uniform(0.03, 0.09), 4)
        state["cache-service"]["mem_pct"] = round(random.uniform(78.0, 88.0), 2)
    elif hotspot == "order-service":
        state["db-proxy"]["latency_ms"] = round(random.uniform(180.0, 400.0), 2)
        state["db-proxy"]["cpu_pct"] = round(random.uniform(68.0, 82.0), 2)
    elif hotspot == "auth-service":
        state["cache-service"]["error_rate"] = round(random.uniform(0.02, 0.07), 4)

    return {
        "scenario_id": f"medium-{index:03d}",
        "task_id": "medium",
        "initial_state": state,
        "initial_config": config,
        "deploy_history": _base_deploy_history(hotspot, {"replica_count": random.choice([1, 2])}),
        "ground_truth": {
            "root_cause_service": hotspot,
            "correct_action": "SCALE_UP",
            "correct_config_key": None,
            "correct_config_value": None,
        },
    }


# ---------------------------------------------------------------------------
# Hard — The Architect
# db_timeout misconfiguration buried among 3 deploy events.
# Red-herring configs (retry_count, connection_pool) that look suspicious
# but are not the actual root cause.
# ---------------------------------------------------------------------------

def _make_hard(index: int) -> dict:
    state = deepcopy(BASELINE_STATE)
    config = deepcopy(BASE_CONFIG)

    # The real misconfiguration
    config["db_timeout"] = 100
    # Red-herring configs (suboptimal but not causative)
    config["pool_size"] = random.choice([6, 7, 8])
    config["retry_count"] = random.choice([1, 2])
    config["connection_pool"] = random.choice([4, 5])
    config["ttl"] = random.choice([180, 240, 300])

    for svc in ["order-service", "user-service", "auth-service"]:
        state[svc]["error_rate"] = round(random.uniform(0.25, 0.55), 4)
        state[svc]["latency_ms"] = round(random.uniform(1200.0, 2600.0), 2)

    state["db-proxy"]["cpu_pct"] = round(random.uniform(80.0, 93.0), 2)
    state["db-proxy"]["mem_pct"] = round(random.uniform(75.0, 90.0), 2)
    state["db-proxy"]["error_rate"] = round(random.uniform(0.22, 0.45), 4)
    state["db-proxy"]["latency_ms"] = round(random.uniform(1400.0, 2600.0), 2)

    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    deploy_history = [
        # d001: innocuous pool tune
        {
            "deploy_id": "d001",
            "timestamp": _iso(base),
            "service": "db-proxy",
            "changes": {"pool_size": random.choice([10, 12])},
        },
        # d002: the bad deploy — db_timeout set too low
        {
            "deploy_id": "d002",
            "timestamp": _iso(base + timedelta(hours=1)),
            "service": "db-proxy",
            "changes": {"db_timeout": 100, "retry_count": config["retry_count"]},
        },
        # d003: a distracting red-herring deploy that changes a different key
        {
            "deploy_id": "d003",
            "timestamp": _iso(base + timedelta(hours=2)),
            "service": "db-proxy",
            "changes": {"connection_pool": config["connection_pool"], "ttl": config["ttl"]},
        },
    ]

    return {
        "scenario_id": f"hard-{index:03d}",
        "task_id": "hard",
        "initial_state": state,
        "initial_config": config,
        "deploy_history": deploy_history,
        "ground_truth": {
            "root_cause_service": "db-proxy",
            "correct_action": "UPDATE_CONFIG",
            "correct_config_key": "db_timeout",
            "correct_config_value": 5000,
        },
    }


# ---------------------------------------------------------------------------
# Expert — The Storm Chaser
# Dual failure: cache-service pool exhaustion AND db-proxy spike.
# Correct fix order: RESTART cache-service FIRST (its recovery eases db load),
# then RESTART db-proxy. Reverse order leaves db-proxy degraded.
# ---------------------------------------------------------------------------

def _make_expert(index: int) -> dict:
    state = deepcopy(BASELINE_STATE)
    config = deepcopy(BASE_CONFIG)

    # cache-service: pool exhaustion
    state["cache-service"]["cpu_pct"] = round(random.uniform(83.0, 96.0), 2)
    state["cache-service"]["mem_pct"] = round(random.uniform(86.0, 97.0), 2)
    state["cache-service"]["error_rate"] = round(random.uniform(0.18, 0.45), 4)
    state["cache-service"]["latency_ms"] = round(random.uniform(900.0, 2000.0), 2)

    # db-proxy: overloaded because cache is missed — cascades from cache
    state["db-proxy"]["cpu_pct"] = round(random.uniform(78.0, 92.0), 2)
    state["db-proxy"]["mem_pct"] = round(random.uniform(72.0, 88.0), 2)
    state["db-proxy"]["error_rate"] = round(random.uniform(0.12, 0.35), 4)
    state["db-proxy"]["latency_ms"] = round(random.uniform(800.0, 1800.0), 2)

    # Downstream: services depending on both are degraded
    for svc in ["auth-service", "user-service"]:
        state[svc]["error_rate"] = round(random.uniform(0.05, 0.18), 4)
        state[svc]["latency_ms"] = round(random.uniform(350.0, 900.0), 2)

    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    deploy_history = [
        {
            "deploy_id": "d001",
            "timestamp": _iso(base),
            "service": "cache-service",
            "changes": {"ttl": random.choice([240, 300])},
        },
        {
            "deploy_id": "d002",
            "timestamp": _iso(base + timedelta(minutes=30)),
            "service": "cache-service",
            "changes": {"pool_size": random.choice([4, 5])},
        },
        {
            "deploy_id": "d003",
            "timestamp": _iso(base + timedelta(hours=1)),
            "service": "db-proxy",
            "changes": {"replica_count": 1},
        },
    ]

    return {
        "scenario_id": f"expert-{index:03d}",
        "task_id": "expert",
        "initial_state": state,
        "initial_config": config,
        "deploy_history": deploy_history,
        "ground_truth": {
            "root_cause_service": "cache-service",
            "secondary_cause_service": "db-proxy",
            "correct_action": "RESTART_SERVICE",
            "correct_order": ["cache-service", "db-proxy"],
            "correct_config_key": None,
            "correct_config_value": None,
        },
    }


def generate_all_scenarios(seed: int = 42):
    random.seed(seed)
    for task in ["easy", "medium", "hard", "expert"]:
        (SCENARIOS_DIR / task).mkdir(parents=True, exist_ok=True)

    generators = {
        "easy": _make_easy,
        "medium": _make_medium,
        "hard": _make_hard,
        "expert": _make_expert,
    }

    for task, builder in generators.items():
        for idx in range(1, 11):
            scenario = builder(idx)
            out_path = SCENARIOS_DIR / task / f"{scenario['scenario_id']}.json"
            out_path.write_text(json.dumps(scenario, indent=2))


if __name__ == "__main__":
    generate_all_scenarios()
    print("Generated scenarios: easy=10, medium=10, hard=10, expert=10")
