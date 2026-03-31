from .models import EpisodeState


def _load_ground_truth(state: EpisodeState) -> dict:
    import json
    from pathlib import Path

    scenario_path = (
        Path(__file__).parent.parent
        / "scenarios"
        / state.task_id
        / f"{state.scenario_id}.json"
    )
    if not scenario_path.exists():
        return {}
    try:
        payload = json.loads(scenario_path.read_text())
        return payload.get("ground_truth", {})
    except Exception:
        return {}


def grade_easy(state: EpisodeState) -> tuple[float, dict]:
    """
    Task 1 — The Detective.
    Score breakdown:
      - Correct root service targeted with a diagnostic or remediation action: 0.40
      - Zero false-positive restarts (healthy services restarted): 0.30
      - System health restored (mean health score >= 0.85): 0.20
      - Efficiency bonus (done in <= 8 steps): 0.10

    Grader requires BOTH the `reason` field to mention the root service AND
    at least one action targeting that service (CHECK_LOGS, INSPECT_SERVICE,
    or RESTART_SERVICE) — simple string-matching on reason alone is insufficient.
    """
    root_cause = _load_ground_truth(state).get("root_cause_service", "db-proxy")

    breakdown = {
        "root_identified": 0.0,
        "no_false_positives": 0.0,
        "health_restored": 0.0,
        "efficiency": 0.0,
    }

    # Require: (a) reason text mentions the root service AND
    #          (b) an action actually targets the root service.
    DIAGNOSTIC_ACTIONS = {"CHECK_LOGS", "INSPECT_SERVICE", "RESTART_SERVICE"}
    root_mentioned_in_reason = False
    root_targeted_correctly = False

    for action in state.action_history:
        reason = (action.get("reason") or "").lower()
        target = action.get("target_service", "")
        action_type = action.get("action_type", "")

        if root_cause.replace("-", "") in reason or root_cause in reason:
            root_mentioned_in_reason = True

        if target == root_cause and action_type in DIAGNOSTIC_ACTIONS:
            root_targeted_correctly = True

    if root_mentioned_in_reason and root_targeted_correctly:
        breakdown["root_identified"] = 0.40
    elif root_targeted_correctly:
        # Partial credit: targeted correctly but didn't explain why
        breakdown["root_identified"] = 0.20

    # False positives: RESTART_SERVICE on a service that is NOT the root cause
    false_positives = sum(
        1
        for action in state.action_history
        if action.get("action_type") == "RESTART_SERVICE"
        and action.get("target_service") != root_cause
    )
    breakdown["no_false_positives"] = max(0.0, 0.30 - (false_positives * 0.10))

    final_health = state.observation.health_summary.overall
    breakdown["health_restored"] = round(min(0.20, final_health * 0.20), 4)

    if state.step <= 8:
        breakdown["efficiency"] = 0.10

    score = sum(breakdown.values())
    return round(min(1.0, score), 4), breakdown


def grade_medium(state: EpisodeState) -> tuple[float, dict]:
    """
    Task 2 — The First Responder.
    Score = mean of 4 independent metric scores (each 0.0-1.0).
    Each metric scored linearly from worst-case to threshold.
    """
    from .simulator import HEALTH_THRESHOLDS

    metrics = state.observation.metrics
    thresholds = HEALTH_THRESHOLDS
    worst_cases = {
        "cpu_pct": 100.0,
        "mem_pct": 100.0,
        "error_rate": 1.0,
        "latency_ms": 2000.0,
    }

    metric_scores: dict[str, float] = {}
    for metric in ["cpu_pct", "mem_pct", "error_rate", "latency_ms"]:
        values = getattr(metrics, metric).values()
        worst_val = max(values)
        threshold = thresholds[metric]
        worst = worst_cases[metric]
        if worst_val <= threshold:
            metric_scores[metric] = 1.0
        else:
            metric_scores[metric] = round(
                max(0.0, 1.0 - (worst_val - threshold) / (worst - threshold)),
                4,
            )

    score = round(sum(metric_scores.values()) / 4.0, 4)

    worst_service_per_metric: dict[str, str] = {}
    worst_value_per_metric: dict[str, float] = {}
    for metric in ["cpu_pct", "mem_pct", "error_rate", "latency_ms"]:
        series = getattr(metrics, metric)
        service, value = max(series.items(), key=lambda item: item[1])
        worst_service_per_metric[metric] = service
        worst_value_per_metric[metric] = value

    breakdown: dict[str, object] = dict(metric_scores)
    breakdown["worst_service_per_metric"] = worst_service_per_metric
    breakdown["worst_value_per_metric"] = worst_value_per_metric
    breakdown["thresholds"] = thresholds

    root = _load_ground_truth(state).get("root_cause_service")
    if root:
        breakdown["scenario_root_cause_service"] = root

    return score, breakdown


def grade_hard(state: EpisodeState) -> tuple[float, dict]:
    """
    Task 3 — The Architect.
    Score breakdown:
      - Diagnosis actions on db-proxy (CHECK_LOGS/INSPECT_SERVICE): 0.15
      - Correct config key identified (UPDATE_CONFIG with db_timeout): 0.25
      - Correct config value applied (5000): 0.35
      - Value-progress milestone (partial credit for plausible value range): up to 0.20
      - System health restored (overall >= 0.80): up to 0.03
      - Efficiency bonus (correct fix by step <= 10): 0.02
    """
    expected = _load_ground_truth(state)
    correct_key = expected.get("correct_config_key", "db_timeout")
    correct_value = expected.get("correct_config_value", 5000)

    components: dict[str, float] = {
        "diagnosis": 0.0,
        "correct_key": 0.0,
        "value_progress": 0.0,
        "correct_value": 0.0,
        "health_restored": 0.0,
        "efficiency": 0.0,
    }

    if any(
        action.get("target_service") == "db-proxy"
        and action.get("action_type") in {"CHECK_LOGS", "INSPECT_SERVICE"}
        for action in state.action_history
    ):
        components["diagnosis"] = 0.15

    for action in state.action_history:
        if action.get("action_type") != "UPDATE_CONFIG":
            continue
        if action.get("config_key") != correct_key:
            continue

        components["correct_key"] = 0.25
        val = action.get("config_value")
        if val is None:
            continue
        try:
            val_int = int(val)
        except (TypeError, ValueError):
            continue

        if val_int == correct_value:
            components["correct_value"] = 0.35
            components["value_progress"] = max(components["value_progress"], 0.20)
        elif 2500 <= val_int <= 8000:
            components["value_progress"] = max(components["value_progress"], 0.15)
        elif 1000 <= val_int < 2500:
            components["value_progress"] = max(components["value_progress"], 0.08)

    final_health = state.observation.health_summary.overall
    components["health_restored"] = round(min(0.03, final_health * 0.03), 4)

    if state.step <= 10 and components["correct_value"] > 0:
        components["efficiency"] = 0.02

    breakdown: dict[str, object] = dict(components)
    breakdown["scenario_expected_key"] = str(correct_key)
    breakdown["scenario_expected_value"] = int(correct_value)
    breakdown["update_actions_attempted"] = sum(
        1 for action in state.action_history if action.get("action_type") == "UPDATE_CONFIG"
    )

    score = sum(components.values())
    return round(min(1.0, score), 4), breakdown


def grade_expert(state: EpisodeState) -> tuple[float, dict]:
    """
    Task 4 — The Storm Chaser.
    Dual-service cascading failure: both cache-service AND db-proxy are degraded.
    The correct fix order is: RESTART cache-service FIRST, then RESTART db-proxy.
    Fixing db-proxy first does not cascade to heal cache-service.

    Score breakdown:
      - cache-service restarted before db-proxy: 0.25
      - db-proxy restarted (in any order): 0.20
      - System health restored (overall >= 0.85): 0.30
      - No unnecessary restarts of healthy services (api-gateway, auth, user, order): 0.15
      - Efficiency bonus (healed in <= 12 steps): 0.10
    """
    components: dict[str, float] = {
        "correct_order": 0.0,
        "db_restarted": 0.0,
        "health_restored": 0.0,
        "no_collateral": 0.0,
        "efficiency": 0.0,
    }

    SIDE_SERVICES = {"api-gateway", "auth-service", "user-service", "order-service"}

    restart_order: list[str] = [
        action["target_service"]
        for action in state.action_history
        if action.get("action_type") == "RESTART_SERVICE"
    ]

    cache_idx = next(
        (i for i, s in enumerate(restart_order) if s == "cache-service"), None
    )
    db_idx = next(
        (i for i, s in enumerate(restart_order) if s == "db-proxy"), None
    )

    if cache_idx is not None and db_idx is not None and cache_idx < db_idx:
        components["correct_order"] = 0.25
    elif cache_idx is not None:
        # Restarted cache but not db, or in wrong order — partial
        components["correct_order"] = 0.10

    if db_idx is not None:
        components["db_restarted"] = 0.20

    final_health = state.observation.health_summary.overall
    components["health_restored"] = round(min(0.30, final_health * 0.30), 4)

    collateral_restarts = sum(
        1
        for action in state.action_history
        if action.get("action_type") == "RESTART_SERVICE"
        and action.get("target_service") in SIDE_SERVICES
    )
    components["no_collateral"] = max(0.0, 0.15 - collateral_restarts * 0.05)

    if state.step <= 12 and final_health >= 0.85:
        components["efficiency"] = 0.10

    breakdown: dict[str, object] = dict(components)
    breakdown["restart_order"] = restart_order
    breakdown["cache_restarted_first"] = (
        cache_idx is not None and db_idx is not None and cache_idx < db_idx
    )

    score = sum(components.values())
    return round(min(1.0, score), 4), breakdown


GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
    "expert": grade_expert,
}
