from .models import EpisodeState


def _load_ground_truth(state: EpisodeState) -> dict:
    import json
    from pathlib import Path

    scenario_path = Path(__file__).parent.parent / "scenarios" / state.task_id / f"{state.scenario_id}.json"
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
      - Correct root service in any action's reason field: 0.40
      - Zero false-positive restarts (healthy services restarted): 0.30
      - System health restored (mean health score >= 0.85): 0.20
      - Efficiency bonus (done in <= 8 steps): 0.10
    """
    root_cause = _load_ground_truth(state).get("root_cause_service", "db-proxy")

    breakdown = {
        "root_identified": 0.0,
        "no_false_positives": 0.0,
        "health_restored": 0.0,
        "efficiency": 0.0,
    }

    for action in state.action_history:
        reason = (action.get("reason") or "").lower()
        if root_cause.replace("-", "") in reason or root_cause in reason:
            breakdown["root_identified"] = 0.40
            break

    false_positives = sum(
        1
        for action in state.action_history
        if action.get("action_type") == "RESTART_SERVICE" and action.get("target_service") != root_cause
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
      - Correct config key identified (UPDATE_CONFIG with db_timeout): 0.30
      - Correct config value applied (5000): 0.40
      - System health restored (overall >= 0.80): 0.10
      - Efficiency (found fix in <= 10 steps): 0.20
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


GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}
