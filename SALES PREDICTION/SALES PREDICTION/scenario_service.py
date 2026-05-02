from copy import deepcopy


def _top_factor_deltas(base_data, scenario_data):
    keys = [
        ("marketing_spend", "marketing_spend"),
        ("discount_percent", "discount_percent"),
        ("is_festival", "festival_flag"),
        ("campaign_active", "campaign_active"),
        ("price_index", "price_index")
    ]
    deltas = []
    for raw_key, label in keys:
        base_v = base_data.get(raw_key)
        new_v = scenario_data.get(raw_key)
        if base_v != new_v:
            deltas.append({"factor": label, "before": base_v, "after": new_v})
    return deltas


def _driver_lists(changes):
    positives = []
    negatives = []
    for ch in changes:
        factor = ch["factor"]
        before = ch["before"]
        after = ch["after"]
        try:
            before_num = float(before)
            after_num = float(after)
            if after_num > before_num:
                positives.append(factor)
            elif after_num < before_num:
                negatives.append(factor)
        except Exception:
            if str(after).strip().lower() in ("1", "true", "yes", "on"):
                positives.append(factor)
            elif str(after).strip().lower() in ("0", "false", "no", "off"):
                negatives.append(factor)
    return positives[:3], negatives[:3]


def run_scenario_simulation(base_data, scenario_changes, predictor):
    if not isinstance(base_data, dict) or not base_data:
        raise ValueError("base_data must be a non-empty object")
    if not isinstance(scenario_changes, dict):
        raise ValueError("scenario_changes must be an object")

    scenario_data = deepcopy(base_data)
    scenario_data.update(scenario_changes)

    base_prediction = predictor(base_data)
    scenario_prediction = predictor(scenario_data)

    base_value = float(base_prediction["predicted_sales"])
    scenario_value = float(scenario_prediction["predicted_sales"])

    expected_change = round(scenario_value - base_value, 2)
    percentage_change = round(((expected_change / base_value) * 100.0) if base_value > 0 else 0.0, 2)

    factor_changes = _top_factor_deltas(base_data, scenario_data)
    top_positive, top_negative = _driver_lists(factor_changes)

    return {
        "base_prediction": {
            "predicted_sales": round(base_value, 2),
            "model_prediction": base_prediction.get("model_prediction"),
            "warning": base_prediction.get("warning")
        },
        "scenario_prediction": {
            "predicted_sales": round(scenario_value, 2),
            "model_prediction": scenario_prediction.get("model_prediction"),
            "warning": scenario_prediction.get("warning")
        },
        "expected_change": expected_change,
        "percentage_change": percentage_change,
        "factor_changes": factor_changes,
        "top_positive_factors": top_positive,
        "top_negative_factors": top_negative,
        "summary": (
            f"Scenario changes are expected to {'increase' if expected_change >= 0 else 'decrease'} "
            f"sales by {abs(percentage_change)}%."
        )
    }
