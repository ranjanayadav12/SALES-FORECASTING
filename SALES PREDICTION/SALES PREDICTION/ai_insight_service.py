import os


def _fallback_insight(payload):
    base = payload.get("base_prediction", {}) if isinstance(payload, dict) else {}
    scenario = payload.get("scenario_prediction", {}) if isinstance(payload, dict) else {}
    pct = float(payload.get("percentage_change", 0) or 0)
    pos = payload.get("top_positive_factors", []) or []
    neg = payload.get("top_negative_factors", []) or []

    trend_word = "increase" if pct >= 0 else "decrease"
    pos_text = ", ".join(pos[:2]) if pos else "higher demand signals"
    neg_text = ", ".join(neg[:2]) if neg else "cost and seasonality effects"

    base_val = float(base.get("predicted_sales", 0) or 0)
    scenario_val = float(scenario.get("predicted_sales", 0) or 0)
    text = (
        f"Sales are expected to {trend_word} from {base_val:,.0f} to {scenario_val:,.0f} "
        f"({pct:+.1f}%). Main upside drivers: {pos_text}. Main downside risks: {neg_text}."
    )
    return {"insight_text": text, "source": "rule_based_fallback"}


def generate_ai_business_insight(payload):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _fallback_insight(payload)

    try:
        from openai import OpenAI

        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        client = OpenAI(api_key=api_key)
        prompt = (
            "You are an analytics copilot for a sales dashboard. "
            "Write 2 concise business insight sentences using this JSON payload. "
            "Focus on reason for change and suggested action. Payload: "
            f"{payload}"
        )
        response = client.responses.create(
            model=model_name,
            input=prompt,
            max_output_tokens=180
        )
        text = (response.output_text or "").strip()
        if not text:
            return _fallback_insight(payload)
        return {"insight_text": text, "source": f"openai:{model_name}"}
    except Exception:
        return _fallback_insight(payload)
