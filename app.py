"""PlantIQX AI Backend.

Flask service that powers the Admin UI AI workflows (`/iot/ai-insights`):
operational summaries, recommendations, predictions, anomaly scan, root-cause
analysis and an interactive operations assistant.

Provider-agnostic: works with OpenAI **or** any OpenAI-compatible endpoint
(Groq, Together, OpenRouter, local vLLM, ...). The provider is auto-detected
from the API key prefix, or can be forced with AI_PROVIDER / AI_BASE_URL.

If no key is configured (or the model call fails) the service degrades
gracefully to deterministic local heuristics so the UI never breaks.
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

load_dotenv()

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - optional import for fallback mode
    ChatPromptTemplate = None
    ChatOpenAI = None


# ─── Provider configuration ───────────────────────────────────────────────
# Known OpenAI-compatible providers. Key = provider id, value = (base_url,
# default model, api-key prefix used for auto-detection).
PROVIDERS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "prefix": "sk-",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "prefix": "gsk_",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "openai/gpt-4o-mini",
        "prefix": "sk-or-",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "prefix": "tgp_",
    },
}


def detect_provider(api_key: str) -> str:
    """Resolve the provider id from explicit env or the API-key prefix."""
    forced = os.getenv("AI_PROVIDER", "").strip().lower()
    if forced in PROVIDERS:
        return forced
    key = (api_key or "").strip()
    # Order matters: sk-or- must be checked before sk-.
    if key.startswith("sk-or-"):
        return "openrouter"
    if key.startswith("gsk_"):
        return "groq"
    if key.startswith("tgp_"):
        return "together"
    if key.startswith("sk-"):
        return "openai"
    return "openai"


def resolve_ai_config() -> Dict[str, Any]:
    """Return the active AI configuration (provider, model, base_url, key)."""
    api_key = (
        os.getenv("OPENAI_API_KEY", "")
        or os.getenv("AI_API_KEY", "")
        or os.getenv("GROQ_API_KEY", "")
    ).strip()
    provider = detect_provider(api_key)
    spec = PROVIDERS.get(provider, PROVIDERS["openai"])

    base_url = os.getenv("AI_BASE_URL", "").strip() or spec["base_url"]
    # AI_MODEL is honoured only when it is plausibly valid for the provider.
    configured_model = os.getenv("AI_MODEL", "").strip()
    model = configured_model or spec["default_model"]
    # gpt-* models do not exist on non-OpenAI providers — fall back to default.
    if provider != "openai" and model.startswith("gpt-"):
        model = spec["default_model"]

    return {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "configured": bool(api_key),
    }


AI_CONFIG = resolve_ai_config()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp_percent(value: Any) -> int:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, round(parsed)))


def get_llm(temperature: float = 0.2, max_tokens: int = 800):
    """Build a ChatOpenAI client pointed at the active provider, or None."""
    if not AI_CONFIG["configured"] or ChatOpenAI is None:
        return None
    try:
        return ChatOpenAI(
            model=AI_CONFIG["model"],
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=AI_CONFIG["api_key"],
            base_url=AI_CONFIG["base_url"],
            timeout=30,
            max_retries=1,
        )
    except Exception:
        return None


def run_llm_json(system: str, human: str, variables: Dict[str, Any],
                 temperature: float = 0.2, max_tokens: int = 800) -> Dict[str, Any]:
    """Invoke the LLM and parse a strict-JSON response. Returns {} on failure."""
    llm = get_llm(temperature=temperature, max_tokens=max_tokens)
    if llm is None or ChatPromptTemplate is None:
        return {}
    try:
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
        response = (prompt | llm).invoke(variables)
        return parse_llm_json(getattr(response, "content", ""))
    except Exception as exc:  # pragma: no cover - network/runtime failures
        app.logger.warning("LLM JSON call failed: %s", exc)
        return {}


def run_llm_text(system: str, human: str, variables: Dict[str, Any],
                 temperature: float = 0.3, max_tokens: int = 800) -> str:
    """Invoke the LLM and return plain text. Returns '' on failure."""
    llm = get_llm(temperature=temperature, max_tokens=max_tokens)
    if llm is None or ChatPromptTemplate is None:
        return ""
    try:
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
        response = (prompt | llm).invoke(variables)
        return (getattr(response, "content", "") or "").strip()
    except Exception as exc:  # pragma: no cover - network/runtime failures
        app.logger.warning("LLM text call failed: %s", exc)
        return ""


def parse_llm_json(content: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from an LLM response (handles code fences)."""
    raw = (content or "").strip()
    if not raw:
        return {}

    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        # Last resort: grab the outermost {...} block.
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(raw[start:end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}


def _str_list(value: Any, limit: int) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()][:limit]


# ─── Context helpers ──────────────────────────────────────────────────────
def context_totals(context: Dict[str, Any]) -> Dict[str, int]:
    totals = context.get("totals", {}) if isinstance(context, dict) else {}
    return {
        "total_assets": int(totals.get("totalAssets", 0) or 0),
        "online": int(totals.get("online", 0) or 0),
        "warning": int(totals.get("warning", 0) or 0),
        "offline": int(totals.get("offline", 0) or 0),
        "total_alerts": int(totals.get("totalAlerts", 0) or 0),
        "critical_alerts": int(totals.get("criticalAlerts", 0) or 0),
        "open_alerts": int(totals.get("openAlerts", 0) or 0),
    }


# ─── Insight summary ──────────────────────────────────────────────────────
def local_summary(context: Dict[str, Any]) -> Dict[str, Any]:
    t = context_totals(context)
    uptime = clamp_percent(
        (t["online"] / t["total_assets"]) * 100 if t["total_assets"] > 0 else 0
    )

    recommendations: List[str] = []
    if t["critical_alerts"] > 0:
        recommendations.append(f"Investigate {t['critical_alerts']} critical alert(s) immediately.")
    if t["offline"] > 0:
        recommendations.append(f"Recover {t['offline']} offline asset(s) and verify connectivity.")
    if t["warning"] > 0:
        recommendations.append(f"Calibrate warning thresholds on {t['warning']} asset(s).")
    if not recommendations:
        recommendations.append("System is stable. Continue monitoring for drift and anomalies.")

    predictions: List[str] = []
    if t["warning"] > 0:
        predictions.append(
            f"{max(1, round(t['warning'] * 0.4))} warning asset(s) may degrade in the next 24 hours without intervention."
        )
    if t["total_alerts"] > 0:
        predictions.append(
            f"Alert volume may increase by {max(5, round((t['total_alerts'] / max(t['total_assets'], 1)) * 12))}% in the next shift."
        )
    if not predictions:
        predictions.append("No high-risk drift signal detected for the selected context.")

    risk_level = (
        "high" if t["critical_alerts"] > 0
        else "medium" if (t["warning"] > 0 or t["offline"] > 0)
        else "low"
    )

    return {
        "summary": (
            f"Fleet analysis shows {uptime}% uptime with {t['total_alerts']} alerts "
            f"({t['critical_alerts']} critical). {t['offline']} assets are offline "
            f"and {t['warning']} are in warning state."
        ),
        "headline": f"{uptime}% fleet uptime — {risk_level.upper()} risk",
        "recommendedActions": recommendations,
        "predictions": predictions,
        "keyMetrics": [
            {"label": "Fleet Uptime", "value": f"{uptime}%"},
            {"label": "Critical Alerts", "value": str(t["critical_alerts"])},
            {"label": "Offline Assets", "value": str(t["offline"])},
            {"label": "Open Alerts", "value": str(t["open_alerts"])},
        ],
        "confidence": 62,
        "riskLevel": risk_level,
        "source": "fallback",
        "model": "heuristic",
        "generatedAt": utc_now_iso(),
    }


def generate_summary_with_llm(context: Dict[str, Any], focus: str) -> Dict[str, Any]:
    parsed = run_llm_json(
        system=(
            "You are an industrial IoT operations copilot for a manufacturing plant. "
            "Analyse the telemetry context and return STRICT JSON only — no prose, no markdown. "
            "Required keys: "
            "summary (string, under 75 words, concrete numbers), "
            "headline (string, under 12 words, punchy), "
            "recommendedActions (array of 3-5 short imperative strings, most urgent first), "
            "predictions (array of 2-4 short forward-looking strings with rough magnitudes), "
            "keyMetrics (array of {{label, value}} objects, 3-5 items), "
            "confidence (integer 0-100), "
            "riskLevel (one of: low, medium, high)."
        ),
        human="Focus area: {focus}\n\nTelemetry context (JSON):\n{context_json}",
        variables={
            "focus": focus or "iot-operations",
            "context_json": json.dumps(context, ensure_ascii=True)[:12000],
        },
        temperature=0.2,
        max_tokens=900,
    )
    if not parsed:
        return local_summary(context)

    fallback = local_summary(context)
    normalized = {
        "summary": str(parsed.get("summary") or fallback["summary"]),
        "headline": str(parsed.get("headline") or fallback["headline"]),
        "recommendedActions": _str_list(parsed.get("recommendedActions"), 5)
        or fallback["recommendedActions"],
        "predictions": _str_list(parsed.get("predictions"), 4) or fallback["predictions"],
        "keyMetrics": parsed.get("keyMetrics")
        if isinstance(parsed.get("keyMetrics"), list) and parsed.get("keyMetrics")
        else fallback["keyMetrics"],
        "confidence": clamp_percent(parsed.get("confidence", 70)),
        "riskLevel": str(parsed.get("riskLevel", "medium")).lower(),
        "source": "llm",
        "model": AI_CONFIG["model"],
        "generatedAt": utc_now_iso(),
    }
    return normalized


# ─── Anomaly scan ─────────────────────────────────────────────────────────
def local_anomalies(context: Dict[str, Any]) -> Dict[str, Any]:
    t = context_totals(context)
    anomalies: List[Dict[str, Any]] = []

    if t["critical_alerts"] > 0:
        anomalies.append({
            "title": "Critical alert cluster",
            "detail": f"{t['critical_alerts']} critical alert(s) are open across the fleet.",
            "severity": "high",
            "scope": "fleet",
        })
    if t["offline"] > 0:
        anomalies.append({
            "title": "Connectivity loss",
            "detail": f"{t['offline']} asset(s) stopped reporting telemetry.",
            "severity": "high" if t["offline"] > 2 else "medium",
            "scope": "connectivity",
        })
    if t["warning"] > 0:
        anomalies.append({
            "title": "Threshold drift",
            "detail": f"{t['warning']} asset(s) are trending toward limit breaches.",
            "severity": "medium",
            "scope": "thresholds",
        })

    dept_health = context.get("departmentHealth", []) if isinstance(context, dict) else []
    for dept in dept_health[:3]:
        health = int(dept.get("health", 100) or 100)
        if health < 60:
            anomalies.append({
                "title": f"Low department health — {dept.get('department', 'Unknown')}",
                "detail": f"Only {health}% of assets healthy in this department.",
                "severity": "high" if health < 40 else "medium",
                "scope": "department",
            })

    if not anomalies:
        anomalies.append({
            "title": "No anomalies detected",
            "detail": "Telemetry is within expected ranges for the selected window.",
            "severity": "low",
            "scope": "fleet",
        })

    return {
        "anomalies": anomalies,
        "anomalyCount": len([a for a in anomalies if a["severity"] != "low"]),
        "source": "fallback",
        "model": "heuristic",
        "generatedAt": utc_now_iso(),
    }


def generate_anomalies_with_llm(context: Dict[str, Any]) -> Dict[str, Any]:
    parsed = run_llm_json(
        system=(
            "You are an anomaly-detection engine for industrial IoT telemetry. "
            "Inspect the context and identify operational anomalies, outliers and risk clusters. "
            "Return STRICT JSON only with key 'anomalies': an array of objects, each with "
            "title (string), detail (string with numbers), severity (low|medium|high), "
            "scope (string, e.g. fleet/department/connectivity/thresholds). "
            "Return 0-6 anomalies, most severe first. If nothing is wrong return an empty array."
        ),
        human="Telemetry context (JSON):\n{context_json}",
        variables={"context_json": json.dumps(context, ensure_ascii=True)[:12000]},
        temperature=0.15,
        max_tokens=800,
    )
    anomalies = parsed.get("anomalies") if isinstance(parsed, dict) else None
    if not isinstance(anomalies, list):
        return local_anomalies(context)

    cleaned: List[Dict[str, Any]] = []
    for item in anomalies[:6]:
        if not isinstance(item, dict):
            continue
        cleaned.append({
            "title": str(item.get("title", "Anomaly")).strip()[:120],
            "detail": str(item.get("detail", "")).strip()[:300],
            "severity": str(item.get("severity", "medium")).lower(),
            "scope": str(item.get("scope", "fleet")).strip()[:40],
        })

    if not cleaned:
        local = local_anomalies(context)
        local["source"] = "llm"
        local["model"] = AI_CONFIG["model"]
        return local

    return {
        "anomalies": cleaned,
        "anomalyCount": len([a for a in cleaned if a["severity"] != "low"]),
        "source": "llm",
        "model": AI_CONFIG["model"],
        "generatedAt": utc_now_iso(),
    }


# ─── Root-cause analysis ──────────────────────────────────────────────────
def local_root_cause(context: Dict[str, Any], subject: str) -> Dict[str, Any]:
    t = context_totals(context)
    causes = []
    if t["critical_alerts"] > 0:
        causes.append("Unresolved critical alerts indicate active fault conditions.")
    if t["offline"] > 0:
        causes.append("Offline assets suggest gateway connectivity or power issues.")
    if t["warning"] > 0:
        causes.append("Sustained threshold drift points to wear or calibration loss.")
    if not causes:
        causes.append("No dominant fault signature in the current telemetry window.")

    return {
        "subject": subject or "Fleet operations",
        "probableCauses": causes,
        "investigationSteps": [
            "Pull the asset telemetry trend for the affected window.",
            "Cross-check maintenance history and recent configuration changes.",
            "Verify gateway connectivity and sensor calibration.",
        ],
        "confidence": 55,
        "source": "fallback",
        "model": "heuristic",
        "generatedAt": utc_now_iso(),
    }


def generate_root_cause_with_llm(context: Dict[str, Any], subject: str) -> Dict[str, Any]:
    parsed = run_llm_json(
        system=(
            "You are a reliability engineer doing root-cause analysis on industrial assets. "
            "Given the telemetry context and a subject (asset/department/issue), reason about "
            "the most probable causes. Return STRICT JSON with keys: "
            "subject (string), probableCauses (array of 2-4 strings, most likely first), "
            "investigationSteps (array of 3-5 concrete next steps), confidence (integer 0-100)."
        ),
        human="Subject under investigation: {subject}\n\nTelemetry context (JSON):\n{context_json}",
        variables={
            "subject": subject or "fleet operations",
            "context_json": json.dumps(context, ensure_ascii=True)[:12000],
        },
        temperature=0.25,
        max_tokens=700,
    )
    if not parsed:
        return local_root_cause(context, subject)

    fallback = local_root_cause(context, subject)
    return {
        "subject": str(parsed.get("subject") or subject or "Fleet operations"),
        "probableCauses": _str_list(parsed.get("probableCauses"), 4)
        or fallback["probableCauses"],
        "investigationSteps": _str_list(parsed.get("investigationSteps"), 5)
        or fallback["investigationSteps"],
        "confidence": clamp_percent(parsed.get("confidence", 60)),
        "source": "llm",
        "model": AI_CONFIG["model"],
        "generatedAt": utc_now_iso(),
    }


# ─── Chat assistant ───────────────────────────────────────────────────────
def local_chat_answer(message: str, context: Dict[str, Any]) -> str:
    t = context_totals(context)
    text = (message or "").lower()
    if "predict" in text:
        return (
            f"Prediction: {max(1, round((t['warning'] + t['offline']) * 0.4))} assets may require "
            "intervention within the next 24 hours if the current trend continues."
        )
    if "summary" in text or "summarize" in text:
        return (
            f"Summary: {t['total_assets']} assets in scope, {t['total_alerts']} alerts, "
            f"{t['critical_alerts']} critical. Prioritise critical closure and offline recovery."
        )
    if "root" in text or "cause" in text or "why" in text:
        return (
            "Likely drivers: unresolved critical alerts, connectivity loss on offline assets, "
            "and threshold drift on warning assets. Start with the highest-severity alert."
        )
    return (
        f"Recommended next step: handle {t['critical_alerts']} critical alert(s), then recover "
        f"{t['offline']} offline asset(s), then tune thresholds on {t['warning']} warning asset(s)."
    )


def generate_chat_with_llm(message: str, context: Dict[str, Any],
                           history: List[Dict[str, Any]]) -> str:
    safe_history = history[-8:] if isinstance(history, list) else []
    answer = run_llm_text(
        system=(
            "You are PlantIQX Copilot, an industrial AI assistant for plant operators. "
            "Give concise, actionable, operations-first guidance grounded in the provided "
            "telemetry context. Use short plain-text bullets (prefix with '- '). "
            "Never invent assets or numbers that are not in the context. "
            "No markdown tables, no headings. Keep the answer under 150 words."
        ),
        human=(
            "Telemetry context (JSON): {context_json}\n"
            "Recent conversation: {history_json}\n"
            "Operator question: {question}"
        ),
        variables={
            "context_json": json.dumps(context, ensure_ascii=True)[:12000],
            "history_json": json.dumps(safe_history, ensure_ascii=True)[:4000],
            "question": message,
        },
        temperature=0.35,
        max_tokens=600,
    )
    return answer or local_chat_answer(message, context)


def dynamic_suggestions(context: Dict[str, Any]) -> List[str]:
    """Context-aware follow-up prompts for the chat UI."""
    t = context_totals(context)
    suggestions: List[str] = []
    if t["critical_alerts"] > 0:
        suggestions.append(f"What is driving the {t['critical_alerts']} critical alerts?")
    if t["offline"] > 0:
        suggestions.append(f"How do I recover the {t['offline']} offline assets?")
    if t["warning"] > 0:
        suggestions.append(f"Which {t['warning']} warning assets should I fix first?")
    suggestions.append("Predict the next 24 hours of fleet risk.")
    suggestions.append("Summarise this shift in three bullet points.")
    return suggestions[:4]


# ─── Flask app ────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/health": {"origins": "*"}})


@app.get("/")
def welcome():
    return jsonify({
        "service": "PlantIQX AI Backend",
        "version": "2.0.0",
        "status": "running",
        "provider": AI_CONFIG["provider"],
        "model": AI_CONFIG["model"],
        "llmConfigured": AI_CONFIG["configured"],
        "endpoints": {
            "health": "GET /health",
            "summary": "POST /api/v1/ai/insights/summary",
            "anomalies": "POST /api/v1/ai/insights/anomalies",
            "rootCause": "POST /api/v1/ai/insights/root-cause",
            "chat": "POST /api/v1/ai/chat",
            "feedback": "POST /api/v1/ai/feedback",
        },
    })


@app.get("/health")
def health_check():
    return jsonify({
        "success": True,
        "service": "ai-backend",
        "llmConfigured": AI_CONFIG["configured"],
        "provider": AI_CONFIG["provider"],
        "model": AI_CONFIG["model"],
        "mode": "llm" if AI_CONFIG["configured"] else "fallback",
        "generatedAt": utc_now_iso(),
    })


@app.post("/api/v1/ai/insights/summary")
def ai_summary():
    started = time.time()
    payload = request.get_json(silent=True) or {}
    context = payload.get("context", {}) if isinstance(payload, dict) else {}
    focus = str(payload.get("focus", "iot-operations")) if isinstance(payload, dict) else "iot-operations"

    result = generate_summary_with_llm(context, focus)
    result["latencyMs"] = round((time.time() - started) * 1000)
    return jsonify({"success": True, "data": result})


@app.post("/api/v1/ai/insights/anomalies")
def ai_anomalies():
    started = time.time()
    payload = request.get_json(silent=True) or {}
    context = payload.get("context", {}) if isinstance(payload, dict) else {}

    result = generate_anomalies_with_llm(context)
    result["latencyMs"] = round((time.time() - started) * 1000)
    return jsonify({"success": True, "data": result})


@app.post("/api/v1/ai/insights/root-cause")
def ai_root_cause():
    started = time.time()
    payload = request.get_json(silent=True) or {}
    context = payload.get("context", {}) if isinstance(payload, dict) else {}
    subject = str(payload.get("subject", "")).strip() if isinstance(payload, dict) else ""

    result = generate_root_cause_with_llm(context, subject)
    result["latencyMs"] = round((time.time() - started) * 1000)
    return jsonify({"success": True, "data": result})


@app.post("/api/v1/ai/chat")
def ai_chat():
    started = time.time()
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    context = payload.get("context", {}) if isinstance(payload, dict) else {}
    history = payload.get("history", []) if isinstance(payload, dict) else []

    if not message:
        return jsonify({"success": False, "error": "message is required"}), 400

    answer = generate_chat_with_llm(message, context, history)
    return jsonify({
        "success": True,
        "data": {
            "answer": answer,
            "suggestions": dynamic_suggestions(context),
            "source": "llm" if get_llm() is not None else "fallback",
            "model": AI_CONFIG["model"] if AI_CONFIG["configured"] else "heuristic",
            "latencyMs": round((time.time() - started) * 1000),
            "generatedAt": utc_now_iso(),
        },
    })


@app.post("/api/v1/ai/feedback")
def ai_feedback():
    payload = request.get_json(silent=True) or {}
    vote = str(payload.get("vote", "")).lower()
    if vote not in {"up", "down"}:
        return jsonify({"success": False, "error": "vote must be 'up' or 'down'"}), 400

    # Feedback is logged for later quality review; storage is intentionally
    # lightweight (stdout) so the service stays dependency-free.
    app.logger.info(
        "AI feedback: vote=%s source=%s note=%s",
        vote, payload.get("source"), payload.get("note"),
    )
    return jsonify({
        "success": True,
        "data": {"accepted": True, "vote": vote, "generatedAt": utc_now_iso()},
    })


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", "5005"))
    print(f"[ai-backend] provider={AI_CONFIG['provider']} model={AI_CONFIG['model']} "
          f"configured={AI_CONFIG['configured']}")
    app.run(host="0.0.0.0", port=port,
            debug=os.getenv("FLASK_ENV", "development") == "development")
