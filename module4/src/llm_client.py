"""
llm_client.py
--------------
Wraps the Anthropic API for the NARRATIVE sections of the playbook (Threat
Summary, Root Cause explanation, Executive Summary) — the sections where
fluent prose genuinely helps and isn't operationally dangerous if the
wording isn't perfect. Structured/technical content (commands, rules, MITRE
IDs, CVEs) comes from threat_intel.py's curated templates instead — see that
module's docstring for why an LLM is the wrong tool for that part.

Two layers:
    call_llm(prompt, ...)        — low-level: one API call, one prompt in, text out.
    generate_narrative(ctx, section) — high-level: builds the right grounded
                                        prompt for "threat_summary" /
                                        "root_cause" / "executive_summary"
                                        from the event's actual detected data.

GATING: the caller (playbook_generator.py / main.py) is responsible for
enforcing "never run unless final_risk_probability > 70" — this module does
not duplicate that check, to avoid two sources of truth for the same rule
silently drifting apart. It DOES refuse to run without a valid API key,
loudly rather than silently.

API KEY: reads ANTHROPIC_API_KEY from the environment (never hardcoded).
If unset, call_llm() raises a clear EnvironmentError by default. For
development/testing without live API access or cost, pass mock_mode=True
(or set SOC_LLM_MOCK_MODE=1) to get a deterministic, clearly-labeled
placeholder response instead of a real model call — this is NOT a
substitute for a live key in production, and every mock response is
prefixed so it can never be mistaken for real model output downstream.
"""

import os

from utils import get_logger, require_env

logger = get_logger("llm_client")

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 800
MOCK_PREFIX = "[MOCK LLM OUTPUT — SOC_LLM_MOCK_MODE=1, NOT A REAL MODEL RESPONSE] "


def _mock_mode_enabled(explicit: bool = None) -> bool:
    if explicit is not None:
        return explicit
    return os.environ.get("SOC_LLM_MOCK_MODE", "").strip() in ("1", "true", "True")


def call_llm(prompt: str, system_prompt: str = None, max_tokens: int = MAX_TOKENS,
             model: str = DEFAULT_MODEL, mock_mode: bool = None) -> str:
    """
    Calls the Anthropic API. Raises EnvironmentError with a clear message if
    ANTHROPIC_API_KEY isn't set and mock_mode isn't enabled — never silently
    proceeds with a fake key or skips the call without telling the caller.
    """
    if _mock_mode_enabled(mock_mode):
        logger.warning("llm_client running in MOCK MODE — no real API call is being made.")
        return MOCK_PREFIX + f"(would have sent a {len(prompt)}-char prompt to {model})"

    api_key = require_env("ANTHROPIC_API_KEY", "calling the Anthropic API for playbook narrative generation")

    try:
        import anthropic
    except ImportError as e:
        raise ImportError(
            "The 'anthropic' package is required for live LLM calls. Install with: pip install anthropic"
        ) from e

    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}],
        )
        text_parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        return "".join(text_parts)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise RuntimeError(f"LLM call to {model} failed: {e}") from e


_SECTION_INSTRUCTIONS = {
    "threat_summary": (
        "Write a concise (3-5 sentence) Threat Summary for a SOC analyst, in plain "
        "professional language. State what was detected, how confident the detection is, "
        "and why it matters. Do not invent facts beyond the data provided below."
    ),
    "root_cause": (
        "Write a concise (3-5 sentence) Root Cause analysis explaining, based ONLY on the "
        "evidence provided (the SHAP feature contributions and event details), which "
        "specific signals most likely drove this detection and what that suggests about "
        "the underlying attack mechanism. Be explicit that this is a probable inference "
        "from the detection model's evidence, not a confirmed forensic finding."
    ),
    "executive_summary": (
        "Write a 3-4 sentence Executive Summary suitable for a non-technical leadership "
        "audience: what happened, potential business impact, and what the security team "
        "is doing about it. Avoid jargon."
    ),
}
_SYSTEM_PROMPT = (
    "You are a SOC (Security Operations Center) analyst assistant. Ground every statement "
    "strictly in the event data provided. Never invent IP addresses, CVEs, usernames, or "
    "facts not present in the input. If evidence is ambiguous, say so rather than guessing."
)


def generate_narrative(prompt_context: dict, section: str, mock_mode: bool = None) -> str:
    """
    Generates one narrative section of the playbook.

    Args:
        prompt_context: dict with the event's key facts (risk scores, channel,
            top SHAP features, evidence text, etc.) — grounds the model in
            real detected data rather than letting it free-associate.
        section: "threat_summary" | "root_cause" | "executive_summary"
        mock_mode: passthrough to call_llm() for testing without a live key.

    Returns:
        Generated text.
    """
    instruction = _SECTION_INSTRUCTIONS.get(section, _SECTION_INSTRUCTIONS["threat_summary"])

    context_text = (
        f"Event data:\n"
        f"- risk_category: {prompt_context['risk_category']}\n"
        f"- final_risk_probability: {prompt_context['final_risk_probability']:.3f}\n"
        f"- anomaly_score: {prompt_context['anomaly_score']:.1f} / 100\n"
        f"- phishing_probability: {prompt_context['phishing_probability']:.3f}\n"
        f"- channel: {prompt_context['channel']}\n"
        f"- threat_category: {prompt_context['threat_category']}\n"
        f"- source_ip: {prompt_context['source_ip']}\n"
        f"- top_contributing_features (SHAP, ranked): {prompt_context['top_features']}\n"
        f"- evidence_text (raw content sample): {str(prompt_context.get('evidence_text', 'N/A'))[:300]}\n"
    )

    return call_llm(
        prompt=f"{instruction}\n\n{context_text}",
        system_prompt=_SYSTEM_PROMPT,
        mock_mode=mock_mode,
    )


if __name__ == "__main__":
    # Smoke test in mock mode — doesn't require a real API key, verifies the
    # gating/plumbing logic without incurring cost or requiring credentials.
    result = call_llm("Summarize this test threat event.", mock_mode=True)
    logger.info(f"Mock call_llm result: {result}")
    assert result.startswith(MOCK_PREFIX)

    sample_context = {
        "risk_category": "Critical", "final_risk_probability": 0.92, "anomaly_score": 88.0,
        "phishing_probability": 0.95, "channel": "email", "threat_category": "Phishing (Email)",
        "source_ip": "203.0.113.5", "top_features": ["burst_score", "failed_connection_rate"],
        "evidence_text": "URGENT verify your account now",
    }
    for section in ("threat_summary", "root_cause", "executive_summary"):
        text = generate_narrative(sample_context, section, mock_mode=True)
        assert text.startswith(MOCK_PREFIX)
        logger.info(f"[{section}] mock output OK")

    logger.info("llm_client.py self-tests passed (mock mode).")
