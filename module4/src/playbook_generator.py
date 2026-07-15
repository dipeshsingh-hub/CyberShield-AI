"""
playbook_generator.py
-----------------------
Builds the full incident-response playbook for a high-risk event.

PUBLIC CONTRACT (do not rename): generate_response_playbook(event, mock_llm=None)

GATING (enforced here, the single source of truth for this rule): the LLM
is never called unless the caller has already confirmed
final_risk_probability > 70 (on the 0-100 scale) / > 0.70 (0-1 scale) — see
the explicit check at the top of generate_response_playbook(). main.py also
checks this before even calling this function, but the check is repeated
here defensively: this function must never be the reason the rule gets
violated just because some future caller forgets to check first.

Sections are split into two provenance classes, and every playbook says so
explicitly:
    - LLM-narrated (Threat Summary, Root Cause, Executive Summary): fluent
      prose grounded in the event's real data, via llm_client.py.
    - Template-derived (everything else — Evidence, Confidence Score,
      Affected Systems, Immediate Actions, Linux/Windows Commands, Firewall
      Rules, YARA/Snort/Sigma, MITRE mapping, CVE Suggestions, Recovery
      Steps, Post-Incident Checklist): deterministic, curated content from
      threat_intel.py. See that module's docstring for why detection rules
      and remediation commands are deliberately NOT left to an LLM.
"""

import os

from utils import get_logger
import threat_intel as ti
from llm_client import generate_narrative

logger = get_logger("playbook_generator")

RISK_THRESHOLD_0_100 = 70.0


def _extract_indicators(event: dict) -> list:
    """Pulls concrete, observable IOCs out of the event for use in generated detection rules."""
    indicators = []
    text = event.get("text") or event.get("evidence_text")
    if text:
        # Pull out the most suspicious-looking token(s) rather than the whole
        # message — keeps generated rules tight rather than matching on
        # generic prose that would false-positive constantly.
        import re
        urls = re.findall(r"https?://[^\s]+|\b[\w.-]+\.(?:tk|ml|ga|cf|xyz|info)\b", text)
        indicators.extend(urls[:2])
        injection_patterns = re.findall(r"(?:' OR |UNION SELECT|<script>|DROP TABLE|\.\./\.\.)", text)
        indicators.extend(injection_patterns[:2])
    if event.get("source_ip"):
        indicators.append(event["source_ip"])
    return [i for i in indicators if i][:5]


def _confidence_score(event: dict) -> dict:
    """
    Confidence isn't just final_risk_probability restated — it reflects
    AGREEMENT between the two independent upstream signals (network
    behavior vs. content), since two independent detectors agreeing is
    stronger evidence than either alone (this is the same conditional-
    independence logic Module 2's Bayesian layer is built on).
    """
    anomaly_norm = event["anomaly_score"] / 100.0
    phishing = event["phishing_probability"]
    agreement = 1.0 - abs(anomaly_norm - phishing)  # 1.0 = both signals fully agree
    confidence_pct = round(100 * (0.6 * event["final_risk_probability"] + 0.4 * agreement), 1)
    return {
        "confidence_percent": confidence_pct,
        "basis": (
            f"final_risk_probability={event['final_risk_probability']:.2f} weighted with "
            f"cross-signal agreement={agreement:.2f} (anomaly_score and phishing_probability "
            f"{'agree closely' if agreement > 0.7 else 'partially agree' if agreement > 0.4 else 'diverge'})"
        ),
    }


def generate_response_playbook(event: dict, mock_llm: bool = None) -> dict:
    """
    PUBLIC CONTRACT (do not rename): generate_response_playbook(event, mock_llm=None)

    Args:
        event: dict containing at minimum:
            anomaly_score (0-100), phishing_probability (0-1),
            final_risk_probability (0-1), risk_level, important_features,
            feature_contributions, source_ip, channel, text (raw content).
        mock_llm: passthrough to llm_client for testing without a live API
            key. None defers to the SOC_LLM_MOCK_MODE env var.

    Returns:
        dict with every section listed in the spec: threat_summary,
        evidence, confidence_score, root_cause, affected_systems,
        immediate_actions, linux_commands, windows_commands, firewall_rules,
        yara_rule, snort_rule, sigma_rule, mitre_attack_mapping,
        cve_suggestions, recovery_steps, post_incident_checklist,
        executive_summary.

    Raises:
        ValueError if final_risk_probability <= 0.70 — this function REFUSES
        to run below threshold, independent of whatever check main.py did,
        so it can never be misused as a bypass.
    """
    frp = event["final_risk_probability"]
    frp_0_100 = frp * 100 if frp <= 1.0 else frp
    if frp_0_100 <= RISK_THRESHOLD_0_100:
        raise ValueError(
            f"generate_response_playbook() refuses to run: final_risk_probability "
            f"({frp_0_100:.1f}) is not > {RISK_THRESHOLD_0_100}. This is a hard gate, "
            f"not a suggestion — see module docstring."
        )

    logger.info(f"Generating playbook for event (source_ip={event.get('source_ip')}, "
                f"final_risk_probability={frp_0_100:.1f})")

    channel = event.get("channel", "unknown")
    important_features = event.get("important_features", [])
    source_ip = event.get("source_ip", "UNKNOWN_IP")
    category = ti.classify_threat_category(channel, important_features)
    indicators = _extract_indicators(event)
    mitre = ti.get_mitre_mapping(category)
    cve = ti.get_cve_suggestions(category)

    narrative_ctx = {
        "risk_category": event.get("risk_level", "Critical"),
        "final_risk_probability": frp if frp <= 1.0 else frp / 100.0,
        "anomaly_score": event["anomaly_score"],
        "phishing_probability": event["phishing_probability"],
        "channel": channel,
        "threat_category": category,
        "source_ip": source_ip,
        "top_features": important_features[:5],
        "evidence_text": event.get("text", ""),
    }

    try:
        threat_summary = generate_narrative(narrative_ctx, "threat_summary", mock_llm)
        root_cause = generate_narrative(narrative_ctx, "root_cause", mock_llm)
        executive_summary = generate_narrative(narrative_ctx, "executive_summary", mock_llm)
    except (EnvironmentError, RuntimeError) as e:
        # Narrative generation failing should not silently produce a broken
        # playbook with missing sections — surface it clearly, but let the
        # deterministic sections (the operationally critical ones: commands,
        # rules, MITRE mapping) still be returned rather than losing
        # everything because the LLM step failed.
        logger.error(f"Narrative generation failed: {e}. Deterministic sections still included.")
        error_note = f"[Narrative generation unavailable: {e}]"
        threat_summary = root_cause = executive_summary = error_note

    playbook = {
        "threat_summary": threat_summary,
        "evidence": {
            "feature_contributions": event.get("feature_contributions", {}),
            "top_features": important_features[:5],
            "extracted_indicators": indicators,
            "raw_content_sample": (event.get("text") or "")[:300],
        },
        "confidence_score": _confidence_score(event),
        "root_cause": root_cause,
        "affected_systems": {
            "source_ip": source_ip,
            "destination_ip": event.get("destination_ip", "UNKNOWN"),
            "channel": channel,
            "protocol": event.get("protocol", "UNKNOWN"),
        },
        "immediate_actions": ti.get_immediate_actions(category, source_ip),
        "linux_commands": ti.get_linux_commands(category, source_ip),
        "windows_commands": ti.get_windows_commands(category, source_ip),
        "firewall_rules": ti.get_firewall_rules(category, source_ip),
        "yara_rule": ti.get_yara_rule(category.split(" ")[0] + "_" + source_ip.replace(".", "_"), indicators, source_ip),
        "snort_rule": ti.get_snort_rule(1000001, category, source_ip, indicators),
        "sigma_rule": ti.get_sigma_rule(category, source_ip, indicators),
        "mitre_attack_mapping": mitre,
        "cve_suggestions": cve,
        "recovery_steps": [
            "Confirm the source of compromise is fully contained (no active sessions/connections remain).",
            "Restore affected accounts/services from known-good state; rotate any credentials that may have been exposed.",
            "Re-scan previously affected hosts for persistence mechanisms before returning them to production.",
            "Validate that the deployed firewall/detection rules from this playbook are active and correctly matching.",
            "Monitor the source IP/domain/account for 14 days for renewed activity before closing the incident.",
        ],
        "post_incident_checklist": [
            "[ ] Root cause confirmed and documented (not just inferred from SHAP evidence)",
            "[ ] All immediate actions completed and verified",
            "[ ] Detection rules (YARA/Snort/Sigma) deployed to production and validated against a known-good test case",
            "[ ] Affected users/systems notified per incident communication policy",
            "[ ] Playbook and evidence archived for compliance / audit",
            "[ ] Lessons-learned review scheduled with the team",
            "[ ] Any newly-identified detection gaps fed back into Module 1/Module 2 training data",
        ],
        "executive_summary": executive_summary,
        "_metadata": {
            "threat_category": category,
            "generated_by": "Module 4 SOC Assistant — generate_response_playbook()",
            "llm_sections": ["threat_summary", "root_cause", "executive_summary"],
            "template_sections": [
                "evidence", "confidence_score", "affected_systems", "immediate_actions",
                "linux_commands", "windows_commands", "firewall_rules", "yara_rule",
                "snort_rule", "sigma_rule", "mitre_attack_mapping", "cve_suggestions",
                "recovery_steps", "post_incident_checklist",
            ],
        },
    }
    logger.info(f"Playbook generated: category={category}, {len(mitre)} MITRE techniques, "
                f"{len(cve['candidates'])} CVE candidates.")
    return playbook


if __name__ == "__main__":
    sample_event = {
        "anomaly_score": 88.0,
        "phishing_probability": 0.95,
        "final_risk_probability": 0.92,
        "risk_level": "Critical",
        "important_features": ["burst_score", "failed_connection_rate", "connection_density"],
        "feature_contributions": {"burst_score": 1.8, "failed_connection_rate": 0.9, "connection_density": 0.5},
        "source_ip": "203.0.113.9",
        "destination_ip": "198.51.100.4",
        "channel": "login_attempt",
        "protocol": "TCP",
        "text": "username=user4821 login_country=RU failed_attempts=22 country_mismatch=True new_device=True",
    }

    logger.info("Test 1: below-threshold event should raise ValueError")
    try:
        generate_response_playbook({**sample_event, "final_risk_probability": 0.3}, mock_llm=True)
        raise AssertionError("Should have raised ValueError for low-risk event!")
    except ValueError as e:
        logger.info(f"Correctly refused: {e}")

    logger.info("Test 2: above-threshold event should generate a full playbook (mock LLM)")
    playbook = generate_response_playbook(sample_event, mock_llm=True)
    required_keys = [
        "threat_summary", "evidence", "confidence_score", "root_cause", "affected_systems",
        "immediate_actions", "linux_commands", "windows_commands", "firewall_rules",
        "yara_rule", "snort_rule", "sigma_rule", "mitre_attack_mapping", "cve_suggestions",
        "recovery_steps", "post_incident_checklist", "executive_summary",
    ]
    missing = [k for k in required_keys if k not in playbook]
    assert not missing, f"Missing required playbook sections: {missing}"
    logger.info(f"All {len(required_keys)} required playbook sections present.")
    logger.info(f"Sample YARA rule:\n{playbook['yara_rule']}")
    logger.info(f"Confidence score: {playbook['confidence_score']}")
    logger.info("playbook_generator.py self-tests passed.")
