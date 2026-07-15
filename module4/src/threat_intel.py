"""
threat_intel.py
-----------------
Curated, deterministic threat-intelligence knowledge base used by
playbook_generator.py to build the technical sections of a response
playbook (MITRE mapping, CVE suggestions, YARA/Snort/Sigma rules, Linux/
Windows commands, firewall rules).

WHY THIS IS TEMPLATE-DRIVEN, NOT LLM-GENERATED:
Detection rules and remediation commands are exactly the wrong place to let
an LLM freely improvise — a subtly wrong YARA/Snort/Sigma rule syntax error,
or a hallucinated CVE ID, is actively worse than no rule at all in a SOC
context (false confidence, wasted analyst time, or a rule that silently
doesn't compile/fire). This module maps observed evidence (channel, top
SHAP features, actual IOCs from the event) to a small, curated set of
verified, real content instead. The LLM (see llm_client.py) is used only
for the narrative sections (Threat Summary, Root Cause explanation,
Executive Summary) where fluent prose genuinely helps and a stylistic
imperfection isn't operationally dangerous.

CVE ACCURACY: every CVE ID referenced below was verified against public
vulnerability databases / vendor advisories before being included here (see
Module 4 README for the verification pass). They are explicitly framed as
"illustrative examples commonly associated with this attack pattern" — NOT
a confirmed match for the specific flagged event, since we have no software/
version telemetry to confirm an actual match. Presenting them as confirmed
findings would be irresponsible; presenting them as category-relevant
reference points, with that caveat attached, is standard SOC practice.
"""

from utils import get_logger

logger = get_logger("threat_intel")

CHANNEL_CATEGORY_MAP = {
    "email": "Phishing (Email)",
    "sms": "Phishing (SMS / Smishing)",
    "url": "Phishing (Malicious URL)",
    "login_attempt": "Credential-Based Attack (Brute Force / Account Takeover)",
    "api_payload": "Web/API Exploitation (Injection or Abuse)",
}

# -----------------------------------------------------------------------------
# MITRE ATT&CK mapping (technique IDs verified against attack.mitre.org)
# -----------------------------------------------------------------------------
MITRE_MAP = {
    "Phishing (Email)": [
        {"tactic": "Initial Access", "technique_id": "T1566.001", "technique_name": "Phishing: Spearphishing Attachment"},
        {"tactic": "Initial Access", "technique_id": "T1566.002", "technique_name": "Phishing: Spearphishing Link"},
        {"tactic": "Credential Access", "technique_id": "T1656", "technique_name": "Impersonation"},
    ],
    "Phishing (SMS / Smishing)": [
        {"tactic": "Initial Access", "technique_id": "T1660", "technique_name": "Phishing (via SMS / smishing)"},
        {"tactic": "Credential Access", "technique_id": "T1656", "technique_name": "Impersonation"},
    ],
    "Phishing (Malicious URL)": [
        {"tactic": "Initial Access", "technique_id": "T1566.002", "technique_name": "Phishing: Spearphishing Link"},
        {"tactic": "Resource Development", "technique_id": "T1583.001", "technique_name": "Acquire Infrastructure: Domains"},
    ],
    "Credential-Based Attack (Brute Force / Account Takeover)": [
        {"tactic": "Credential Access", "technique_id": "T1110", "technique_name": "Brute Force"},
        {"tactic": "Credential Access", "technique_id": "T1110.003", "technique_name": "Brute Force: Password Spraying"},
        {"tactic": "Defense Evasion / Persistence", "technique_id": "T1078", "technique_name": "Valid Accounts"},
        {"tactic": "Command and Control", "technique_id": "T1090.003", "technique_name": "Proxy: Multi-hop Proxy (Tor)"},
    ],
    "Web/API Exploitation (Injection or Abuse)": [
        {"tactic": "Initial Access", "technique_id": "T1190", "technique_name": "Exploit Public-Facing Application"},
        {"tactic": "Execution", "technique_id": "T1059", "technique_name": "Command and Scripting Interpreter"},
        {"tactic": "Collection", "technique_id": "T1005", "technique_name": "Data from Local System"},
    ],
    "Network-Layer Anomaly (Scan / DoS-like Behavior)": [
        {"tactic": "Reconnaissance", "technique_id": "T1595.001", "technique_name": "Active Scanning: Scanning IP Blocks"},
        {"tactic": "Impact", "technique_id": "T1498", "technique_name": "Network Denial of Service"},
        {"tactic": "Command and Control", "technique_id": "T1071", "technique_name": "Application Layer Protocol"},
    ],
}

# -----------------------------------------------------------------------------
# CVE reference table (verified real CVE IDs — see module docstring)
# -----------------------------------------------------------------------------
CVE_MAP = {
    "Phishing (Email)": [
        {"cve_id": "CVE-2023-23397", "name": "Microsoft Outlook Elevation of Privilege (NTLM hash theft via crafted email)",
         "note": "Relevant if the organization runs Outlook for Windows and the email contains reminder/UNC-path artifacts."},
    ],
    "Credential-Based Attack (Brute Force / Account Takeover)": [
        {"cve_id": "CVE-2020-1472", "name": "Zerologon (Netlogon privilege escalation)",
         "note": "Relevant if brute-forced/compromised credentials could be used to reach an unpatched Domain Controller."},
    ],
    "Web/API Exploitation (Injection or Abuse)": [
        {"cve_id": "CVE-2021-44228", "name": "Log4Shell (Apache Log4j2 remote code execution)",
         "note": "Relevant if the targeted API backend logs user-controlled input via a vulnerable Log4j2 version."},
        {"cve_id": "CVE-2017-5638", "name": "Apache Struts 2 OGNL injection RCE",
         "note": "Relevant if the API/web backend runs an unpatched Apache Struts 2 (Content-Type header RCE)."},
    ],
    "Network-Layer Anomaly (Scan / DoS-like Behavior)": [
        {"cve_id": "CVE-2021-34527", "name": "PrintNightmare (Windows Print Spooler RCE)",
         "note": "Relevant if lateral movement follows a network scan and Windows Print Spooler is exposed/unpatched."},
    ],
}
CVE_DISCLAIMER = (
    "Illustrative CVEs commonly associated with this attack category — NOT a confirmed "
    "match for this specific event. This platform has no software/version inventory for "
    "the affected host; confirm applicability against actual asset data before acting on any of these."
)


def classify_threat_category(channel: str, important_features: list) -> str:
    """
    Maps the flagged event's channel + top SHAP features to one of the
    curated threat categories above.
    """
    if channel == "login_attempt":
        return "Credential-Based Attack (Brute Force / Account Takeover)"
    if channel == "api_payload":
        return "Web/API Exploitation (Injection or Abuse)"
    if channel in ("email", "sms", "url"):
        return CHANNEL_CATEGORY_MAP[channel]
    return "Network-Layer Anomaly (Scan / DoS-like Behavior)"


def get_mitre_mapping(category: str) -> list:
    return MITRE_MAP.get(category, MITRE_MAP["Network-Layer Anomaly (Scan / DoS-like Behavior)"])


def get_cve_suggestions(category: str) -> dict:
    return {
        "disclaimer": CVE_DISCLAIMER,
        "candidates": CVE_MAP.get(category, []),
    }


# -----------------------------------------------------------------------------
# Immediate actions / commands, templated per category
# -----------------------------------------------------------------------------
def get_immediate_actions(category: str, source_ip: str) -> list:
    common = [
        f"Isolate/quarantine traffic from source IP {source_ip} pending investigation.",
        "Preserve logs and relevant packet captures for the affected window before any remediation that could overwrite them.",
    ]
    specific = {
        "Phishing (Email)": [
            "Quarantine the reported message across all mailboxes (search-and-purge by sender/subject/hash).",
            "Block the sender domain and any embedded URLs at the email gateway.",
            "Notify the targeted user(s) and any recipients who interacted with the message.",
            "Force a password reset + re-authentication (all sessions) for any user who submitted credentials.",
        ],
        "Phishing (SMS / Smishing)": [
            "Block the originating number/short-code at the carrier/MDM level if managed devices are involved.",
            "Notify the targeted user and advise against clicking the linked URL if not already done.",
        ],
        "Phishing (Malicious URL)": [
            "Block the URL/domain at the web proxy and DNS resolver.",
            "Check proxy/DNS logs for any other users who resolved or visited the domain.",
        ],
        "Credential-Based Attack (Brute Force / Account Takeover)": [
            "Lock or force-reset the targeted account(s); invalidate active sessions/tokens.",
            "Enforce MFA re-enrollment for the affected account.",
            "Block the source IP at the perimeter firewall / WAF.",
            "Review authentication logs for the account over the preceding 30 days for earlier compromise indicators.",
        ],
        "Web/API Exploitation (Injection or Abuse)": [
            "Block the source IP at the WAF; add a temporary rate-limit rule on the targeted endpoint.",
            "Review application logs for successful responses (200/302) to the same payload pattern from other sources.",
            "Snapshot the affected service/container for forensic review before any restart.",
        ],
        "Network-Layer Anomaly (Scan / DoS-like Behavior)": [
            "Rate-limit or null-route the source IP at the edge router/firewall.",
            "Enable SYN cookies / connection-rate protections on the targeted host if not already active.",
        ],
    }
    return common + specific.get(category, [])


def get_linux_commands(category: str, source_ip: str) -> list:
    common = [
        f"# Block the source IP at the host firewall\nsudo iptables -A INPUT -s {source_ip} -j DROP",
        f"# Confirm the block is in place\nsudo iptables -L INPUT -v -n | grep {source_ip}",
        "# Capture current network connections for evidence\nss -tunap > /tmp/soc_evidence_connections_$(date +%s).txt",
    ]
    specific = {
        "Credential-Based Attack (Brute Force / Account Takeover)": [
            f"# Review recent auth failures from this source\nsudo grep '{source_ip}' /var/log/auth.log | tail -100",
            "# Force logout of all active sessions for a compromised local user (replace USERNAME)\nsudo pkill -KILL -u USERNAME",
        ],
        "Web/API Exploitation (Injection or Abuse)": [
            f"# Search web server logs for the same payload pattern\nsudo grep -E \"(UNION|OR '1'='1'|<script>)\" /var/log/nginx/access.log | grep '{source_ip}'",
            "# Snapshot the container/service for forensic review\ndocker commit <container_id> forensic_snapshot_$(date +%s)",
        ],
        "Network-Layer Anomaly (Scan / DoS-like Behavior)": [
            f"# Rate-limit the source IP instead of a hard block, if it may include legitimate mixed traffic\nsudo iptables -A INPUT -s {source_ip} -m limit --limit 10/minute -j ACCEPT",
        ],
    }
    return common + specific.get(category, [])


def get_windows_commands(category: str, source_ip: str) -> list:
    common = [
        f'# Block the source IP via Windows Firewall\nNew-NetFirewallRule -DisplayName "SOC-Block-{source_ip}" -Direction Inbound -RemoteAddress {source_ip} -Action Block',
        "# Capture current network connections for evidence\nGet-NetTCPConnection | Export-Csv -Path C:\\soc_evidence\\connections.csv",
    ]
    specific = {
        "Credential-Based Attack (Brute Force / Account Takeover)": [
            f"# Review recent failed logons from this source (Event ID 4625)\nGet-WinEvent -FilterHashtable @{{LogName='Security'; Id=4625}} | Where-Object {{$_.Message -match '{source_ip}'}}",
            "# Force sign-out and disable a compromised account (replace USERNAME)\nDisable-ADAccount -Identity USERNAME",
        ],
        "Web/API Exploitation (Injection or Abuse)": [
            f"# Search IIS logs for the same payload pattern\nSelect-String -Path C:\\inetpub\\logs\\LogFiles\\*.log -Pattern '{source_ip}'",
        ],
    }
    return common + specific.get(category, [])


def get_firewall_rules(category: str, source_ip: str) -> list:
    return [
        f"# Generic (iptables) — block\niptables -A INPUT -s {source_ip} -j DROP",
        f"# Generic (nftables) — block\nnft add rule inet filter input ip saddr {source_ip} drop",
        f"# Cisco ASA — block\naccess-list SOC_BLOCK deny ip host {source_ip} any",
        f"# pfSense/OPNsense (alias-based) — add {source_ip} to a 'SOC_Blocklist' alias and apply the existing deny rule referencing it.",
    ]


# -----------------------------------------------------------------------------
# Detection rules (YARA / Snort / Sigma), templated from the ACTUAL observed
# indicators in this event — not generic boilerplate.
# -----------------------------------------------------------------------------
def get_yara_rule(rule_name: str, indicators: list, source_ip: str) -> str:
    import re
    # YARA rule identifiers must be alphanumeric + underscore only.
    # Sanitize defensively here rather than trusting the caller — a rule
    # name built from a threat category ("Credential-Based...") or an IP
    # address contains hyphens/dots that would make the generated rule
    # fail to compile.
    safe_name = re.sub(r"[^A-Za-z0-9_]", "_", rule_name)
    if safe_name and safe_name[0].isdigit():
        safe_name = "_" + safe_name

    string_defs = "\n".join(f'        $ioc_{i} = "{ind}" nocase' for i, ind in enumerate(indicators) if ind)
    if not string_defs:
        string_defs = '        $placeholder = "no textual IOC extracted for this event"'
    return f"""rule SOC_{safe_name}
{{
    meta:
        description = "Auto-generated by SOC Assistant (Module 4) from a flagged high-risk event"
        source_ip = "{source_ip}"
        generated = "template-driven, not LLM-authored — review before deployment"

    strings:
{string_defs}

    condition:
        any of ($ioc_*)
}}"""


def get_snort_rule(rule_sid: int, category: str, source_ip: str, indicators: list) -> str:
    content_clauses = "".join(f'content:"{ind}"; nocase; ' for ind in indicators if ind)
    msg = category.replace('"', "'")
    return (
        f'alert tcp {source_ip} any -> $HOME_NET any '
        f'(msg:"SOC: {msg} from {source_ip}"; {content_clauses}'
        f'sid:{rule_sid}; rev:1; classtype:trojan-activity;)'
    )


def get_sigma_rule(category: str, source_ip: str, indicators: list) -> str:
    indicator_yaml = "\n".join(f"        - '{ind}'" for ind in indicators if ind) or "        - 'PLACEHOLDER'"
    return f"""title: SOC Auto-Generated Detection — {category}
status: experimental
description: Generated by SOC Assistant (Module 4) from a flagged high-risk event. Review before deployment.
logsource:
    category: network
    product: generic
detection:
    selection:
        src_ip: '{source_ip}'
        payload|contains:
{indicator_yaml}
    condition: selection
level: high
tags:
    - soc.autogenerated
"""


if __name__ == "__main__":
    for category in MITRE_MAP:
        mitre = get_mitre_mapping(category)
        cve = get_cve_suggestions(category)
        assert len(mitre) > 0, f"No MITRE mapping for {category}"
        logger.info(f"{category}: {len(mitre)} MITRE technique(s), {len(cve['candidates'])} CVE candidate(s)")

    sample_yara = get_yara_rule("test_rule", ["evil.tk", "203.0.113.5"], "203.0.113.5")
    sample_snort = get_snort_rule(1000001, "Phishing (Email)", "203.0.113.5", ["evil.tk"])
    sample_sigma = get_sigma_rule("Phishing (Email)", "203.0.113.5", ["evil.tk"])
    assert "rule SOC_test_rule" in sample_yara
    assert "alert tcp" in sample_snort
    assert "title:" in sample_sigma
    logger.info("Sample YARA/Snort/Sigma rules generated successfully.")

    cat = classify_threat_category("login_attempt", ["failed_connection_rate"])
    assert cat == "Credential-Based Attack (Brute Force / Account Takeover)"
    logger.info("threat_intel.py self-test passed.")
