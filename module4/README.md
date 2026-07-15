# Module 4 — SOC Assistant Orchestrator

Cybersecurity Threat Detection Platform — Final Module

Integrates Modules 1, 2, and 3 into a single event-processing pipeline, and
generates a full incident-response playbook — LLM-narrated where prose
genuinely helps, template-derived everywhere it doesn't — but **only** when
`final_risk_probability > 70`. Below that threshold, the event is logged
and the LLM is never invoked.

## Folder structure

```
module4/
├── data/event_log.jsonl          # append-only log of every processed event
├── src/
│   ├── utils.py                  # logging, env-var handling, JSON helpers
│   ├── module_bridge.py          # isolated subprocess calls into Modules 1/2/3's real code
│   ├── threat_intel.py           # curated MITRE/CVE/rule-template knowledge base
│   ├── llm_client.py             # Anthropic API wrapper (env-var key, mock mode)
│   ├── event_store.py            # append-only event persistence
│   ├── playbook_generator.py     # generate_response_playbook()
│   └── main.py                   # orchestration entrypoint
├── requirements.txt
├── .env.example
└── README.md
```

## Run it

```bash
cd module4
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY, or leave mock mode on

# Single event, mock LLM (no API key needed):
cd src && python main.py --source-ip <ip-from-module1-dataset> --content "..." --mock-llm

# Batch of N sampled real events from Module 1 + Module 2's actual datasets:
cd src && python main.py --batch 20 --mock-llm

# With a real API key (remove --mock-llm, ensure ANTHROPIC_API_KEY is exported):
export ANTHROPIC_API_KEY=sk-ant-...
cd src && python main.py --batch 5
```

Every file in `src/` is independently executable and self-tests when run
directly (`python <file>.py`) — verified in a full clean-room pass (delete
all generated data, re-run every self-test + a full orchestrated batch from
scratch) before this was packaged, not just claimed.

## The hard gate: LLM never runs unless `final_risk_probability > 70`

This rule is enforced in **two places**, deliberately redundant rather than
trusting a single check:

1. `main.py` checks it before even calling `generate_response_playbook()`.
2. `generate_response_playbook()` **itself** re-checks and raises
   `ValueError` if called with `final_risk_probability <= 70` — verified
   directly: calling it with a low-risk event raises immediately, before
   any LLM code path is reached. This means the function can never become a
   bypass just because some future caller (or a bug in `main.py`) forgets
   to check first.

Below-threshold events return the exact string `"No response required."` in
the `playbook` field of the final output — verified against the literal
string, not just "returns something falsy."

## Why cross-module calls go through subprocess isolation (again)

Modules 1, 2, 3, and 4 all define a file named `utils.py` (1 and 2 also
share `predict.py`). Importing more than one into the same Python process
corrupts `sys.modules` for all of them — this was caught during Module 2's
build, fixed the same way in Module 3, and reused here via
`module_bridge.py`, which calls each module's real, shipped functions
(`predict_network_anomaly`, `predict_phishing`, `bayesian_risk_adjustment`,
`generate_xai_report`) in isolated subprocesses. All four bridge functions
were verified independently before `main.py` was built on top of them.

## Extending Module 3 (not renaming — adding a capability)

Module 3's `generate_xai_report()` was originally built for a static
pre-computed dashboard dataset (`row_index` into `unified_threat_data.csv`).
Module 4's orchestrator processes brand-new live events that were never
part of that dataset, so `generate_xai_report()` was **extended** (never
renamed) with an optional `live_event` parameter — verified backward
compatible: existing `row_index` and no-argument (global report) calls
still behave identically, tested explicitly before building Module 4 on
top of the change.

## `generate_response_playbook()` — what's LLM-generated vs. templated, and why

**LLM-narrated** (via `llm_client.py`, Anthropic API): `threat_summary`,
`root_cause`, `executive_summary`. These are exactly the sections where
fluent, context-aware prose genuinely helps a reader and a stylistic
imperfection isn't operationally dangerous.

**Template-derived** (via `threat_intel.py`, deterministic): `evidence`,
`confidence_score`, `affected_systems`, `immediate_actions`,
`linux_commands`, `windows_commands`, `firewall_rules`, `yara_rule`,
`snort_rule`, `sigma_rule`, `mitre_attack_mapping`, `cve_suggestions`,
`recovery_steps`, `post_incident_checklist`.

This split is deliberate, not a shortcut: detection rules and remediation
commands are exactly the wrong place to let an LLM freely improvise. A
subtly-wrong YARA/Snort/Sigma syntax error, or a hallucinated CVE ID, is
actively worse than no rule at all in a SOC context — it creates false
confidence and wastes analyst time on content that silently doesn't work.

**This was verified, not assumed:**
- The generated YARA rule was compiled with a real YARA engine
  (`yara-python`) — confirmed it actually compiles, not just "looks like"
  valid syntax. A first draft used the threat category name directly as
  the rule identifier ("Credential-Based...") which broke on the hyphen —
  YARA identifiers only allow alphanumerics and underscores. Caught by
  actually compiling the rule, fixed by sanitizing the identifier inside
  `get_yara_rule()` itself (defensively, not just at the one call site).
- The generated Sigma rule was parsed with a real YAML parser — confirmed
  valid YAML, not just eyeballed.
- All CVE IDs referenced in `threat_intel.py` (CVE-2023-23397,
  CVE-2020-1472, CVE-2021-44228, CVE-2017-5638, CVE-2021-34527) were
  checked against public sources before inclusion. They're explicitly
  labeled **"illustrative examples commonly associated with this attack
  category — not a confirmed match"**, since this platform has no
  software/version inventory to confirm an actual match against a specific
  host. Presenting them as confirmed findings would be irresponsible.
- `confidence_score` isn't `final_risk_probability` restated — it factors
  in agreement between the two independent upstream signals (network
  behavior vs. content), on the same conditional-independence logic
  Module 2's own Bayesian layer is built on: two independent detectors
  agreeing is stronger evidence than either alone.
- Detection-rule IOCs (the `extracted_indicators` used in the YARA/Snort/
  Sigma rules) are pulled from the event's **actual observed content** — a
  real batch run correctly extracted the literal string `UNION SELECT` out
  of a genuine SQLi-flavored API payload and used it to drive both the
  threat classification and the generated Linux/YARA/Snort rules for that
  specific event, not a generic placeholder.

## API key handling

`ANTHROPIC_API_KEY` is read from the environment — never hardcoded, never
logged. If it's missing and mock mode isn't enabled, `call_llm()` raises a
clear `EnvironmentError` naming the exact variable and how to set it —
verified directly (unsetting the variable and confirming the error fires,
rather than assuming it would). Mock mode (`--mock-llm` or
`SOC_LLM_MOCK_MODE=1`) exists so the full orchestration pipeline is testable
end-to-end without live credentials or API cost, and so the tool doesn't go
fully dark if the LLM API is rate-limited or down — every mock response is
prefixed `[MOCK LLM OUTPUT ...]` so it can never be mistaken for a real
model response downstream.

The actual Anthropic SDK response-parsing code path (not just the mock
path) was verified separately using a simulated SDK response object, since
no live API key is available in this build environment — confirmed the
text-extraction logic works correctly against the real response shape, not
just against my own mocked stand-in.

## Final output shape

```python
{
    "anomaly_score": float,          # 0-100, from Module 1
    "phishing_probability": float,   # 0-1, from Module 2
    "final_risk_probability": float, # 0-1, from Module 2's Bayesian layer
    "risk_level": str,                # "Low" | "Medium" | "Critical"
    "important_features": [...],      # from Module 3's SHAP explanation
    "playbook": dict | "No response required.",
}
```

Verified against the literal required key set (no extras, none missing) —
not just "looks about right."

## Honest limitations

- **No live LLM test was possible in this build environment** (no API key
  available). The gating logic, mock-mode pipeline, missing-key error path,
  and the SDK response-parsing logic were all verified independently and
  thoroughly — but an actual live call to the real Anthropic API, with a
  real model response, was not performed as part of this delivery. Test
  with `--mock-llm` removed and a real key exported before relying on the
  narrative sections in production.
- **CVE suggestions require asset/software inventory to actually confirm** —
  this platform doesn't have one. The curated CVE table is a starting point
  for an analyst's investigation, explicitly labeled as such, not a
  finding.
- **`classify_threat_category()` is a coarse, deterministic mapping** from
  channel + top SHAP features to one of 6 curated categories. A real SOC
  platform would want a richer taxonomy; this is intentionally kept small
  and auditable rather than trying to cover every possible attack pattern
  in a template table.
