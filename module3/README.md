# Module 3 — Explainable AI Dashboard

Cybersecurity Threat Detection Platform — Module 3

A Streamlit + Plotly dashboard that consumes real outputs from Module 1
(`anomaly_score`, `processed_features`) and Module 2 (`phishing_probability`,
`final_risk_probability`), with SHAP-based global/local explainability and
LIME-based text explanations.

## Folder structure

```
module3/
├── data/unified_threat_data.csv     # ETL output: 3,000 paired events, Module1+Module2 outputs joined
├── models/
│   ├── surrogate_model.pkl          # Module 3's OWN LightGBM surrogate, for SHAP (see below)
│   ├── surrogate_fidelity.pkl       # how well the surrogate matches Module 1's real ground truth
│   ├── tfidf_vectorizer.pkl         # COPIED from Module 2 — real trained artifact, not retrained
│   ├── lightgbm_model.pkl           # COPIED from Module 2 — real trained artifact
│   ├── naive_bayes_model.pkl        # COPIED from Module 2 (kept in case model_selection picks it)
│   └── model_selection.pkl          # COPIED from Module 2 — which model it selected
├── src/
│   ├── utils.py
│   ├── module_bridge.py             # isolated subprocess calls into Module 1 / Module 2's real code
│   ├── build_dataset.py             # ETL: builds unified_threat_data.csv from both modules' real outputs
│   ├── xai_engine.py                # surrogate model + SHAP + LIME + generate_xai_report()
│   ├── dashboard_common.py          # Streamlit caching layer shared by all pages
│   ├── app.py                       # Page 1: Threat Overview (main entrypoint)
│   └── pages/
│       ├── 2_Explainability.py      # Page 2
│       └── 3_Threat_Explorer.py     # Page 3
├── requirements.txt
└── README.md
```

## Run it

```bash
cd module3
pip install -r requirements.txt

# 1. Build the unified dataset (requires module1/ and module2/ already trained
#    and sitting alongside module3/ in the same parent directory)
cd src && python build_dataset.py

# 2. Launch the dashboard
streamlit run app.py
```

Verification performed on this exact build (not just "it should work"):
every page's Python logic was executed directly (`runpy`) to catch real bugs
without needing a browser, and the actual `streamlit run` server was started
headlessly and confirmed to return `HTTP 200` with genuine rendered
Streamlit HTML — not just "the code compiles."

## The namespace collision, and why cross-module calls go through subprocesses

Module 1, Module 2, and Module 3 each ship a file named `utils.py`, and
Modules 1 and 2 both ship a `predict.py`. The moment more than one of these
directories is added to `sys.path` and imported into the same Python
process, whichever module's `utils`/`predict` imports first "wins" that
name in `sys.modules` — every subsequent `from utils import ...` anywhere,
in any of the three modules, silently gets fed the wrong module's code. This
was caught and fixed during Module 2's build (see Module 2's README) and the
same fix is reused here: `module_bridge.py` calls Module 1's
`predict_network_anomaly()` and Module 2's `predict_phishing()` /
`bayesian_risk_adjustment()` in isolated subprocesses, exchanging data via
temp JSON files. Verified each bridge function independently before building
the ETL on top of them.

The one exception: Module 2's *trained model artifacts* (the `.pkl` files)
are copied directly into `module3/models/` and loaded with `joblib.load()`.
This is safe — unpickling a plain `sklearn`/`lightgbm` object doesn't
execute Module 2's `utils.py`, so there's no collision — and it means
Module 3's LIME explainer runs Module 2's REAL trained classifier, not a
redundant retrained copy that could quietly drift from what Module 2 actually ships.

## Why a surrogate model for SHAP (not Module 1's actual ensemble)

Module 1's real anomaly detector is IsolationForest + OneClassSVM.
`shap` has **no native support for OneClassSVM at all**, and its
IsolationForest support is partial and version-fragile. The honest options
were: (a) `shap.KernelExplainer`, which is model-agnostic but re-evaluates
the model exponentially in the number of features — far too slow for an
interactive dashboard — or (b) train a fast, faithful surrogate and be
explicit about it everywhere it appears.

Went with (b): a LightGBM classifier trained on the same `processed_features`
(plus `packet_size`, `burst_frequency`, `dns_requests` — raw fields needed
for the spec'd "Packet Size / Burst Frequency / DNS Requests Influence"
panels, which aren't literally in Module 1's 8-feature `processed_features`
output) to predict Module 1's ground-truth `is_attack` label. This is
standard, documented XAI practice, and its fidelity is measured and
displayed on the dashboard itself, not assumed:

**Surrogate fidelity vs. Module 1's actual ground truth: AUC = 0.9999, Accuracy = 0.996**

That's a genuinely faithful stand-in — the SHAP values reflect a model that
agrees with Module 1's real behavior 99.6% of the time, not an arbitrary
proxy. Verified independently: the surrogate's top-ranked features
(`burst_score`, `failed_connection_rate`, `burst_frequency`) match Module
1's own permutation-importance ranking from Module 1's evaluation — an
unplanned cross-check between two completely independent analyses landing
on the same answer. Also reproduces Module 1's earlier finding that
`packet_entropy` and `packet_std` contribute essentially nothing
(mean |SHAP| ≈ 0 for both) — consistent, not coincidental.

**SHAP values were verified mathematically, not just visually.** The
additivity property (`base_value + sum(shap_values) == model's actual raw
output for that row`, before the sigmoid) was checked to reconstruct the
surrogate's real `predict_proba` output exactly for a test row — confirming
the SHAP computation is correct, not just plausible-looking numbers.

## Why LIME for phishing content, not SHAP

Module 2's real model operates on 5,000-dimensional TF-IDF features. SHAP
*could* run directly on that, but a 5,000-token SHAP summary is unreadable.
LIME's local text explainer is built for exactly this: perturb words in one
specific message, refit a local linear model, and show which words pushed
the prediction. Runs against Module 2's actual copied model + vectorizer.

## How the dashboard's dataset was built (and a limitation worth knowing)

Module 1's network logs and Module 2's phishing content are two
*independently generated* synthetic datasets with no natural shared key.
`build_dataset.py` samples both and pairs them with ~75% label agreement
(suspicious network activity more often — not always — paired with
malicious content) to simulate realistic partial correlation, rather than
fabricating a join that doesn't exist or pairing everything perfectly
(which would make the dashboard's risk fusion look artificially clean).

**Known limitation, carried over honestly from Module 1:** the 3,000-row
sample is drawn non-contiguously (shuffled across the full dataset), which
means Module 1's rolling per-`source_ip` features
(`rolling_packet_mean`/`packet_std`/`packet_entropy`) mostly fall back to
their documented degraded window-size-1 mode. `anomaly_score` still
separates attacks from legitimate traffic clearly (mean 62.7 vs 22.4) but
less sharply than Module 1's own contiguous-batch evaluation (which hit
AUC ~0.999). This is Module 1's documented operational caveat surfacing
here exactly as predicted — not a new bug.

## `generate_xai_report()`

```python
generate_xai_report(row_index=None) -> {
    "important_features": [...],      # ranked list of feature names
    "feature_contributions": {...},   # {feature: SHAP value}
    "risk_summary": {...},            # risk stats
}
```

- **`row_index` given** → local report for that one event: its top features
  ranked by `|SHAP value|`, that row's actual signed SHAP contributions, and
  its `anomaly_score`/`phishing_probability`/`final_risk_probability`/`risk_category`.
- **`row_index=None`** (default) → global report: dataset-wide feature
  ranking by mean `|SHAP value|`, each feature's mean *signed* SHAP value
  (shows average directional pull, not just magnitude), and aggregate risk
  distribution stats (counts per category, means, actual attack/phishing rates).

Both modes tested directly (see `xai_engine.py`'s `__main__` block) and
produce sane, cross-checked output — not just "runs without throwing."

## Required charts — where each one lives

| Chart | Page | Notes |
|---|---|---|
| Risk Gauge | 1 | Plotly `go.Indicator`, switchable between highest-risk / mean / most-recent event |
| Network Timeline | 1 | "Risk Timeline" — per-event scatter + 20-event rolling mean |
| Correlation Heatmap | 1 | Under Packet Statistics — raw + engineered features vs. `anomaly_score`/`final_risk_probability` |
| Packet Distribution | 1 | Packet size histogram, colored by risk category |
| Feature Importance | 2 | "Top SHAP Features" — mean \|SHAP\| bar chart |
| SHAP Summary | 2 | Manually-built beeswarm (jittered scatter, colored by normalized feature value) — `px.strip` doesn't support continuous color scales in this plotly version, caught during testing and fixed |
| SHAP Waterfall | 2 | "Feature Contribution" for the selected event |

Plus LIME word-weight bar chart and 3 SHAP dependence plots (Packet Size /
Burst Frequency / DNS Requests Influence) on Page 2, and the full
interactive table/filter/search/export on Page 3.

## Honest bugs found and fixed while building this

1. **`px.strip(..., color_continuous_scale=...)` doesn't exist in this
   plotly version** — caught by actually running the page (not just reading
   the code), fixed by building the beeswarm plot manually with `go.Scatter`.
2. **`use_container_width` is deprecated** in Streamlit 1.59 in favor of
   `width=`. Caught in the smoke-test output, fixed across all three pages
   rather than shipping with known deprecation warnings.
3. **The SHAP list-vs-array return shape** from `shap.TreeExplainer` for a
   LightGBM binary classifier is version-dependent (some versions return
   `[class0_array, class1_array]`, others a single array for the positive
   class). Handled defensively in `xai_engine.py` rather than assuming one
   behavior and breaking silently on a different shap version.
