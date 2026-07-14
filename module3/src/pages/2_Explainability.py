"""
pages/2_Explainability.py
---------------------------
Page 2: Explainability. Top SHAP Features, LIME Explanation, Feature
Contribution (SHAP waterfall), and Packet Size / Burst Frequency / DNS
Requests Influence (SHAP dependence plots).
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard_common import (
    get_dataset, cached_global_feature_importance, cached_local_shap,
    cached_dependence_data, cached_lime_explanation, shap_base_value,
    PAGE_ICON, RISK_COLORS,
)
import xai_engine as xe

st.set_page_config(page_title="Explainability", page_icon=PAGE_ICON, layout="wide")
st.title("🔍 Explainability")
st.caption(
    "SHAP explains a surrogate LightGBM model trained to reproduce Module 1's is_attack "
    "label from processed_features (+ 3 raw fields) — necessary because Module 1's actual "
    "IsolationForest + OneClassSVM ensemble has no practical native SHAP support. "
    "LIME explains Module 2's REAL trained phishing classifier directly on text. "
    "See README for the full rationale."
)

df = get_dataset()

# -----------------------------------------------------------------------------
# Surrogate fidelity disclosure — always visible, not buried
# -----------------------------------------------------------------------------
fidelity_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "surrogate_fidelity.pkl")
if os.path.exists(fidelity_path):
    import joblib
    fidelity = joblib.load(fidelity_path)
    f1, f2 = st.columns(2)
    f1.metric("Surrogate fidelity — AUC vs. Module 1's ground truth", f"{fidelity['auc_vs_is_attack']:.4f}")
    f2.metric("Surrogate fidelity — Accuracy vs. Module 1's ground truth", f"{fidelity['accuracy_vs_is_attack']:.4f}")

st.divider()

# -----------------------------------------------------------------------------
# Top SHAP Features (global feature importance) + SHAP Summary
# -----------------------------------------------------------------------------
col1, col2 = st.columns(2)

importance_df = cached_global_feature_importance()

with col1:
    st.subheader("Top SHAP Features")
    fig_imp = go.Figure(go.Bar(
        x=importance_df["mean_abs_shap"], y=importance_df["feature"],
        orientation="h", marker_color="#2E86AB",
    ))
    fig_imp.update_layout(
        height=420, xaxis_title="Mean |SHAP value|", yaxis=dict(autorange="reversed"),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_imp, width='stretch')

with col2:
    st.subheader("SHAP Summary")
    st.caption("Each dot = one event. Color = that event's raw feature value. X = SHAP value (pull toward attack).")
    xe._ensure_surrogate_loaded()
    top_features = importance_df["feature"].tolist()[:8]

    rng = np.random.RandomState(0)
    fig_summary = go.Figure()
    for i, feat in enumerate(top_features):
        idx = xe.SURROGATE_FEATURE_COLS.index(feat)
        vals = df[feat].values
        shap_vals = xe._shap_values[:, idx]
        norm_vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
        jitter = rng.uniform(-0.35, 0.35, size=len(shap_vals))
        fig_summary.add_trace(go.Scatter(
            x=shap_vals, y=[i + j for j in jitter],
            mode="markers",
            marker=dict(
                size=4, opacity=0.55, color=norm_vals,
                colorscale="RdBu_r", showscale=(i == 0),
                colorbar=dict(title="feature<br>value<br>(norm)") if i == 0 else None,
            ),
            name=feat, showlegend=False,
            hovertemplate=f"{feat}<br>SHAP=%{{x:.3f}}<extra></extra>",
        ))
    fig_summary.update_layout(
        height=420, margin=dict(l=20, r=20, t=20, b=20),
        yaxis=dict(tickmode="array", tickvals=list(range(len(top_features))), ticktext=top_features, autorange="reversed"),
        xaxis_title="SHAP value",
    )
    st.plotly_chart(fig_summary, width='stretch')

st.divider()

# -----------------------------------------------------------------------------
# Event selector — drives Feature Contribution (waterfall) + LIME
# -----------------------------------------------------------------------------
st.subheader("Select an event to explain locally")
sort_choice = st.selectbox(
    "Pick from:",
    ["Highest final_risk_probability", "Lowest final_risk_probability", "Random sample", "By row index"],
)
if sort_choice == "Highest final_risk_probability":
    candidates = df.sort_values("final_risk_probability", ascending=False).head(20)
elif sort_choice == "Lowest final_risk_probability":
    candidates = df.sort_values("final_risk_probability", ascending=True).head(20)
elif sort_choice == "Random sample":
    candidates = df.sample(20, random_state=1)
else:
    candidates = df

if sort_choice == "By row index":
    row_index = st.number_input("Row index", min_value=0, max_value=len(df) - 1, value=0, step=1)
else:
    options = candidates.index.tolist()
    row_index = st.selectbox(
        "Event",
        options,
        format_func=lambda i: f"row {i} | risk={df.loc[i, 'final_risk_probability']:.2f} | "
                               f"{df.loc[i, 'channel']} | {df.loc[i, 'risk_category']}",
    )

selected = df.loc[row_index]
st.markdown(
    f"**Selected event {row_index}**: `{selected['source_ip']} -> {selected['destination_ip']}` | "
    f"channel=`{selected['channel']}` | anomaly_score=`{selected['anomaly_score']:.1f}` | "
    f"phishing_probability=`{selected['phishing_probability']:.3f}` | "
    f"final_risk_probability=`{selected['final_risk_probability']:.3f}` (**{selected['risk_category']}**)"
)
with st.expander("Raw content for this event"):
    st.code(selected["text"])

col3, col4 = st.columns(2)

# -----------------------------------------------------------------------------
# Feature Contribution (SHAP Waterfall)
# -----------------------------------------------------------------------------
with col3:
    st.subheader("Feature Contribution (SHAP Waterfall)")
    local_shap = cached_local_shap(row_index)
    base = shap_base_value()

    waterfall_df = local_shap.copy()
    cum = base
    measures, ys, texts = [], [], []
    for _, r in waterfall_df.iterrows():
        measures.append("relative")
        ys.append(r["shap_value"])
        texts.append(f"{r['feature']}={r['value']:.2f}")
    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute"] + measures + ["total"],
        x=["base_value"] + waterfall_df["feature"].tolist() + ["final margin"],
        y=[base] + ys + [0],
        text=[f"{base:.2f}"] + texts + [""],
        connector={"line": {"color": "rgba(100,100,100,0.4)"}},
        decreasing={"marker": {"color": "#2ECC71"}},
        increasing={"marker": {"color": "#E74C3C"}},
        totals={"marker": {"color": "#34495E"}},
    ))
    fig_wf.update_layout(height=420, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
    st.plotly_chart(fig_wf, width='stretch')
    st.caption("Red = pushes toward attack, Green = pushes toward legitimate. Base value is the surrogate model's average raw output.")

# -----------------------------------------------------------------------------
# LIME Explanation
# -----------------------------------------------------------------------------
with col4:
    st.subheader("LIME Explanation (phishing content)")
    lime_result = cached_lime_explanation(row_index)
    st.metric("Module 2 phishing_probability (real model)", f"{lime_result['prediction_proba']:.3f}")

    words = [w for w, _ in lime_result["word_weights"]]
    weights = [w for _, w in lime_result["word_weights"]]
    colors = ["#E74C3C" if w > 0 else "#2ECC71" for w in weights]
    fig_lime = go.Figure(go.Bar(x=weights, y=words, orientation="h", marker_color=colors))
    fig_lime.update_layout(
        height=420, xaxis_title="LIME weight (+ = pushes toward phishing)",
        yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_lime, width='stretch')

st.divider()

# -----------------------------------------------------------------------------
# Packet Size / Burst Frequency / DNS Requests Influence (SHAP dependence plots)
# -----------------------------------------------------------------------------
st.subheader("Feature Influence (SHAP dependence)")
d1, d2, d3 = st.columns(3)

influence_specs = [
    ("packet_size", "Packet Size Influence", d1),
    ("burst_frequency", "Burst Frequency Influence", d2),
    ("dns_requests", "DNS Requests Influence", d3),
]
for feat, title, col in influence_specs:
    with col:
        st.markdown(f"**{title}**")
        dep = cached_dependence_data(feat)
        fig_dep = px.scatter(
            dep, x="value", y="shap_value", color=dep["is_attack"].map({0: "Legitimate", 1: "Attack"}),
            color_discrete_map={"Legitimate": "#2ECC71", "Attack": "#E74C3C"},
            opacity=0.5, labels={"value": feat, "shap_value": "SHAP value", "color": ""},
        )
        fig_dep.update_traces(marker=dict(size=5))
        fig_dep.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_dep, width='stretch')

st.divider()
c_prev, c_next = st.columns(2)
c_prev.page_link("app.py", label="← Back: Threat Overview", icon="🛡️")
c_next.page_link("pages/3_Threat_Explorer.py", label="Next: Threat Explorer →", icon="🗂️")
