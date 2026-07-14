"""
app.py
------
Module 3 dashboard — main entrypoint / Page 1: Threat Overview.

Run:
    cd module3/src && streamlit run app.py

Requires module3/data/unified_threat_data.csv to exist (run build_dataset.py
first) and module3/models/surrogate_model.pkl (auto-trains on first load if
missing, via xai_engine.train_surrogate_model()).
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dashboard_common import get_dataset, PAGE_ICON, RISK_COLORS

st.set_page_config(page_title="Threat Overview", page_icon=PAGE_ICON, layout="wide")

st.title(f"{PAGE_ICON} Cybersecurity Threat Detection Platform")
st.caption("Module 3 — Explainable AI Dashboard  |  Page 1: Threat Overview")

df = get_dataset()

if df.empty:
    st.error("Unified dataset is empty. Run build_dataset.py first.")
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar: scope selector
# -----------------------------------------------------------------------------
st.sidebar.header("Overview scope")
gauge_mode = st.sidebar.radio(
    "Risk gauge shows:",
    ["Highest-risk event", "Mean across all events", "Most recent event"],
)

if gauge_mode == "Highest-risk event":
    focus_row = df.loc[df["final_risk_probability"].idxmax()]
    gauge_value = focus_row["final_risk_probability"] * 100
    gauge_caption = f"Highest-risk event: {focus_row['source_ip']} -> {focus_row['destination_ip']} ({focus_row['channel']})"
elif gauge_mode == "Most recent event":
    focus_row = df.sort_values("timestamp").iloc[-1]
    gauge_value = focus_row["final_risk_probability"] * 100
    gauge_caption = f"Most recent event: {focus_row['timestamp']}"
else:
    gauge_value = df["final_risk_probability"].mean() * 100
    gauge_caption = f"Mean across all {len(df):,} events"

# -----------------------------------------------------------------------------
# Row 1: Risk Gauge + Threat Counts
# -----------------------------------------------------------------------------
col1, col2 = st.columns([1, 1.3])

with col1:
    st.subheader("Risk Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gauge_value,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#34495E"},
            "steps": [
                {"range": [0, 40], "color": RISK_COLORS["Low"]},
                {"range": [40, 70], "color": RISK_COLORS["Medium"]},
                {"range": [70, 100], "color": RISK_COLORS["Critical"]},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.8, "value": gauge_value},
        },
    ))
    fig_gauge.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_gauge, width='stretch')
    st.caption(gauge_caption)

with col2:
    st.subheader("Threat Counts")
    counts = df["risk_category"].value_counts().reindex(["Low", "Medium", "Critical"]).fillna(0)
    fig_counts = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=[RISK_COLORS[c] for c in counts.index],
        text=counts.values, textposition="outside",
    ))
    fig_counts.update_layout(height=320, yaxis_title="Event count", margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_counts, width='stretch')

    m1, m2, m3 = st.columns(3)
    m1.metric("Total events", f"{len(df):,}")
    m2.metric("Actual attack rate", f"{df['is_attack'].mean():.1%}")
    m3.metric("Actual phishing rate", f"{df['is_phishing'].mean():.1%}")

# -----------------------------------------------------------------------------
# Row 2: Risk Timeline (Network Timeline chart)
# -----------------------------------------------------------------------------
st.subheader("Risk Timeline")
timeline_df = df.sort_values("timestamp").copy()
timeline_df["rolling_risk"] = timeline_df["final_risk_probability"].rolling(20, min_periods=1).mean()

fig_timeline = go.Figure()
fig_timeline.add_trace(go.Scatter(
    x=timeline_df["timestamp"], y=timeline_df["final_risk_probability"],
    mode="markers", name="Event risk",
    marker=dict(
        size=5, opacity=0.5,
        color=timeline_df["risk_category"].map(RISK_COLORS),
    ),
))
fig_timeline.add_trace(go.Scatter(
    x=timeline_df["timestamp"], y=timeline_df["rolling_risk"],
    mode="lines", name="Rolling mean (20 events)", line=dict(color="#2C3E50", width=2),
))
fig_timeline.update_layout(
    height=380, yaxis_title="final_risk_probability", xaxis_title="Timestamp",
    margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", y=1.1),
)
st.plotly_chart(fig_timeline, width='stretch')

# -----------------------------------------------------------------------------
# Row 3: Packet Statistics
# -----------------------------------------------------------------------------
st.subheader("Packet Statistics")
p1, p2 = st.columns(2)

with p1:
    st.markdown("**Packet Size Distribution**")
    fig_dist = px.histogram(
        df, x="packet_size", color="risk_category",
        color_discrete_map=RISK_COLORS, nbins=60, opacity=0.75,
        category_orders={"risk_category": ["Low", "Medium", "Critical"]},
    )
    fig_dist.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_dist, width='stretch')

with p2:
    st.markdown("**Correlation Heatmap** (raw + engineered network features)")
    corr_cols = ["packet_size", "packets_per_second", "burst_frequency", "failed_connections",
                 "dns_requests", "http_requests", "bytes_sent", "bytes_received",
                 "session_duration", "anomaly_score", "final_risk_probability"]
    corr = df[corr_cols].corr()
    fig_heat = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}", textfont={"size": 8},
    ))
    fig_heat.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_heat, width='stretch')

st.markdown("**Summary statistics**")
st.dataframe(
    df[["packet_size", "packets_per_second", "burst_frequency", "failed_connections",
        "dns_requests", "http_requests", "session_duration"]].describe().round(2),
    width='stretch',
)

st.divider()
st.page_link("pages/2_Explainability.py", label="Next: Explainability →", icon="🔍")
