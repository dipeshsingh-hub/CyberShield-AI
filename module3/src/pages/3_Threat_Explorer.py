"""
pages/3_Threat_Explorer.py
-----------------------------
Page 3: Threat Explorer. Interactive table, filtering, search, CSV export.
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard_common import get_dataset, PAGE_ICON, RISK_COLORS

st.set_page_config(page_title="Threat Explorer", page_icon=PAGE_ICON, layout="wide")
st.title("🗂️ Threat Explorer")
st.caption("Filter, search, and export the underlying event data.")

df = get_dataset()

DISPLAY_COLS = [
    "timestamp", "source_ip", "destination_ip", "protocol", "channel", "country",
    "anomaly_score", "module1_risk_level", "phishing_probability",
    "final_risk_probability", "risk_category", "is_attack", "is_phishing", "text",
]

# -----------------------------------------------------------------------------
# Sidebar filters
# -----------------------------------------------------------------------------
st.sidebar.header("Filters")

risk_filter = st.sidebar.multiselect(
    "Risk category", options=["Low", "Medium", "Critical"], default=["Low", "Medium", "Critical"],
)
channel_filter = st.sidebar.multiselect(
    "Channel", options=sorted(df["channel"].unique()), default=sorted(df["channel"].unique()),
)
protocol_filter = st.sidebar.multiselect(
    "Protocol", options=sorted(df["protocol"].unique()), default=sorted(df["protocol"].unique()),
)
country_filter = st.sidebar.multiselect(
    "Country", options=sorted(df["country"].unique()), default=sorted(df["country"].unique()),
)

risk_range = st.sidebar.slider(
    "final_risk_probability range", 0.0, 1.0, (0.0, 1.0), step=0.01,
)
anomaly_range = st.sidebar.slider(
    "anomaly_score range", 0.0, 100.0, (0.0, 100.0), step=1.0,
)

date_min = df["timestamp"].min().to_pydatetime()
date_max = df["timestamp"].max().to_pydatetime()
date_range = st.sidebar.slider(
    "Timestamp range", min_value=date_min, max_value=date_max, value=(date_min, date_max),
)

search_query = st.sidebar.text_input("Search (IP, content text, country)")

only_attacks = st.sidebar.checkbox("Only actual attacks (is_attack=1)", value=False)
only_phishing = st.sidebar.checkbox("Only actual phishing (is_phishing=1)", value=False)

# -----------------------------------------------------------------------------
# Apply filters
# -----------------------------------------------------------------------------
filtered = df[
    df["risk_category"].isin(risk_filter)
    & df["channel"].isin(channel_filter)
    & df["protocol"].isin(protocol_filter)
    & df["country"].isin(country_filter)
    & df["final_risk_probability"].between(*risk_range)
    & df["anomaly_score"].between(*anomaly_range)
    & df["timestamp"].between(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))
]

if only_attacks:
    filtered = filtered[filtered["is_attack"] == 1]
if only_phishing:
    filtered = filtered[filtered["is_phishing"] == 1]

if search_query:
    q = search_query.lower()
    mask = (
        filtered["source_ip"].str.lower().str.contains(q)
        | filtered["destination_ip"].str.lower().str.contains(q)
        | filtered["text"].str.lower().str.contains(q, regex=False)
        | filtered["country"].str.lower().str.contains(q)
    )
    filtered = filtered[mask]

# -----------------------------------------------------------------------------
# Summary + table
# -----------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Matching events", f"{len(filtered):,}")
m2.metric("Mean final_risk_probability", f"{filtered['final_risk_probability'].mean():.3f}" if len(filtered) else "—")
m3.metric("Critical count", f"{(filtered['risk_category'] == 'Critical').sum():,}")
m4.metric("% of total dataset", f"{len(filtered) / len(df):.1%}" if len(df) else "—")

st.divider()

sort_col = st.selectbox("Sort by", ["final_risk_probability", "anomaly_score", "phishing_probability", "timestamp"], index=0)
sort_desc = st.checkbox("Descending", value=True)
filtered_sorted = filtered.sort_values(sort_col, ascending=not sort_desc)

st.dataframe(
    filtered_sorted[DISPLAY_COLS].style.format({
        "anomaly_score": "{:.1f}",
        "phishing_probability": "{:.3f}",
        "final_risk_probability": "{:.3f}",
    }),
    width='stretch',
    height=480,
)

# -----------------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------------
st.download_button(
    label=f"⬇ Export {len(filtered_sorted):,} filtered rows as CSV",
    data=filtered_sorted[DISPLAY_COLS].to_csv(index=False).encode("utf-8"),
    file_name="threat_explorer_export.csv",
    mime="text/csv",
)

st.divider()
st.page_link("pages/2_Explainability.py", label="← Back: Explainability", icon="🔍")
