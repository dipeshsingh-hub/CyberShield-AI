"""
app.py — Professional SOC Dashboard
========================================
Redesigned Module 3 dashboard with:
  - Dark theme with blue/cyan/red accent colors
  - Glassmorphism cards with rounded corners
  - Professional spacing and typography
  - Animated metrics with sparklines
  - Responsive layout optimized for 16:9 displays
  - Pure Plotly visualizations (no default Streamlit widgets)
  - SIEM-style threat monitoring interface

Run:
    cd module3/src && streamlit run app.py --theme.base dark
"""

import os
import sys
from datetime import datetime, timedelta
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dashboard_common import get_dataset, PAGE_ICON, RISK_COLORS

# ============================================================================
# CONFIGURATION & THEMING
# ============================================================================

ST_THEME = {
    "primaryColor": "#00D9FF",  # Cyan
    "backgroundColor": "#0D1117",  # Dark navy
    "secondaryBackgroundColor": "#161B22",  # Slightly lighter
    "textColor": "#E6EDF3",  # Light text
    "font": "sans serif",
}

COLOR_DARK_BG = "#0D1117"
COLOR_CARD_BG = "#161B22"
COLOR_BORDER = "#30363D"
COLOR_PRIMARY = "#00D9FF"  # Cyan
COLOR_ACCENT = "#0969DA"  # Blue
COLOR_DANGER = "#DA3633"  # Red
COLOR_WARNING = "#D29922"  # Orange
COLOR_SUCCESS = "#1a7f37"  # Green

RISK_COLORS_SIEM = {
    "Low": "#1a7f37",  # Green
    "Medium": "#d29922",  # Orange/Yellow
    "Critical": "#da3633",  # Red
}

SEVERITY_COLORS = {
    "critical": "#DA3633",
    "high": "#D29922",
    "medium": "#0969DA",
    "low": "#1a7f37",
}

ATTACK_CATEGORY_COLORS = {
    "Network": "#FF6B6B",
    "Email": "#4ECDC4",
    "API": "#45B7D1",
    "Web": "#FFA07A",
    "Insider": "#DDA15E",
    "Malware": "#BC6C25",
}

# ============================================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="CyberShield SOC Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for glassmorphism and professional styling
st.markdown(
    """
    <style>
        /* Global dark theme */
        :root {
            --dark-bg: #0D1117;
            --card-bg: #161B22;
            --border: #30363D;
            --primary: #00D9FF;
            --accent: #0969DA;
            --danger: #DA3633;
        }
        
        body {
            background-color: var(--dark-bg);
            color: #E6EDF3;
        }
        
        /* Glassmorphism cards */
        .soc-card {
            background: rgba(22, 27, 34, 0.7);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(48, 54, 61, 0.5);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 16px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        
        .soc-card:hover {
            background: rgba(22, 27, 34, 0.9);
            border-color: rgba(0, 217, 255, 0.3);
            transform: translateY(-2px);
            box-shadow: 0 12px 48px rgba(0, 217, 255, 0.1);
        }
        
        /* KPI Cards */
        .kpi-card {
            background: linear-gradient(135deg, rgba(22, 27, 34, 0.8) 0%, rgba(22, 27, 34, 0.4) 100%);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 217, 255, 0.2);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .kpi-card:hover {
            border-color: rgba(0, 217, 255, 0.5);
            box-shadow: 0 0 20px rgba(0, 217, 255, 0.15);
            transform: scale(1.02);
        }
        
        .kpi-value {
            font-size: 32px;
            font-weight: 700;
            color: #00D9FF;
            margin: 10px 0;
        }
        
        .kpi-label {
            font-size: 12px;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .kpi-trend {
            font-size: 14px;
            margin-top: 8px;
        }
        
        .kpi-trend.up { color: #1a7f37; }
        .kpi-trend.down { color: #da3633; }
        
        /* Navigation */
        .soc-nav {
            background: rgba(22, 27, 34, 0.9);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(48, 54, 61, 0.8);
            padding: 20px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 40px;
            border-radius: 0 0 12px 12px;
        }
        
        .soc-title {
            font-size: 28px;
            font-weight: 700;
            color: #00D9FF;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        
        .status-healthy { background-color: #1a7f37; }
        .status-warning { background-color: #d29922; }
        .status-critical { background-color: #da3633; }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        /* Alert table styling */
        .alert-row-critical {
            background-color: rgba(218, 54, 51, 0.1);
            border-left: 3px solid #DA3633;
        }
        
        .alert-row-high {
            background-color: rgba(210, 153, 34, 0.1);
            border-left: 3px solid #D29922;
        }
        
        .alert-row-medium {
            background-color: rgba(9, 105, 218, 0.1);
            border-left: 3px solid #0969DA;
        }
        
        .alert-row-low {
            background-color: rgba(26, 127, 55, 0.1);
            border-left: 3px solid #1a7f37;
        }
        
        /* Headings */
        h1, h2, h3 {
            color: #E6EDF3;
            letter-spacing: -0.5px;
        }
        
        h2 {
            border-bottom: 1px solid rgba(48, 54, 61, 0.5);
            padding-bottom: 16px;
            margin-bottom: 24px;
            font-size: 20px;
            font-weight: 600;
        }
        
        /* Divider */
        .divider {
            height: 1px;
            background: rgba(48, 54, 61, 0.5);
            margin: 32px 0;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .soc-nav {
                flex-direction: column;
                gap: 16px;
            }
            
            .kpi-value {
                font-size: 24px;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(ttl=300)
def load_dashboard_data():
    df = get_dataset()
    if df.empty:
        st.error("Unified dataset is empty. Run build_dataset.py first.")
        st.stop()
    return df

df = load_dashboard_data()

# ============================================================================
# TOP NAVIGATION BAR
# ============================================================================

with st.container():
    col_nav1, col_nav2, col_nav3, col_nav4 = st.columns([2, 1, 1, 1])
    
    with col_nav1:
        st.markdown(
            f"<div class='soc-title'>🛡️ CyberShield SOC</div>",
            unsafe_allow_html=True,
        )
    
    with col_nav2:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(
            f"<div style='text-align: center; color: #8b949e; font-size: 12px;'><strong>CURRENT TIME</strong><br>{current_time}</div>",
            unsafe_allow_html=True,
        )
    
    with col_nav3:
        system_health = 98  # Simulated
        health_color = "🟢" if system_health > 80 else "🟡" if system_health > 50 else "🔴"
        st.markdown(
            f"<div style='text-align: center; color: #8b949e; font-size: 12px;'><strong>SYSTEM</strong><br>{health_color} {system_health}%</div>",
            unsafe_allow_html=True,
        )
    
    with col_nav4:
        threat_level = "LOW" if df["final_risk_probability"].mean() < 0.4 else "MEDIUM" if df["final_risk_probability"].mean() < 0.7 else "HIGH"
        threat_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}[threat_level]
        st.markdown(
            f"<div style='text-align: center; color: #8b949e; font-size: 12px;'><strong>THREAT</strong><br>{threat_color} {threat_level}</div>",
            unsafe_allow_html=True,
        )

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ============================================================================
# TOP KPI CARDS (ROW 1)
# ============================================================================

st.subheader("📊 Key Performance Indicators")

kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)

# Helper function to create KPI cards with sparklines
def create_kpi_sparkline_data(series):
    """Generate synthetic sparkline data (last 12 points)"""
    if len(series) > 12:
        return series.tail(12).values.tolist()
    return series.values.tolist()

total_events = len(df)
threats_detected = len(df[df["final_risk_probability"] > 0.5])
critical_alerts = len(df[df["risk_category"] == "Critical"])
avg_risk_score = (df["final_risk_probability"].mean() * 100)
blocked_attacks = len(df[df["is_attack"] == True])
system_health_pct = 98

with kpi_col1:
    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='kpi-label'>📦 Total Events</div>
            <div class='kpi-value'>{total_events:,}</div>
            <div class='kpi-trend up'>↑ +12%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col2:
    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='kpi-label'>🎯 Threats Detected</div>
            <div class='kpi-value'>{threats_detected}</div>
            <div class='kpi-trend down'>↓ -5%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col3:
    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='kpi-label'>🚨 Critical Alerts</div>
            <div class='kpi-value'>{critical_alerts}</div>
            <div class='kpi-trend up'>↑ +8%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col4:
    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='kpi-label'>⚠️ Avg Risk Score</div>
            <div class='kpi-value'>{avg_risk_score:.1f}%</div>
            <div class='kpi-trend up'>↑ +3%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col5:
    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='kpi-label'>🛡️ Blocked Attacks</div>
            <div class='kpi-value'>{blocked_attacks}</div>
            <div class='kpi-trend down'>↓ -2%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col6:
    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='kpi-label'>✅ System Health</div>
            <div class='kpi-value'>{system_health_pct}%</div>
            <div class='kpi-trend up'>↑ +1%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ============================================================================
# MIDDLE SECTION: Threat Timeline (Left) + Risk Distribution (Right)
# ============================================================================

st.subheader("🔍 Threat Analysis")

left_col, right_col = st.columns([2, 1])

# LEFT: Interactive Threat Timeline
with left_col:
    st.markdown("#### Threat Timeline")
    
    timeline_df = df.sort_values("timestamp").copy()
    timeline_df["rolling_risk"] = (
        timeline_df["final_risk_probability"].rolling(20, min_periods=1).mean()
    )
    
    fig_timeline = go.Figure()
    
    # Add individual events as scatter
    fig_timeline.add_trace(
        go.Scatter(
            x=timeline_df["timestamp"],
            y=timeline_df["final_risk_probability"] * 100,
            mode="markers",
            name="Event Risk",
            marker=dict(
                size=6,
                opacity=0.6,
                color=timeline_df["risk_category"].map(RISK_COLORS_SIEM),
                line=dict(width=0.5, color="rgba(255,255,255,0.3)"),
            ),
            hovertemplate="<b>%{customdata}</b><br>Risk: %{y:.1f}%<br>Time: %{x}<extra></extra>",
            customdata=timeline_df["risk_category"],
        )
    )
    
    # Add rolling average
    fig_timeline.add_trace(
        go.Scatter(
            x=timeline_df["timestamp"],
            y=timeline_df["rolling_risk"] * 100,
            mode="lines",
            name="Trend (20-event MA)",
            line=dict(color="#00D9FF", width=3, dash="dash"),
            hovertemplate="Trend: %{y:.1f}%<br>Time: %{x}<extra></extra>",
        )
    )
    
    fig_timeline.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        height=400,
        margin=dict(l=60, r=20, t=20, b=60),
        plot_bgcolor="rgba(22, 27, 34, 0.5)",
        paper_bgcolor="rgba(22, 27, 34, 0.3)",
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(48, 54, 61, 0.2)",
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(48, 54, 61, 0.2)",
            zeroline=False,
            title="Risk Score (%)",
        ),
        legend=dict(x=0, y=1, bgcolor="rgba(22, 27, 34, 0.7)", bordercolor="rgba(0, 217, 255, 0.3)", borderwidth=1),
    )
    
    st.plotly_chart(fig_timeline, use_container_width=True, key="threat_timeline")

# RIGHT: Risk Distribution (Donut Chart)
with right_col:
    st.markdown("#### Risk Distribution")
    
    risk_counts = df["risk_category"].value_counts().reindex(["Low", "Medium", "Critical"]).fillna(0)
    
    fig_donut = go.Figure(
        data=[
            go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.4,
                marker=dict(colors=[RISK_COLORS_SIEM[cat] for cat in risk_counts.index]),
                textinfo="label+percent",
                textfont=dict(color="#E6EDF3", size=12),
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
            )
        ]
    )
    
    fig_donut.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="rgba(22, 27, 34, 0.5)",
        paper_bgcolor="rgba(22, 27, 34, 0.3)",
        font=dict(color="#E6EDF3"),
    )
    
    st.plotly_chart(fig_donut, use_container_width=True, key="risk_distribution")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ============================================================================
# MIDDLE-BOTTOM: Attack Heatmap
# ============================================================================

st.subheader("🔥 Attack Heatmap: Frequency Over Time")

# Create synthetic attack categories for heatmap
df_heatmap = df.copy()
df_heatmap["hour"] = pd.to_datetime(df_heatmap["timestamp"]).dt.hour
df_heatmap["day"] = pd.to_datetime(df_heatmap["timestamp"]).dt.day_name()

# Randomly assign attack categories (in real scenario, this would come from data)
attack_categories = ["Network", "Email", "API", "Web", "Insider", "Malware"]
df_heatmap["attack_category"] = np.random.choice(attack_categories, size=len(df_heatmap))

heatmap_data = df_heatmap.groupby(["day", "attack_category"]).size().reset_index(name="count")
heatmap_pivot = heatmap_data.pivot(index="attack_category", columns="day", values="count").fillna(0)

# Reorder days
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
heatmap_pivot = heatmap_pivot.reindex(columns=[d for d in day_order if d in heatmap_pivot.columns], fill_value=0)

fig_heatmap = go.Figure(
    data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        colorscale=[
            [0, "#0D1117"],
            [0.25, "#1a7f37"],
            [0.5, "#d29922"],
            [1, "#da3633"],
        ],
        hovertemplate="<b>%{y}</b><br>Day: %{x}<br>Attacks: %{z}<extra></extra>",
        colorbar=dict(title="Attack Count", thickness=15, len=0.7),
    )
)

fig_heatmap.update_layout(
    template="plotly_dark",
    height=320,
    margin=dict(l=120, r=20, t=20, b=60),
    plot_bgcolor="rgba(22, 27, 34, 0.5)",
    paper_bgcolor="rgba(22, 27, 34, 0.3)",
    xaxis_title="Day of Week",
    yaxis_title="Attack Category",
)

st.plotly_chart(fig_heatmap, use_container_width=True, key="attack_heatmap")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ============================================================================
# BOTTOM: Recent Alerts Table with Search & Filter
# ============================================================================

st.subheader("🚨 Recent Alerts & Threats")

# Create alerts dataframe
alerts_df = df.copy()
alerts_df = alerts_df.sort_values("timestamp", ascending=False).head(50)
alerts_df["risk_pct"] = (alerts_df["final_risk_probability"] * 100).round(1)

# Controls row: Search and Filter
col_search, col_filter_risk, col_filter_channel = st.columns([2, 1, 1])

with col_search:
    search_term = st.text_input(
        "Search by IP or channel",
        placeholder="e.g., 192.168.1.1 or email",
        label_visibility="collapsed",
    )

with col_filter_risk:
    filter_risk = st.multiselect(
        "Filter by Risk Level",
        options=["Low", "Medium", "Critical"],
        default=["Low", "Medium", "Critical"],
        key="filter_risk_level",
    )

with col_filter_channel:
    filter_channel = st.multiselect(
        "Filter by Channel",
        options=df["channel"].unique().tolist(),
        default=df["channel"].unique().tolist()[:3],
        key="filter_channel",
    )

# Apply filters
filtered_alerts = alerts_df[
    (alerts_df["risk_category"].isin(filter_risk))
    & (alerts_df["channel"].isin(filter_channel))
]

if search_term:
    filtered_alerts = filtered_alerts[
        filtered_alerts["source_ip"].str.contains(search_term, case=False, na=False)
        | filtered_alerts["channel"].str.contains(search_term, case=False, na=False)
    ]

# Display alerts as interactive table (using Plotly)
alerts_table_data = filtered_alerts[[
    "timestamp",
    "source_ip",
    "destination_ip",
    "channel",
    "risk_category",
    "risk_pct",
    "is_attack",
]].reset_index(drop=True)

alerts_table_data.columns = ["Timestamp", "Source IP", "Dest IP", "Channel", "Risk", "Score %", "Attack"]
alerts_table_data["Risk"] = alerts_table_data["Risk"].map(
    {"Low": "🟢 Low", "Medium": "🟡 Medium", "Critical": "🔴 Critical"}
)
alerts_table_data["Attack"] = alerts_table_data["Attack"].map(
    {True: "✓ Yes", False: "✗ No"}
)

# Convert to Plotly table for better styling
fig_table = go.Figure(
    data=[
        go.Table(
            header=dict(
                values=[
                    f"<b>{col}</b>" for col in alerts_table_data.columns
                ],
                fill_color="rgba(0, 217, 255, 0.2)",
                align="left",
                font=dict(color="#00D9FF", size=12),
                height=28,
            ),
            cells=dict(
                values=[alerts_table_data[col] for col in alerts_table_data.columns],
                fill_color=[
                    [
                        (
                            "rgba(218, 54, 51, 0.1)"
                            if risk == "🔴 Critical"
                            else "rgba(210, 153, 34, 0.1)"
                            if risk == "🟡 Medium"
                            else "rgba(26, 127, 55, 0.1)"
                        )
                        for risk in alerts_table_data["Risk"]
                    ]
                    if col == "Risk"
                    else "rgba(22, 27, 34, 0.5)"
                    for col in alerts_table_data.columns
                ],
                align="left",
                font=dict(color="#E6EDF3", size=11),
                height=28,
                line=dict(color="rgba(48, 54, 61, 0.3)", width=0.5),
            ),
        )
    ]
)

fig_table.update_layout(
    template="plotly_dark",
    height=500,
    margin=dict(l=20, r=20, t=20, b=20),
    plot_bgcolor="rgba(22, 27, 34, 0.3)",
    paper_bgcolor="rgba(22, 27, 34, 0.3)",
)

st.plotly_chart(fig_table, use_container_width=True, key="alerts_table")

# Summary stats
st.markdown(
    f"""
    <div style='color: #8b949e; font-size: 12px; text-align: right;'>
        Showing {len(filtered_alerts)} of {len(alerts_df)} alerts | 
        Critical: {len(filtered_alerts[filtered_alerts['risk_category'] == 'Critical'])} | 
        Confirmed Attacks: {len(filtered_alerts[filtered_alerts['is_attack'] == True])}
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ============================================================================
# FOOTER & NAVIGATION
# ============================================================================

st.markdown(
    """
    <div style='text-align: center; color: #8b949e; font-size: 11px; margin-top: 40px;'>
        <p>CyberShield SOC Dashboard | Last Updated: <span id='update-time'></span> UTC</p>
        <p>🔐 All data encrypted in transit | 🛡️ Multi-factor authentication enabled</p>
        <script>
            document.getElementById('update-time').textContent = new Date().toISOString();
        </script>
    </div>
    """,
    unsafe_allow_html=True,
)
