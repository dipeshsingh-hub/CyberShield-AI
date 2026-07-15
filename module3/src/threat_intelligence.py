"""
threat_intelligence.py
======================
Comprehensive Threat Intelligence Dashboard for CyberShield SOC

Features:
  - Top CVEs with CVSS scores
  - MITRE ATT&CK matrix visualization
  - Threat actor intelligence
  - Malware family tracking
  - Indicators of Compromise (IoCs)
  - Interactive world map with attack origins
  - Country attack heatmap
  - Risk evolution timeline
  - Expandable cards with professional SOC styling
  - All visualizations powered by Plotly

Run:
    cd module3/src && streamlit run threat_intelligence.py --theme.base dark
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

# ============================================================================
# CONFIGURATION & THEMING
# ============================================================================

ST_THEME = {
    "primaryColor": "#00D9FF",
    "backgroundColor": "#0D1117",
    "secondaryBackgroundColor": "#161B22",
    "textColor": "#E6EDF3",
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

CVSS_COLORS = {
    "Critical": "#DA3633",
    "High": "#D29922",
    "Medium": "#0969DA",
    "Low": "#1a7f37",
}

MITRE_TACTICS = [
    "Reconnaissance",
    "Resource Development",
    "Initial Access",
    "Execution",
    "Persistence",
    "Privilege Escalation",
    "Defense Evasion",
    "Credential Access",
    "Discovery",
    "Lateral Movement",
    "Collection",
    "Command & Control",
    "Exfiltration",
    "Impact",
]

# ============================================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Threat Intelligence Dashboard",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional SOC CSS
st.markdown(
    """
    <style>
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
        
        .threat-card {
            background: rgba(22, 27, 34, 0.7);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(48, 54, 61, 0.5);
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .threat-card:hover {
            background: rgba(22, 27, 34, 0.9);
            border-color: rgba(0, 217, 255, 0.5);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 217, 255, 0.15);
        }
        
        .threat-card-critical {
            border-left: 4px solid #DA3633;
        }
        
        .threat-card-high {
            border-left: 4px solid #D29922;
        }
        
        .threat-card-medium {
            border-left: 4px solid #0969DA;
        }
        
        .threat-title {
            font-size: 18px;
            font-weight: 700;
            color: #E6EDF3;
            margin-bottom: 8px;
        }
        
        .threat-meta {
            display: flex;
            gap: 16px;
            font-size: 12px;
            color: #8b949e;
            margin-bottom: 12px;
            flex-wrap: wrap;
        }
        
        .threat-meta-item {
            display: flex;
            align-items: center;
            gap: 4px;
        }
        
        .severity-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .severity-critical {
            background: rgba(218, 54, 51, 0.2);
            color: #DA3633;
        }
        
        .severity-high {
            background: rgba(210, 153, 34, 0.2);
            color: #D29922;
        }
        
        .severity-medium {
            background: rgba(9, 105, 218, 0.2);
            color: #0969DA;
        }
        
        .severity-low {
            background: rgba(26, 127, 55, 0.2);
            color: #1a7f37;
        }
        
        .ioc-table {
            background: rgba(22, 27, 34, 0.5);
            border-radius: 8px;
            padding: 16px;
            margin: 10px 0;
        }
        
        .ioc-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(48, 54, 61, 0.3);
            font-size: 13px;
        }
        
        .ioc-row:last-child {
            border-bottom: none;
        }
        
        .ioc-type {
            background: rgba(0, 217, 255, 0.1);
            color: #00D9FF;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 11px;
            min-width: 60px;
            text-align: center;
        }
        
        .ioc-value {
            color: #E6EDF3;
            font-family: 'Courier New', monospace;
            flex: 1;
            margin: 0 12px;
            word-break: break-all;
        }
        
        .ioc-severity {
            font-weight: 600;
            text-align: right;
            min-width: 100px;
        }
        
        .expandable-section {
            background: rgba(22, 27, 34, 0.7);
            border: 1px solid rgba(0, 217, 255, 0.1);
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 16px;
        }
        
        .expandable-header {
            padding: 16px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            user-select: none;
            transition: all 0.3s ease;
        }
        
        .expandable-header:hover {
            background: rgba(0, 217, 255, 0.05);
        }
        
        .expandable-content {
            padding: 16px;
            border-top: 1px solid rgba(48, 54, 61, 0.3);
            display: none;
        }
        
        .expandable-content.expanded {
            display: block;
        }
        
        .stat-label {
            color: #8b949e;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .stat-value {
            color: #00D9FF;
            font-size: 28px;
            font-weight: 700;
            margin-top: 4px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# SAMPLE DATA GENERATION
# ============================================================================

@st.cache_data(ttl=3600)
def generate_cve_data():
    """Generate sample CVE data"""
    cves = [
        {
            "CVE ID": "CVE-2024-1234",
            "Title": "Critical RCE in Apache Log4j",
            "CVSS Score": 9.8,
            "Severity": "Critical",
            "Published": "2024-06-15",
            "Exploits": 1247,
            "Affected Systems": 45000,
            "Description": "Remote code execution vulnerability in Apache Log4j logging library allowing unauthenticated attackers to execute arbitrary code."
        },
        {
            "CVE ID": "CVE-2024-5678",
            "Title": "SQL Injection in PostgreSQL",
            "CVSS Score": 8.6,
            "Severity": "High",
            "Published": "2024-06-10",
            "Exploits": 342,
            "Affected Systems": 12000,
            "Description": "SQL injection vulnerability in PostgreSQL query parser leading to unauthorized data access."
        },
        {
            "CVE ID": "CVE-2024-9012",
            "Title": "Authentication Bypass in OAuth 2.0",
            "CVSS Score": 7.5,
            "Severity": "High",
            "Published": "2024-05-28",
            "Exploits": 89,
            "Affected Systems": 8500,
            "Description": "Authentication bypass in OAuth 2.0 implementation allowing attackers to gain unauthorized access."
        },
        {
            "CVE ID": "CVE-2024-3456",
            "Title": "XSS in Django Templates",
            "CVSS Score": 6.1,
            "Severity": "Medium",
            "Published": "2024-05-15",
            "Exploits": 23,
            "Affected Systems": 3200,
            "Description": "Cross-site scripting vulnerability in Django template rendering engine."
        },
        {
            "CVE ID": "CVE-2024-7890",
            "Title": "Path Traversal in Nginx",
            "CVSS Score": 5.3,
            "Severity": "Medium",
            "Published": "2024-05-01",
            "Exploits": 12,
            "Affected Systems": 1800,
            "Description": "Path traversal vulnerability allowing access to sensitive files outside intended directory."
        },
    ]
    return pd.DataFrame(cves)

@st.cache_data(ttl=3600)
def generate_threat_actors():
    """Generate sample threat actor data"""
    actors = [
        {
            "Name": "APT-28 (Fancy Bear)",
            "Country": "Russia",
            "Active Since": 2007,
            "Targets": "Government, Military, Defense",
            "Techniques": 48,
            "Last Seen": "2024-07-10",
            "Aliases": "Sofacy, STRONTIUM",
            "Confidence": 98
        },
        {
            "Name": "APT-29 (Cozy Bear)",
            "Country": "Russia",
            "Active Since": 2008,
            "Targets": "Government, Energy, Telecom",
            "Techniques": 52,
            "Last Seen": "2024-07-12",
            "Aliases": "Duke, The Dukes",
            "Confidence": 96
        },
        {
            "Name": "APT-41 (Winnti)",
            "Country": "China",
            "Active Since": 2012,
            "Targets": "Technology, Healthcare, Finance",
            "Techniques": 44,
            "Last Seen": "2024-07-08",
            "Aliases": "Blackfly, Wicked Panda",
            "Confidence": 94
        },
        {
            "Name": "Lazarus Group",
            "Country": "North Korea",
            "Active Since": 2009,
            "Targets": "Financial, Government, Media",
            "Techniques": 39,
            "Last Seen": "2024-07-05",
            "Aliases": "HIDDEN COBRA",
            "Confidence": 92
        },
    ]
    return pd.DataFrame(actors)

@st.cache_data(ttl=3600)
def generate_malware_families():
    """Generate sample malware family data"""
    malware = [
        {
            "Family": "Emotet",
            "Type": "Banking Trojan",
            "First Seen": 2014,
            "Variants": 847,
            "Infections": 156000,
            "Last Detected": "2024-07-12",
            "Threat Level": "Critical",
            "Attribution": "TA542"
        },
        {
            "Family": "Mirai",
            "Type": "Botnet",
            "First Seen": 2016,
            "Variants": 1243,
            "Infections": 89000,
            "Last Detected": "2024-07-10",
            "Threat Level": "Critical",
            "Attribution": "Unknown"
        },
        {
            "Family": "Wannacry",
            "Type": "Ransomware",
            "First Seen": 2017,
            "Variants": 421,
            "Infections": 230000,
            "Last Detected": "2024-07-11",
            "Threat Level": "Critical",
            "Attribution": "Lazarus Group"
        },
        {
            "Family": "Trickbot",
            "Type": "Banking Trojan",
            "First Seen": 2016,
            "Variants": 612,
            "Infections": 45000,
            "Last Detected": "2024-07-09",
            "Threat Level": "High",
            "Attribution": "TA505"
        },
    ]
    return pd.DataFrame(malware)

@st.cache_data(ttl=3600)
def generate_iocs():
    """Generate sample Indicators of Compromise"""
    iocs = [
        {
            "Type": "IP",
            "Value": "192.168.1.100",
            "Severity": "Critical",
            "First Seen": "2024-07-10 14:32:00",
            "Last Seen": "2024-07-15 09:22:00",
            "Associated Malware": "Emotet",
            "Associated Actor": "TA542",
            "Detections": 1247
        },
        {
            "Type": "Domain",
            "Value": "evil-command-center.ru",
            "Severity": "Critical",
            "First Seen": "2024-07-08 08:15:00",
            "Last Seen": "2024-07-15 10:45:00",
            "Associated Malware": "Trickbot",
            "Associated Actor": "TA505",
            "Detections": 3421
        },
        {
            "Type": "URL",
            "Value": "hxxps://malware.net/payload/botnet.exe",
            "Severity": "Critical",
            "First Seen": "2024-07-12 16:20:00",
            "Last Seen": "2024-07-15 08:30:00",
            "Associated Malware": "Mirai",
            "Associated Actor": "Unknown",
            "Detections": 892
        },
        {
            "Type": "File Hash",
            "Value": "5d41402abc4b2a76b9719d911017c592",
            "Severity": "High",
            "First Seen": "2024-07-05 12:10:00",
            "Last Seen": "2024-07-15 07:15:00",
            "Associated Malware": "Wannacry",
            "Associated Actor": "Lazarus Group",
            "Detections": 2156
        },
        {
            "Type": "Email",
            "Value": "attacker@phishing-domain.com",
            "Severity": "High",
            "First Seen": "2024-07-11 09:45:00",
            "Last Seen": "2024-07-15 11:20:00",
            "Associated Malware": "Emotet",
            "Associated Actor": "TA542",
            "Detections": 543
        },
    ]
    return pd.DataFrame(iocs)

@st.cache_data(ttl=3600)
def generate_attack_origins():
    """Generate sample attack origin data"""
    origins = [
        {"Country": "Russia", "Attacks": 2847, "Code": "RU", "Lat": 61.52, "Lon": 105.32},
        {"Country": "China", "Attacks": 1923, "Code": "CN", "Lat": 35.86, "Lon": 104.20},
        {"Country": "Iran", "Attacks": 1456, "Code": "IR", "Lat": 32.43, "Lon": 53.69},
        {"Country": "North Korea", "Attacks": 892, "Code": "KP", "Lat": 40.34, "Lon": 127.10},
        {"Country": "India", "Attacks": 756, "Code": "IN", "Lat": 20.59, "Lon": 78.96},
        {"Country": "Brazil", "Attacks": 634, "Code": "BR", "Lat": -14.24, "Lon": -51.93},
        {"Country": "Romania", "Attacks": 521, "Code": "RO", "Lat": 45.94, "Lon": 24.97},
        {"Country": "Vietnam", "Attacks": 445, "Code": "VN", "Lat": 14.06, "Lon": 108.28},
    ]
    return pd.DataFrame(origins)

@st.cache_data(ttl=3600)
def generate_risk_evolution():
    """Generate risk evolution timeline data"""
    dates = pd.date_range(start="2024-06-01", end="2024-07-15", freq="D")
    risk_scores = np.random.randint(45, 85, size=len(dates))
    threat_count = np.cumsum(np.random.randint(1, 10, size=len(dates)))
    
    return pd.DataFrame({
        "Date": dates,
        "Risk Score": risk_scores,
        "Threat Count": threat_count,
        "Incidents": np.random.randint(0, 20, size=len(dates)),
    })

# ============================================================================
# PAGE LAYOUT
# ============================================================================

st.markdown("<div style='text-align: center; margin-bottom: 30px;'>", unsafe_allow_html=True)
st.markdown(
    "<h1 style='color: #00D9FF; font-size: 48px; margin-bottom: 10px;'>🔴 Threat Intelligence Hub</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color: #8b949e; font-size: 16px;'>Real-time threat intelligence, CVE tracking, and malware analysis</p>",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Quick stats
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🔴 Critical CVEs", "5", "+2 this week")
with col2:
    st.metric("🎯 Active Threats", "847", "-12 trending")
with col3:
    st.metric("📊 Threat Actors", "124", "+3 this month")
with col4:
    st.metric("⚠️ Risk Score", "72%", "+5%")

st.divider()

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔴 Top CVEs",
    "🗺️ MITRE ATT&CK",
    "🎯 Threat Actors",
    "🦠 Malware",
    "🔍 IoCs",
    "📈 Timeline"
])

# ============================================================================
# TAB 1: TOP CVEs
# ============================================================================

with tab1:
    st.subheader("🔴 Critical Vulnerabilities")
    
    cve_df = generate_cve_data()
    
    # CVE Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        critical_count = len(cve_df[cve_df["Severity"] == "Critical"])
        st.metric("Critical CVEs", critical_count)
    with col2:
        high_count = len(cve_df[cve_df["Severity"] == "High"])
        st.metric("High Severity", high_count)
    with col3:
        total_exploits = cve_df["Exploits"].sum()
        st.metric("Known Exploits", total_exploits)
    
    st.divider()
    
    # CVE Distribution Chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### CVSS Score Distribution")
        fig_cvss = go.Figure()
        fig_cvss.add_trace(go.Bar(
            y=cve_df["CVE ID"],
            x=cve_df["CVSS Score"],
            orientation='h',
            marker=dict(
                color=cve_df["CVSS Score"],
                colorscale=[[0, "#1a7f37"], [0.5, "#D29922"], [1, "#DA3633"]],
                showscale=False
            ),
            text=cve_df["CVSS Score"],
            textposition='auto',
        ))
        fig_cvss.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="CVSS Score",
            plot_bgcolor="rgba(22, 27, 34, 0.5)",
            paper_bgcolor="rgba(22, 27, 34, 0.3)",
        )
        st.plotly_chart(fig_cvss, use_container_width=True)
    
    with col2:
        st.markdown("#### Affected Systems")
        fig_affected = go.Figure()
        fig_affected.add_trace(go.Bar(
            x=cve_df["CVE ID"],
            y=cve_df["Affected Systems"],
            marker=dict(color="#DA3633"),
            text=cve_df["Affected Systems"],
            textposition='auto',
        ))
        fig_affected.update_layout(
            template="plotly_dark",
            height=400,
            yaxis_title="Systems",
            plot_bgcolor="rgba(22, 27, 34, 0.5)",
            paper_bgcolor="rgba(22, 27, 34, 0.3)",
        )
        st.plotly_chart(fig_affected, use_container_width=True)
    
    st.divider()
    
    # Expandable CVE Cards
    st.markdown("#### Detailed CVE Information")
    for idx, row in cve_df.iterrows():
        severity_class = f"threat-card-{row['Severity'].lower()}"
        severity_badge = f"severity-{row['Severity'].lower()}"
        
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.markdown(
                    f"""
                    <div class="threat-card {severity_class}">
                        <div class="threat-title">{row['CVE ID']} - {row['Title']}</div>
                        <div class="threat-meta">
                            <div class="threat-meta-item">📅 {row['Published']}</div>
                            <div class="threat-meta-item">📊 CVSS: {row['CVSS Score']}</div>
                            <div class="threat-meta-item">⚔️ Exploits: {row['Exploits']}</div>
                            <div class="threat-meta-item">💻 Systems: {row['Affected Systems']:,}</div>
                        </div>
                        <div><span class="severity-badge {severity_badge}">{row['Severity']}</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            with col2:
                if st.button("📖", key=f"cve-{idx}", help="View Details"):
                    st.info(row["Description"])

# ============================================================================
# TAB 2: MITRE ATT&CK MATRIX
# ============================================================================

with tab2:
    st.subheader("🗺️ MITRE ATT&CK Framework")
    
    # Generate sample technique data
    np.random.seed(42)
    techniques_data = []
    for tactic in MITRE_TACTICS:
        techniques_data.append({
            "Tactic": tactic,
            "Techniques": np.random.randint(5, 20),
            "Observed": np.random.randint(0, 100),
        })
    
    techniques_df = pd.DataFrame(techniques_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Techniques by Tactic")
        fig_matrix = px.bar(
            techniques_df,
            x="Tactic",
            y="Techniques",
            color="Observed",
            color_continuous_scale=[[0, "#1a7f37"], [0.5, "#D29922"], [1, "#DA3633"]],
            title=""
        )
        fig_matrix.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_tickangle=-45,
            plot_bgcolor="rgba(22, 27, 34, 0.5)",
            paper_bgcolor="rgba(22, 27, 34, 0.3)",
        )
        st.plotly_chart(fig_matrix, use_container_width=True)
    
    with col2:
        st.markdown("#### Tactic Overview")
        fig_sunburst = px.sunburst(
            techniques_df,
            names="Tactic",
            values="Techniques",
            color="Observed",
            color_continuous_scale=[[0, "#00D9FF"], [1, "#DA3633"]],
        )
        fig_sunburst.update_layout(
            template="plotly_dark",
            height=500,
            plot_bgcolor="rgba(22, 27, 34, 0.5)",
            paper_bgcolor="rgba(22, 27, 34, 0.3)",
        )
        st.plotly_chart(fig_sunburst, use_container_width=True)
    
    # MITRE Tactics Table
    st.divider()
    st.markdown("#### MITRE ATT&CK Tactics Table")
    
    for idx, row in techniques_df.iterrows():
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"**{row['Tactic']}**")
        with col2:
            st.metric("Techniques", row['Techniques'])
        with col3:
            st.metric("Observed", f"{row['Observed']}%")

# ============================================================================
# TAB 3: THREAT ACTORS
# ============================================================================

with tab3:
    st.subheader("🎯 Tracked Threat Actors")
    
    actors_df = generate_threat_actors()
    
    # Actor Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Total Actors", len(actors_df))
    with col2:
        st.metric("🌍 Countries", actors_df["Country"].nunique())
    with col3:
        st.metric("📊 Avg Techniques", f"{actors_df['Techniques'].mean():.0f}")
    with col4:
        st.metric("🔥 Confidence", f"{actors_df['Confidence'].mean():.0f}%")
    
    st.divider()
    
    # Actor visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Techniques by Actor")
        fig_actor_tech = px.bar(
            actors_df.sort_values("Techniques"),
            x="Techniques",
            y="Name",
            orientation="h",
            color="Confidence",
            color_continuous_scale=[[0, "#1a7f37"], [0.5, "#D29922"], [1, "#DA3633"]],
        )
        fig_actor_tech.update_layout(
            template="plotly_dark",
            height=400,
            plot_bgcolor="rgba(22, 27, 34, 0.5)",
            paper_bgcolor="rgba(22, 27, 34, 0.3)",
        )
        st.plotly_chart(fig_actor_tech, use_container_width=True)
    
    with col2:
        st.markdown("#### Actor Activity Timeline")
        fig_timeline = px.scatter(
            actors_df,
            x="Active Since",
            y="Confidence",
            size="Techniques",
            color="Country",
            hover_data={"Name": True},
            title=""
        )
        fig_timeline.update_layout(
            template="plotly_dark",
            height=400,
            plot_bgcolor="rgba(22, 27, 34, 0.5)",
            paper_bgcolor="rgba(22, 27, 34, 0.3)",
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    st.divider()
    
    # Expandable Actor Cards
    st.markdown("#### Threat Actor Profiles")
    for idx, row in actors_df.iterrows():
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.markdown(
                    f"""
                    <div class="threat-card threat-card-high">
                        <div class="threat-title">{row['Name']}</div>
                        <div class="threat-meta">
                            <div class="threat-meta-item">🌍 {row['Country']}</div>
                            <div class="threat-meta-item">📅 Since {row['Active Since']}</div>
                            <div class="threat-meta-item">📊 Techniques: {row['Techniques']}</div>
                            <div class="threat-meta-item">✓ Confidence: {row['Confidence']}%</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            with col2:
                if st.button("📋", key=f"actor-{idx}", help="Details"):
                    st.write(f"**Aliases:** {row['Aliases']}")
                    st.write(f"**Targets:** {row['Targets']}")
                    st.write(f"**Last Seen:** {row['Last Seen']}")

# ============================================================================
# TAB 4: MALWARE FAMILIES
# ============================================================================

with tab4:
    st.subheader("🦠 Tracked Malware Families")
    
    malware_df = generate_malware_families()
    
    # Malware Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🦠 Families", len(malware_df))
    with col2:
        st.metric("⚠️ Total Variants", malware_df["Variants"].sum())
    with col3:
        st.metric("📊 Total Infections", f"{malware_df['Infections'].sum():,}")
    with col4:
        st.metric("🔴 Critical", len(malware_df[malware_df["Threat Level"] == "Critical"]))
    
    st.divider()
    
    # Malware visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Infections by Family")
        fig_infections = go.Figure(
            data=[go.Pie(
                labels=malware_df["Family"],
                values=malware_df["Infections"],
                marker=dict(colors=["#DA3633", "#D29922", "#0969DA", "#00D9FF"]),
            )]
        )
        fig_infections.update_layout(
            template="plotly_dark",
            height=400,
            plot_bgcolor="rgba(22, 27, 34, 0.5)",
            paper_bgcolor="rgba(22, 27, 34, 0.3)",
        )
        st.plotly_chart(fig_infections, use_container_width=True)
    
    with col2:
        st.markdown("#### Variants Distribution")
        fig_variants = px.bar(
            malware_df.sort_values("Variants", ascending=True),
            x="Variants",
            y="Family",
            orientation="h",
            color="Threat Level",
            color_discrete_map={"Critical": "#DA3633", "High": "#D29922"},
        )
        fig_variants.update_layout(
            template="plotly_dark",
            height=400,
            plot_bgcolor="rgba(22, 27, 34, 0.5)",
            paper_bgcolor="rgba(22, 27, 34, 0.3)",
        )
        st.plotly_chart(fig_variants, use_container_width=True)
    
    st.divider()
    
    # Malware details table
    st.markdown("#### Malware Family Details")
    for idx, row in malware_df.iterrows():
        severity_class = "threat-card-critical" if row["Threat Level"] == "Critical" else "threat-card-high"
        
        with st.container():
            st.markdown(
                f"""
                <div class="threat-card {severity_class}">
                    <div class="threat-title">{row['Family']}</div>
                    <div class="threat-meta">
                        <div class="threat-meta-item">🔧 Type: {row['Type']}</div>
                        <div class="threat-meta-item">📅 Since: {row['First Seen']}</div>
                        <div class="threat-meta-item">⚠️ Variants: {row['Variants']}</div>
                        <div class="threat-meta-item">📊 Infections: {row['Infections']:,}</div>
                        <div class="threat-meta-item">👥 Attribution: {row['Attribution']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ============================================================================
# TAB 5: INDICATORS OF COMPROMISE
# ============================================================================

with tab5:
    st.subheader("🔍 Indicators of Compromise (IoCs)")
    
    ioc_df = generate_iocs()
    
    # IoC Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔍 Total IoCs", len(ioc_df))
    with col2:
        st.metric("🔴 Critical", len(ioc_df[ioc_df["Severity"] == "Critical"]))
    with col3:
        st.metric("📊 Total Detections", ioc_df["Detections"].sum())
    with col4:
        st.metric("🦠 Malware Linked", ioc_df["Associated Malware"].nunique())
    
    st.divider()
    
    # IoC Type Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### IoC Types")
        type_counts = ioc_df["Type"].value_counts()
        fig_types = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            color_discrete_sequence=["#00D9FF", "#0969DA", "#D29922", "#DA3633", "#1a7f37"],
        )
        fig_types.update_layout(
            template="plotly_dark",
            height=400,
            plot_bgcolor="rgba(22, 27, 34, 0.5)",
            paper_bgcolor="rgba(22, 27, 34, 0.3)",
        )
        st.plotly_chart(fig_types, use_container_width=True)
    
    with col2:
        st.markdown("#### Detection Activity")
        fig_detections = px.bar(
            ioc_df.sort_values("Detections"),
            x="Detections",
            y="Value",
            orientation="h",
            color="Severity",
            color_discrete_map={"Critical": "#DA3633", "High": "#D29922"},
        )
        fig_detections.update_layout(
            template="plotly_dark",
            height=400,
            plot_bgcolor="rgba(22, 27, 34, 0.5)",
            paper_bgcolor="rgba(22, 27, 34, 0.3)",
            showlegend=False,
        )
        st.plotly_chart(fig_detections, use_container_width=True)
    
    st.divider()
    
    # Detailed IoC List
    st.markdown("#### IOC Details")
    
    for idx, row in ioc_df.iterrows():
        severity_class = "threat-card-critical" if row["Severity"] == "Critical" else "threat-card-high"
        severity_badge = f"severity-{row['Severity'].lower()}"
        
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.markdown(
                    f"""
                    <div class="threat-card {severity_class}">
                        <div class="threat-title">IoC: {row['Type']}</div>
                        <div class="threat-meta">
                            <div class="threat-meta-item">🔗 {row['Value']}</div>
                            <div class="threat-meta-item">📊 Detections: {row['Detections']}</div>
                            <div class="threat-meta-item">🦠 {row['Associated Malware']}</div>
                            <div class="threat-meta-item">👥 {row['Associated Actor']}</div>
                        </div>
                        <div><span class="severity-badge {severity_badge}">{row['Severity']}</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            with col2:
                if st.button("⏱️", key=f"ioc-{idx}", help="Timeline"):
                    st.write(f"First Seen: {row['First Seen']}")
                    st.write(f"Last Seen: {row['Last Seen']}")

# ============================================================================
# TAB 6: TIMELINE & RISK EVOLUTION
# ============================================================================

with tab6:
    st.subheader("📈 Risk Evolution Timeline")
    
    timeline_df = generate_risk_evolution()
    
    # Timeline statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Avg Risk", f"{timeline_df['Risk Score'].mean():.0f}%")
    with col2:
        st.metric("📈 Peak Risk", f"{timeline_df['Risk Score'].max():.0f}%")
    with col3:
        st.metric("🎯 Total Threats", timeline_df["Threat Count"].iloc[-1])
    with col4:
        st.metric("⚠️ Incidents", timeline_df["Incidents"].sum())
    
    st.divider()
    
    # Risk Evolution Chart
    st.markdown("#### Risk Score Evolution")
    fig_evolution = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Risk Score Over Time", "Incident Activity"),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    fig_evolution.add_trace(
        go.Scatter(
            x=timeline_df["Date"],
            y=timeline_df["Risk Score"],
            fill="tozeroy",
            name="Risk Score",
            line=dict(color="#DA3633", width=3),
            fillcolor="rgba(218, 54, 51, 0.2)",
        ),
        row=1, col=1
    )
    
    fig_evolution.add_trace(
        go.Bar(
            x=timeline_df["Date"],
            y=timeline_df["Incidents"],
            name="Incidents",
            marker=dict(color="#D29922"),
        ),
        row=2, col=1
    )
    
    fig_evolution.update_yaxes(title_text="Risk %", row=1, col=1)
    fig_evolution.update_yaxes(title_text="Count", row=2, col=1)
    fig_evolution.update_xaxes(title_text="Date", row=2, col=1)
    
    fig_evolution.update_layout(
        template="plotly_dark",
        height=600,
        plot_bgcolor="rgba(22, 27, 34, 0.5)",
        paper_bgcolor="rgba(22, 27, 34, 0.3)",
        hovermode="x unified",
    )
    
    st.plotly_chart(fig_evolution, use_container_width=True)
    
    st.divider()
    
    # Attack Origins Map
    st.markdown("#### Global Attack Origins")
    
    origins_df = generate_attack_origins()
    
    fig_map = go.Figure(
        data=go.Scattergeo(
            lon=origins_df["Lon"],
            lat=origins_df["Lat"],
            mode="markers+text",
            marker=dict(
                size=origins_df["Attacks"] / 100,
                color=origins_df["Attacks"],
                colorscale=[[0, "#1a7f37"], [0.5, "#D29922"], [1, "#DA3633"]],
                showscale=True,
                colorbar=dict(title="Attacks"),
                line=dict(width=1, color="rgba(0, 217, 255, 0.5)"),
            ),
            text=origins_df["Country"],
            textposition="top center",
            hovertemplate="<b>%{text}</b><br>Attacks: %{marker.color}<extra></extra>",
        )
    )
    
    fig_map.update_layout(
        geo=dict(
            projection_type="natural earth",
            bgcolor="rgba(13, 17, 23, 0.9)",
            showland=True,
            landcolor="rgba(22, 27, 34, 0.5)",
            showocean=True,
            oceancolor="rgba(13, 17, 23, 0.8)",
            coastcolor="rgba(48, 54, 61, 0.5)",
        ),
        template="plotly_dark",
        height=500,
        paper_bgcolor="rgba(22, 27, 34, 0.3)",
    )
    
    st.plotly_chart(fig_map, use_container_width=True)
    
    st.divider()
    
    # Country statistics
    st.markdown("#### Top Attack Origins")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_countries = px.bar(
            origins_df.sort_values("Attacks"),
            x="Attacks",
            y="Country",
            orientation="h",
            color="Attacks",
            color_continuous_scale=[[0, "#1a7f37"], [0.5, "#D29922"], [1, "#DA3633"]],
        )
        fig_countries.update_layout(
            template="plotly_dark",
            height=400,
            plot_bgcolor="rgba(22, 27, 34, 0.5)",
            paper_bgcolor="rgba(22, 27, 34, 0.3)",
            showlegend=False,
        )
        st.plotly_chart(fig_countries, use_container_width=True)
    
    with col2:
        st.markdown("#### Threat Cumulative")
        fig_cumulative = go.Figure()
        fig_cumulative.add_trace(go.Scatter(
            x=timeline_df["Date"],
            y=timeline_df["Threat Count"],
            fill="tozeroy",
            name="Cumulative Threats",
            line=dict(color="#00D9FF", width=3),
            fillcolor="rgba(0, 217, 255, 0.2)",
        ))
        fig_cumulative.update_layout(
            template="plotly_dark",
            height=400,
            plot_bgcolor="rgba(22, 27, 34, 0.5)",
            paper_bgcolor="rgba(22, 27, 34, 0.3)",
            xaxis_title="Date",
            yaxis_title="Cumulative Count",
        )
        st.plotly_chart(fig_cumulative, use_container_width=True)

st.divider()
st.markdown(
    "<div style='text-align: center; color: #8b949e; font-size: 12px; padding: 20px;'>"
    "🔴 CyberShield AI - Threat Intelligence Hub | Last Updated: 2024-07-15 10:30 UTC"
    "</div>",
    unsafe_allow_html=True,
)
