"""
dashboard_common.py
-------------------
Shared utilities for the SOC dashboard.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

PAGE_ICON = "🛡️"

RISK_COLORS = {
    "Low": "#1a7f37",      # Green
    "Medium": "#d29922",    # Orange
    "Critical": "#da3633",  # Red
}

@st.cache_data(ttl=300)
def get_dataset():
    """
    Load unified threat dataset from module3/data/unified_threat_data.csv
    Returns empty DataFrame if file doesn't exist.
    """
    data_path = Path(__file__).parent.parent / "data" / "unified_threat_data.csv"
    
    if not data_path.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(data_path, parse_dates=["timestamp"])
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()
