"""
dashboard_common.py
---------------------
Streamlit-specific caching layer shared by app.py and pages/*.py. Wraps
xai_engine's functions with @st.cache_data / @st.cache_resource so the
(fairly expensive) dataset load, surrogate model training, and SHAP value
computation only happen once per session, not on every widget interaction
(Streamlit reruns the whole script top-to-bottom on every interaction).
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import xai_engine as xe

PAGE_ICON = "🛡️"


@st.cache_data(show_spinner="Loading unified threat dataset...")
def get_dataset():
    return xe.load_dataset()


@st.cache_resource(show_spinner="Training XAI surrogate model + computing SHAP values...")
def get_shap_ready():
    """Triggers xai_engine's internal surrogate/SHAP loading once, cached process-wide."""
    xe._ensure_surrogate_loaded()
    return True


@st.cache_resource(show_spinner="Loading Module 2's phishing model for LIME...")
def get_lime_ready():
    xe._ensure_lime_loaded()
    return True


@st.cache_data(show_spinner=False)
def cached_global_feature_importance():
    get_shap_ready()
    return xe.get_global_feature_importance()


@st.cache_data(show_spinner=False)
def cached_local_shap(row_index: int):
    get_shap_ready()
    return xe.get_local_shap_values(row_index)


@st.cache_data(show_spinner=False)
def cached_dependence_data(feature: str):
    get_shap_ready()
    return xe.get_dependence_data(feature)


@st.cache_data(show_spinner="Computing LIME explanation (perturbing text locally)...")
def cached_lime_explanation(row_index: int):
    get_lime_ready()
    return xe.get_lime_explanation(row_index)


@st.cache_data(show_spinner=False)
def cached_xai_report(row_index=None):
    get_shap_ready()
    return xe.generate_xai_report(row_index)


def shap_base_value():
    get_shap_ready()
    return xe.get_shap_base_value()


RISK_COLORS = {"Low": "#2ECC71", "Medium": "#F39C12", "Critical": "#E74C3C"}
