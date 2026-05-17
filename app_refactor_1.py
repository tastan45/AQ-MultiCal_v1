import streamlit as st
import io
import pandas as pd
import time
import re
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.base import clone
from copy import deepcopy

# NEW IMPORTS FOR ADDITIONAL MODELS
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# --- NEW IMPORTS FOR BAYESIAN OPTIMIZATION ---
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    _SKOPT_AVAILABLE = True
except ImportError:
    _SKOPT_AVAILABLE = False
# --- END OF NEW IMPORTS ---

# Scipy is imported only for KDE plotting, otherwise it won't throw an error
try:
    import scipy.stats as stats
    _SCIPY_AVAILABLE = True
    from scipy.stats import loguniform, uniform, randint
except ImportError:
    _SCIPY_AVAILABLE = False

# --- NEW IMPORTS FOR SEABORN PLOTTING ---
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    _SEABORN_AVAILABLE = True
except ImportError:
    _SEABORN_AVAILABLE = False
# --- END OF NEW IMPORTS ---

# --- EXISTING NEW MODEL LIBRARIES FROM YOUR ORIGINAL CODE ---
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# --- End of library additions ---

try:
    from sklearn.metrics import mean_absolute_percentage_error
except ImportError:
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        non_zero_indices = y_true != 0
        if not np.any(non_zero_indices):
            return np.nan
        y_true_filtered = y_true[non_zero_indices]
        y_pred_filtered = y_pred[non_zero_indices]
        return np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100

from config import POLLUTANT_DISPLAY_UNITS, PLOTLY_CONFIG, DEFAULT_PLOT_STYLES
from model_registry import (
    MODELS,
    COMMON_PARAM_GRID,
    RANDOM_PARAM_SPACES,
    BAYES_PARAM_SPACES,
    AUTO_OPTIMIZE_PARAMS,
    AUTO_SEARCH_PARAMS,
    EXTENDED_COMMON_PARAM_GRID,
    EXTENDED_RANDOM_PARAM_SPACES,
    EXTENDED_BAYES_PARAM_SPACES,
    ALL_OPTIMIZABLE_PARAM_NAMES,
)
from data_processing import merge_and_prepare_data
from plotting import (
    plot_dataset_distributions,
    plot_dataset_distributions_seaborn,
    plot_pair_plots,
    plot_pair_plots_seaborn,
    get_native_feature_importance_df,
    get_permutation_importance_df,
    plot_importance_bar,
    build_cross_model_importance_table,
    plot_parallel_coordinates,
    plot_residuals,
    plot_residuals_histogram,
    plot_residuals_histogram_seaborn,
    plot_residuals_kde,
    plot_residuals_kde_seaborn,
    plot_residuals_seaborn,
    plot_scatter,
    plot_scatter_seaborn,
    plot_time_series,
    plot_model_summary_2x3,
)
from training import (
    calculate_all_metrics,
    display_optimization_comparison_results,
    run_comparative_optimization_analysis,
    run_model_analysis,
    update_progress_bar_with_eta,
)

# V30.2-fixed

# --- Application Interface Settings ---
st.set_page_config(page_title="Hyperparameter Optimization for Air Quality Sensor Calibration", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS Styles (For Professional Look) ---
st.title("AQ-MultiCal: Air Quality Multi-Model Calibration")

st.markdown("""
<style>
:root {
  --accent-1: #0f4c81;
  --accent-2: #11b5ae;
  --accent-3: #ff8c42;
  --card-bg: linear-gradient(145deg, rgba(255,255,255,0.95), rgba(245,248,252,0.98));
  --soft-border: rgba(15, 76, 129, 0.10);
  --soft-shadow: 0 12px 28px rgba(15, 76, 129, 0.10);
  --soft-shadow-2: 0 8px 18px rgba(17, 181, 174, 0.08);
}

.block-container {
  padding-top: 3.0rem !important;
  padding-bottom: 2rem !important;
}

h1 {
  margin-top: 0.35rem !important;
  padding-top: 0.2rem !important;
  line-height: 1.15 !important;
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #f4f8fc 0%, #edf4fb 100%);
  border-right: 1px solid rgba(15, 76, 129, 0.08);
}

section[data-testid="stSidebar"] .stExpander {
  background: rgba(255,255,255,0.72);
  border: 1px solid rgba(15, 76, 129, 0.10);
  border-radius: 18px;
  box-shadow: var(--soft-shadow-2);
  margin-bottom: 0.8rem;
  overflow: hidden;
}

section[data-testid="stSidebar"] .stExpander summary {
  background: linear-gradient(90deg, rgba(15,76,129,0.10), rgba(17,181,174,0.08));
  border-radius: 18px;
  font-weight: 700;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
  border-radius: 14px !important;
  border: 1px solid rgba(15, 76, 129, 0.10) !important;
  background: rgba(255,255,255,0.92) !important;
  box-shadow: 0 4px 10px rgba(15, 76, 129, 0.05);
}

section[data-testid="stSidebar"] .stCheckbox,
section[data-testid="stSidebar"] .stRadio,
section[data-testid="stSidebar"] .stSlider {
  padding-top: 0.15rem;
  padding-bottom: 0.15rem;
}

section[data-testid="stSidebar"] .stSlider [role="slider"] {
  background: #0f4c81 !important;
  border: 2px solid white !important;
  box-shadow: 0 4px 10px rgba(15, 76, 129, 0.18);
}

section[data-testid="stSidebar"] [data-baseweb="radio"] > div {
  gap: 0.5rem;
}

section[data-testid="stSidebar"] label p,
section[data-testid="stSidebar"] .stMarkdown p {
  color: #31475e !important;
}

section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
  border-radius: 10px !important;
  background: linear-gradient(135deg, rgba(15,76,129,0.92), rgba(17,181,174,0.86)) !important;
  color: white !important;
  box-shadow: 0 6px 14px rgba(15, 76, 129, 0.16);
}

section[data-testid="stSidebar"] .stCaptionContainer,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
  color: #4d637a !important;
}

.stButton > button {
  border-radius: 16px !important;
  min-height: 3.25rem;
  font-weight: 700 !important;
  font-size: 1rem !important;
  border: 1px solid rgba(15, 76, 129, 0.10) !important;
  box-shadow: 0 10px 22px rgba(15, 76, 129, 0.10), inset 0 1px 0 rgba(255,255,255,0.7);
  transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
}

.stButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 14px 26px rgba(15, 76, 129, 0.14), inset 0 1px 0 rgba(255,255,255,0.72);
}

.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #0f4c81, #1569a8) !important;
  color: white !important;
  border: none !important;
}

.stButton > button[kind="secondary"] {
  background: linear-gradient(145deg, rgba(255,255,255,0.96), rgba(242,246,251,0.98)) !important;
  color: #1f3650 !important;
}

.stButton > button:disabled {
  opacity: 0.72 !important;
  transform: none !important;
  box-shadow: 0 8px 18px rgba(15, 76, 129, 0.06) !important;
}

div[data-testid="stMetric"] {
  background: var(--card-bg);
  border: 1px solid var(--soft-border);
  border-radius: 20px;
  padding: 0.7rem 0.8rem;
  box-shadow: var(--soft-shadow);
}

.element-container .stAlert, .stDataFrame, .stTabs, .stPlotlyChart {
  border-radius: 18px;
}

.dashboard-card {
  background: var(--card-bg);
  border: 1px solid var(--soft-border);
  border-radius: 22px;
  padding: 1rem 1.1rem;
  box-shadow: var(--soft-shadow);
  min-height: 122px;
  position: relative;
  overflow: hidden;
}

.dashboard-card::after {
  content: "";
  position: absolute;
  inset: auto -20px -40px auto;
  width: 120px;
  height: 120px;
  background: radial-gradient(circle, rgba(17,181,174,0.16), rgba(17,181,174,0.00) 70%);
}

.dashboard-label {
  color: #506273;
  font-size: 0.90rem;
  margin-bottom: 0.4rem;
  font-weight: 600;
}

.dashboard-value {
  color: #0f2740;
  font-size: 1.55rem;
  font-weight: 800;
  line-height: 1.15;
}

.dashboard-sub {
  color: #6a7f92;
  font-size: 0.84rem;
  margin-top: 0.45rem;
}

.app-hero {
  background: linear-gradient(135deg, rgba(15,76,129,0.10), rgba(17,181,174,0.07), rgba(255,140,66,0.08));
  border: 1px solid rgba(15, 76, 129, 0.09);
  border-radius: 24px;
  padding: 1.15rem 1.25rem;
  box-shadow: var(--soft-shadow);
  margin-bottom: 1rem;
}

.app-hero-title {
  font-size: 1.1rem;
  font-weight: 800;
  color: #17324d;
  margin-bottom: 0.3rem;
}

.app-hero-text {
  font-size: 0.95rem;
  color: #4f6275;
}

div[data-testid="stDataFrame"] {
  border: 1px solid rgba(15,76,129,0.08);
  border-radius: 18px;
  box-shadow: var(--soft-shadow-2);
  overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="app-hero">
        <div class="app-hero-title">Air quality sensor calibration workspace</div>
        <div class="app-hero-text">
            Configure calibration experiments, review evaluation results, compare model behavior, and prepare publication-ready outputs.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- NEW: COMBINED PARAMETER POOL FOR ALL OPTIMIZATION TYPES ---

# --- DEFAULT PLOT STYLES (New Global Variable with Hex Colors and Float Widths) ---

# --- Helper function for consistent progress bar updates with ETA ---
    # --- END OF MODIFICATION ---



def create_summary_table_from_history(history_records):
    rows = []
    for record in history_records or []:
        rows.append({
            "Analysis Time": record.get("Analysis Time"),
            "Model": record.get("Model Name"),
            "Optimization": record.get("Opt. Status"),
            "Optimization Mode": format_optimization_mode_label(record.get("Opt. Mode")),
            "Parameter Selection": record.get("Parameter Selection"),
            "Best Hyperparameters": str(
                record.get("Best Parameters")
                or record.get("Best Params")
                or record.get("Best Hyperparameters")
                or record.get("best_params")
                or record.get("Best_Params")
                or ""
            ),
            "Train R²": record.get("Training R²"),
            "Validation R²": record.get("Validation R²"),
            "Test R²": record.get("Test R²"),
            "Train RMSE": record.get("Training RMSE"),
            "Validation RMSE": record.get("Validation RMSE"),
            "Test RMSE": record.get("Test RMSE"),
            "Train MAE": record.get("Training MAE"),
            "Validation MAE": record.get("Validation MAE"),
            "Test MAE": record.get("Test MAE"),
            "Train MAPE": record.get("Training MAPE"),
            "Validation MAPE": record.get("Validation MAPE"),
            "Test MAPE": record.get("Test MAPE"),
            "Duration (s)": record.get("Analysis Duration (s)"),
            "Processed Rows": record.get("Processed Rows"),
            "Sampling Period": record.get("Sampling Period"),
            "Split Method": record.get("Splitting Method"),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "Test RMSE" in df.columns:
        df = df.sort_values(by=["Test RMSE", "Test MAE"], na_position="last").reset_index(drop=True)
    return df


def get_best_model_row(summary_df):
    if summary_df is None or summary_df.empty:
        return None

    valid_df = summary_df.dropna(subset=["Test RMSE"], how="any") if "Test RMSE" in summary_df.columns else summary_df
    if valid_df.empty:
        return None

    return valid_df.loc[valid_df["Test RMSE"].idxmin()]



def format_optimization_mode_label(mode):
    mapping = {
        "Manual Optimization": "Grid Search",
        "RandomizedSearchCV": "Randomized Search",
        "Bayesian Optimization (skopt)": "Bayesian Optimization",
        "Bayesian Optimization": "Bayesian Optimization",
        "None": "None",
        None: "None",
    }
    return mapping.get(mode, mode)


def render_top_kpi_cards(summary_df, best_row):
    col1, col2, col3, col4 = st.columns(4)
    total_runs = len(summary_df)
    distinct_models = summary_df["Model"].nunique() if "Model" in summary_df.columns else total_runs
    latest_model = summary_df.iloc[0]["Model"] if not summary_df.empty and "Model" in summary_df.columns else "-"
    latest_rmse = summary_df.iloc[0]["Test RMSE"] if not summary_df.empty and "Test RMSE" in summary_df.columns else np.nan

    cards = [
        ("Best Model", best_row["Model"] if best_row is not None else "-", f"Test RMSE: {best_row['Test RMSE']:.4f}" if best_row is not None else "No result yet"),
        ("Latest Run", latest_model, f"Test RMSE: {latest_rmse:.4f}" if pd.notna(latest_rmse) else "Not available"),
        ("Completed Runs", f"{total_runs}", f"Across {distinct_models} model types"),
        ("Top Test R²", f"{best_row['Test R²']:.4f}" if best_row is not None and pd.notna(best_row.get("Test R²", np.nan)) else "-", "Current best-performing score"),
    ]
    for col, (label, value, sub) in zip([col1, col2, col3, col4], cards):
        with col:
            st.markdown(
                f"""
                <div class="dashboard-card">
                    <div class="dashboard-label">{label}</div>
                    <div class="dashboard-value">{value}</div>
                    <div class="dashboard-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )



def _dataframe_to_excel_bytes(df, sheet_name="Results"):
    buffer = io.BytesIO()
    safe_sheet_name = str(sheet_name)[:31] or "Results"
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=safe_sheet_name)
    return buffer.getvalue()


def _download_table_as_csv_excel(df, base_filename, key_prefix, sheet_name="Results"):
    if df is None or df.empty:
        return
    clean_df = df.copy()

    # Do NOT drop columns such as "Opt. Param Values".
    # Only remove heavy raw array columns that should not be exported.
    drop_cols = [
        c for c in clean_df.columns
        if str(c).lower() in {
            "y_test_values",
            "y_pred_values",
            "y_train_values",
            "y_val_values",
            "y_train_pred_values",
            "y_val_pred_values",
        }
    ]
    clean_df = clean_df.drop(columns=drop_cols, errors="ignore")
    col_csv, col_xlsx = st.columns(2)
    with col_csv:
        st.download_button(
            label="📥 Download CSV",
            data=clean_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{base_filename}.csv",
            mime="text/csv",
            key=f"{key_prefix}_csv_download",
            use_container_width=True,
        )
    with col_xlsx:
        try:
            st.download_button(
                label="📥 Download Excel",
                data=_dataframe_to_excel_bytes(clean_df, sheet_name=sheet_name),
                file_name=f"{base_filename}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"{key_prefix}_xlsx_download",
                use_container_width=True,
            )
        except Exception as exc:
            st.warning(f"Excel export could not be prepared: {exc}")

def render_results_summary_panel():
    history_records = st.session_state.get("history", [])
    if not history_records:
        return

    st.header(" Results Summary Dashboard")
    summary_df = create_summary_table_from_history(history_records)
    if summary_df.empty:
        st.info("No completed analyses are available yet.")
        return

    best_row = get_best_model_row(summary_df)
    render_top_kpi_cards(summary_df, best_row)

    if best_row is not None:
        st.success(
            f" Best model so far: {best_row['Model']} | Test RMSE: {best_row['Test RMSE']:.4f} | Test R²: {best_row['Test R²']:.4f}"
        )

    st.dataframe(summary_df, use_container_width=True)

    _download_table_as_csv_excel(
        summary_df,
        base_filename="model_results_summary",
        key_prefix="model_results_summary",
        sheet_name="Model Results",
    )



def render_explainability_summary_panel():
    results_by_model = st.session_state.get("results_by_model", {})
    if not results_by_model:
        return

    comparison_df = build_cross_model_importance_table(results_by_model)
    if comparison_df is None or comparison_df.empty:
        return

    st.header(" Explainability Comparison Dashboard")
    st.caption("For tree-based models, native feature importance is used. For other compatible models, permutation importance is used. Values are normalized within each model to support comparison.")
    st.dataframe(comparison_df, use_container_width=True)

    _download_table_as_csv_excel(
        comparison_df,
        base_filename="model_explainability_comparison",
        key_prefix="model_explainability_comparison",
        sheet_name="Explainability",
    )



def display_batch_summary_results(valid_models, results_by_model):
    """Render lightweight batch results without plots/download-heavy per-model graphics."""
    st.header("Batch Analysis Results")
    st.caption("Batch mode shows comparison tables only. Detailed plots are generated only in individual model analysis to keep the app fast and avoid Streamlit media/key conflicts.")

    rows = []
    best_param_rows = []
    for model_name in valid_models:
        res = results_by_model.get(model_name)
        if not res:
            continue
        rows.append({
            "Model": res.get("model_name", model_name),
            "Optimization": "Yes" if res.get("optimized", False) else "No",
            "Optimization Mode": format_optimization_mode_label(res.get("optimization_mode", "N/A")),
            "Parameter Selection": res.get("param_scope", "N/A"),
            "Train R²": res.get("train_metrics", {}).get("r2", np.nan),
            "Validation R²": res.get("val_metrics", {}).get("r2", np.nan),
            "Test R²": res.get("test_metrics", {}).get("r2", np.nan),
            "Train RMSE": res.get("train_metrics", {}).get("rmse", np.nan),
            "Validation RMSE": res.get("val_metrics", {}).get("rmse", np.nan),
            "Test RMSE": res.get("test_metrics", {}).get("rmse", np.nan),
            "Train MAE": res.get("train_metrics", {}).get("mae", np.nan),
            "Validation MAE": res.get("val_metrics", {}).get("mae", np.nan),
            "Test MAE": res.get("test_metrics", {}).get("mae", np.nan),
            "Duration (s)": res.get("analysis_duration", np.nan),
            "Features": ", ".join(res.get("features", [])),
            "Sampling Period": res.get("interval", "N/A"),
            "Split Method": res.get("split_method", "N/A"),
        })

        best_params = res.get("best_params") or {}
        if isinstance(best_params, dict) and best_params:
            for param_name, param_value in best_params.items():
                best_param_rows.append({
                    "Model": res.get("model_name", model_name),
                    "Parameter": param_name,
                    "Best Value": str(param_value),
                })

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        st.info("No batch results are available yet.")
        return

    if "Test RMSE" in summary_df.columns:
        summary_df = summary_df.sort_values(by=["Test RMSE", "Test MAE"], na_position="last").reset_index(drop=True)

    best_row = get_best_model_row(summary_df)
    if best_row is not None:
        st.success(f"Best batch model: {best_row['Model']} | Test RMSE: {best_row['Test RMSE']:.4f} | Test R²: {best_row['Test R²']:.4f}")

    st.subheader("Model Comparison Table")
    formatter = {
        "Train R²": "{:.4f}", "Validation R²": "{:.4f}", "Test R²": "{:.4f}",
        "Train RMSE": "{:.4f}", "Validation RMSE": "{:.4f}", "Test RMSE": "{:.4f}",
        "Train MAE": "{:.4f}", "Validation MAE": "{:.4f}", "Test MAE": "{:.4f}",
        "Duration (s)": "{:.2f}",
    }
    st.dataframe(summary_df.style.format(formatter), use_container_width=True)
    _download_table_as_csv_excel(
        summary_df,
        base_filename="batch_model_comparison_results",
        key_prefix="batch_model_comparison_results",
        sheet_name="Batch Results",
    )

    if best_param_rows:
        st.subheader("Best Hyperparameters by Model")
        best_params_df = pd.DataFrame(best_param_rows)
        st.dataframe(best_params_df, use_container_width=True)
        _download_table_as_csv_excel(
            best_params_df,
            base_filename="batch_best_hyperparameters",
            key_prefix="batch_best_hyperparameters",
            sheet_name="Best Parameters",
        )
    else:
        st.info("No optimized hyperparameters are available for this batch run.")

def display_results(res, key_prefix=None):
    # Unique prefix prevents StreamlitDuplicateElementKey errors when multiple models are rendered.
    if key_prefix is None:
        key_prefix = str(res.get('model_name', 'model')).replace(' ', '_').replace('/', '_')
    st.header("Analysis Results")
    st.markdown(
        f"""
        <div class="app-hero" style="margin-top:0.2rem;">
            <div class="app-hero-title">{res['model_name']} Analysis Overview</div>
            <div class="app-hero-text">Review evaluation metrics, diagnostic plots, explainability outputs, and comparative visualizations for the selected model.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    kpi_cards = [
        ("Test RMSE", f"{res['test_metrics']['rmse']:.4f}", "Lower is better"),
        ("Test R²", f"{res['test_metrics']['r2']:.4f}", "Higher is better"),
        ("Features Used", f"{len(res['features'])}", ", ".join(res['features'])),
        ("Sampling Period", f"{res['interval']}", f"{res['split_method']} split"),
    ]
    for col, (label, value, sub) in zip([c1, c2, c3, c4], kpi_cards):
        with col:
            st.markdown(
                f"""
                <div class="dashboard-card">
                    <div class="dashboard-label">{label}</div>
                    <div class="dashboard-value">{value}</div>
                    <div class="dashboard-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    plot_config_for_display = st.session_state.get('plot_config', DEFAULT_PLOT_STYLES)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([" Performance Metrics", "Analysis Charts", " Model Insights", " Residuals & Location Insights", " Prediction Comparison", " Pair Plots", " Parallel Coordinates", " Model Summary Graphic"]) 
    
    with tab1:
        st.info(f"**Model:** `{res['model_name']}` | **Split:** `{res['split_method']}` ({res['train_perc']}/{res['val_perc']}/{res['test_perc']}) | **Time Interval:** `{res['interval']}`")
        st.caption(f"Features Used: **{', '.join(res['features'])}**")
        
        st.subheader(f"Performance Metrics for {res['model_name']} Model (Training, Validation, Test Sets)")
        metrics_data = {
            "Metric": ["RMSE", "MAE", "MAPE", "R²"],
            "Training Set": [
                res['train_metrics']['rmse'], res['train_metrics']['mae'], 
                res['train_metrics']['mape'], res['train_metrics']['r2']
            ],
            "Validation Set": [
                res['val_metrics']['rmse'], res['val_metrics']['mae'], 
                res['val_metrics']['mape'], res['val_metrics']['r2']
            ],
            "Test Set": [
                res['test_metrics']['rmse'], res['test_metrics']['mae'], 
                res['test_metrics']['mape'], res['test_metrics']['r2']
            ],
        }
        df_metrics = pd.DataFrame(metrics_data).set_index("Metric")
        st.dataframe(df_metrics.style.format("{:.4f}"))
        _download_table_as_csv_excel(
            df_metrics.reset_index(),
            base_filename=f"{key_prefix}_performance_metrics",
            key_prefix=f"{key_prefix}_performance_metrics",
            sheet_name="Performance Metrics",
        )

        st.markdown("---")
        st.subheader("Train/Validation/Test Set Distributions")
        plot_library_dist = st.session_state.get("global_plot_library", "Plotly")
        if all(key in res for key in ['y_train', 'y_val', 'y_test']):
            if plot_library_dist == "Seaborn" and _SEABORN_AVAILABLE:
                plot_dataset_distributions_seaborn(
                    y_train=res['y_train'], 
                    y_val=res['y_val'], 
                    y_test=res['y_test'], 
                    display_unit=res['display_unit'], 
                    plot_config=plot_config_for_display
                )
            else:
                plot_dataset_distributions(
                    y_train=res['y_train'], 
                    y_val=res['y_val'], 
                    y_test=res['y_test'], 
                    display_unit=res['display_unit'], 
                    plot_config=plot_config_for_display
                )
        else:
            st.info("Dataset distribution data is not available.")
        st.markdown("---")

        st.subheader("Mean Test Performance (Summary)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean R² Score", f"{r2_score(res['y_test'], res['y_pred_series']):.4f}")
        col2.metric("Mean RMSE", f"{np.sqrt(mean_squared_error(res['y_test'], res['y_pred_series'])):.2f}")
        col3.metric("Mean MAE", f"{mean_absolute_error(res['y_test'], res['y_pred_series']):.2f}")
        
        st.subheader("Location-Based Test Performance")
        if res.get('detailed_loc_metrics_df') is not None and not res['detailed_loc_metrics_df'].empty:
            # Pivot the detailed table for better display
            display_df = res['detailed_loc_metrics_df'].pivot_table(
                index='Location', 
                columns='Set', 
                values=['r2', 'rmse', 'mae', 'mape']
            ).sort_index(axis=1, level=1) # Sort columns for consistent order
            # Improve column names for readability
            display_df.columns = [f'{val[1].capitalize()} {val[0].upper()}' for val in display_df.columns]
            st.dataframe(display_df.style.format("{:.2f}"))
            _download_table_as_csv_excel(
                display_df.reset_index(),
                base_filename=f"{key_prefix}_location_test_performance",
                key_prefix=f"{key_prefix}_location_test_performance",
                sheet_name="Location Metrics",
            )
        else:
            st.info("No location-based test results available.")

        st.subheader("Optimized Hyperparameters")
        if res.get('optimized', False) and res.get('best_params'):
            # Show optimization mode and parameters
            opt_mode_str = res.get('optimization_mode', 'N/A')
            st.write(f"Optimization Mode: **{opt_mode_str}**")
            
            df_params = pd.DataFrame([res['best_params']]).T.reset_index()
            df_params.columns = ["Parameter", "Value"]
            df_params['Value'] = df_params['Value'].astype(str) # PyArrow Error Fix
            st.dataframe(df_params)
            _download_table_as_csv_excel(
                df_params,
                base_filename=f"{key_prefix}_best_hyperparameters",
                key_prefix=f"{key_prefix}_best_hyperparameters",
                sheet_name="Best Parameters",
            )
        else:
            st.info("Model optimization was not enabled for this run, or no parameters were selected for optimization.")
        
        st.subheader("Default (Unoptimized) Parameters")
        if res.get('model'):
            current_model_params = res['model'].get_params()
            optimized_params_keys = res['best_params'].keys() if res.get('optimized', False) and res.get('best_params') else set()

            default_params_to_display = {}
            for param_name, param_value in current_model_params.items():
                if '__' in param_name or param_name == 'estimator':
                    continue
                
                if param_name not in optimized_params_keys:
                    if not isinstance(param_value, (pd.DataFrame, pd.Series, np.ndarray, list, dict)) or (isinstance(param_value, (list, dict)) and len(str(param_value)) < 50):
                        default_params_to_display[param_name] = param_value
                    else:
                        default_params_to_display[param_name] = f"<{type(param_value).__name__} object>"

            if default_params_to_display:
                df_default_params = pd.DataFrame([default_params_to_display]).T.reset_index()
                df_default_params.columns = ["Parameter", "Value"]
                df_default_params['Value'] = df_default_params['Value'].astype(str) # PyArrow Error Fix
                st.dataframe(df_default_params)
            else:
                if not res.get('optimized', False) or not res.get('best_params'):
                    st.info("All parameters are at their default values (optimization not enabled or no parameters selected for optimization).")
                else:
                    st.info("All relevant parameters were optimized, or default parameters are implicitly used.")
        else:
            st.info("Model object not available to retrieve default parameters.")

        st.subheader("Computational Load Summary")
        num_combinations_tried = "N/A"
        if res.get('optimized', False) and res.get('cv_results_'):
            try:
                num_combinations_tried = len(res['cv_results_']['params'])
            except KeyError:
                num_combinations_tried = "Error counting"
        
        total_analysis_duration_per_sample = "N/A"
        if res.get('analysis_duration') is not None and res.get('df_processed', pd.DataFrame()).shape[0] > 0:
            total_analysis_duration_per_sample = (res['analysis_duration'] / res['df_processed'].shape[0]) * 1000

        avg_fit_time_per_optimized_combination = "N/A"
        if res.get('optimized', False) and res.get('cv_results_') and num_combinations_tried != "N/A" and num_combinations_tried > 0:
            avg_fit_time_per_optimized_combination = np.mean(res['cv_results_']['mean_fit_time'])
            
        comp_load_data = {
            "Metric/Parameter": [
                "Total Analysis Duration",
                "Processed Data Rows",
                "Processed Data Columns",
                "Model Input Features (for prediction)",
                "Selected Side Effects (Features)",
                "Selected Model",
                "Optimization Status",
                "Hyperparameter Combinations Tried",
                "Avg Fit Time per Opt. Combination",
                "Total Analysis Duration per Sample",
                "Sampling Period",
                "Data Split Ratio",
                "Splitting Method"
            ],
            "Value": [
                (f"{st.session_state.get('analysis_duration'):.2f} seconds"
                 if st.session_state.get('analysis_duration') is not None else "N/A"),
                str(res.get('df_processed', pd.DataFrame()).shape[0]),
                str(res.get('df_processed', pd.DataFrame()).shape[1]),
                str(len(res.get('feature_names_for_model', []))),
                (lambda features_list: ", ".join(f for f in features_list if f in ["Temperature", "Humidity"]) 
                 if any(item in ["Temperature", "Humidity"] for item in features_list) else "None")(res.get('features', [])),
                res.get('model_name', 'N/A'),
                "Performed" if res['optimized'] else "Not Performed",
                str(num_combinations_tried),
                f"{avg_fit_time_per_optimized_combination:.4f} seconds" if isinstance(avg_fit_time_per_optimized_combination, float) else avg_fit_time_per_optimized_combination,
                f"{total_analysis_duration_per_sample:.4f} ms/sample" if isinstance(total_analysis_duration_per_sample, float) else total_analysis_duration_per_sample,
                res.get('interval', 'N/A'),
                f"{res.get('train_perc', 'N/A')}/{res.get('val_perc', 'N/A')}/{res.get('test_perc', 'N/A')}%",
                res.get('split_method', 'N/A')
            ]
        }
        df_comp_load = pd.DataFrame(comp_load_data).set_index("Metric/Parameter")
        st.dataframe(df_comp_load)

    with tab2:
        st.header("Graphical Analysis")
        plot_loc_options = ['Mean'] + sorted(list(res['df_processed']['location'].unique()))
        plot_loc = st.selectbox("Sensor Location", options=plot_loc_options, key=f"tab2_plot_loc_select_{res['model_name']}")
        plot_library = st.session_state.get("global_plot_library", "Plotly")
        
        if plot_loc != 'Mean':
            all_indices_for_plot_loc = res['df_processed'][res['df_processed']['location'] == plot_loc].index
            plot_indices = all_indices_for_plot_loc.intersection(res['y_test'].index)
        else:
            plot_indices = res['y_test'].index
        
        if not plot_indices.empty:
            y_test_plot = res['y_test'].loc[plot_indices]
            y_pred_plot = res['y_pred_series'].loc[plot_indices]
            
            st.subheader(plot_config_for_display["scatter_plot"]["title"] or f"{res['model_name']} Model Correlation Plot - {plot_loc.capitalize()} {res['pollutant_unit']}")
            if plot_library == "Seaborn" and _SEABORN_AVAILABLE:
                plot_scatter_seaborn(
                    y_test=y_test_plot, 
                    y_pred=y_pred_plot, 
                    pollutant_unit=res['pollutant_unit'], 
                    location_name=plot_loc, 
                    model_name=res['model_name'], 
                    display_unit=res['display_unit'], 
                    plot_config=plot_config_for_display, 
                    chart_key=f"{res['model_name']}_{plot_loc if 'plot_loc' in locals() else 'all'}_scatter_tab2_seaborn"
                )
            else:
                plot_scatter(
                    y_test=y_test_plot, 
                    y_pred=y_pred_plot, 
                    pollutant_unit=res['pollutant_unit'], 
                    location_name=plot_loc, 
                    model_name=res['model_name'], 
                    display_unit=res['display_unit'], 
                    plot_config=plot_config_for_display, 
                    chart_key=f"{res['model_name']}_{plot_loc if 'plot_loc' in locals() else 'all'}_scatter_tab2_plotly"
                )
            
            st.subheader(plot_config_for_display["residuals_plot"]["title"] or f"{res['model_name']} Model Residuals Plot - {plot_loc.capitalize()} {res['pollutant_unit']}")
            if plot_library == "Seaborn" and _SEABORN_AVAILABLE:
                plot_residuals_seaborn(
                    y_test=y_test_plot, 
                    y_pred=y_pred_plot, 
                    pollutant_unit=res['pollutant_unit'], 
                    location_name=plot_loc, 
                    model_name=res['model_name'], 
                    display_unit=res['display_unit'], 
                    plot_config=plot_config_for_display
                )
            else:
                plot_residuals(
                    y_test=y_test_plot, 
                    y_pred=y_pred_plot, 
                    pollutant_unit=res['pollutant_unit'], 
                    location_name=plot_loc, 
                    model_name=res['model_name'], 
                    display_unit=res['display_unit'], 
                    plot_config=plot_config_for_display
                )
            
            st.subheader(plot_config_for_display["time_series"]["title"] or f"{res['model_name']} Model Time Series - {plot_loc.capitalize()} {res['pollutant_unit']}")

            # Build the time-series dataframe from the test samples.
            # IMPORTANT: do not randomly sample time-series data, because random
            # sampling breaks temporal continuity and may create misleading lines.
            plot_df = res['df_processed'].loc[plot_indices].copy()
            plot_df['calibrated_pollutant'] = y_pred_plot

            required_ts_cols = [
                'timestamp',
                'raw_pollutant',
                'calibrated_pollutant',
                'reference_pollutant',
            ]
            plot_df['timestamp'] = pd.to_datetime(plot_df['timestamp'], errors='coerce')
            plot_df = plot_df.dropna(subset=required_ts_cols)

            # For the "Mean" view, df_processed contains multiple sensor/location
            # rows at the same timestamp. If these rows are plotted directly as a
            # line, Plotly connects different sensors at the same timestamp and
            # produces vertical spikes. Therefore, aggregate by timestamp first.
            plot_df = (
                plot_df
                .groupby('timestamp', as_index=False)
                .agg({
                    'raw_pollutant': 'mean',
                    'calibrated_pollutant': 'mean',
                    'reference_pollutant': 'mean',
                })
                .sort_values('timestamp')
                .reset_index(drop=True)
            )

            if plot_df.empty:
                st.warning("No valid time-series data is available for the selected location.")
            else:
                min_ts = plot_df['timestamp'].min()
                max_ts = plot_df['timestamp'].max()

                st.caption(f"Available time-series range: {min_ts} → {max_ts}")

                col_start_date, col_end_date = st.columns(2)
                with col_start_date:
                    ts_start_date = st.date_input(
                        "Start date",
                        value=min_ts.date(),
                        min_value=min_ts.date(),
                        max_value=max_ts.date(),
                        key=f"{key_prefix}_{plot_loc}_ts_start_date",
                    )
                with col_end_date:
                    ts_end_date = st.date_input(
                        "End date",
                        value=max_ts.date(),
                        min_value=min_ts.date(),
                        max_value=max_ts.date(),
                        key=f"{key_prefix}_{plot_loc}_ts_end_date",
                    )

                col_start_time, col_end_time = st.columns(2)
                with col_start_time:
                    ts_start_time = st.time_input(
                        "Start time",
                        value=min_ts.time(),
                        key=f"{key_prefix}_{plot_loc}_ts_start_time",
                    )
                with col_end_time:
                    ts_end_time = st.time_input(
                        "End time",
                        value=max_ts.time(),
                        key=f"{key_prefix}_{plot_loc}_ts_end_time",
                    )

                start_dt = pd.to_datetime(f"{ts_start_date} {ts_start_time}")
                end_dt = pd.to_datetime(f"{ts_end_date} {ts_end_time}")

                if start_dt > end_dt:
                    st.error("Start datetime cannot be later than end datetime.")
                else:
                    plot_df = plot_df[
                        (plot_df['timestamp'] >= start_dt)
                        & (plot_df['timestamp'] <= end_dt)
                    ].copy()

                    if plot_df.empty:
                        st.warning("No time-series data exists in the selected range.")
                    else:
                        # Keep a continuous segment for performance; never use random sampling.
                        max_points = st.number_input(
                            "Maximum time-series points to display",
                            min_value=100,
                            max_value=20000,
                            value=2000,
                            step=100,
                            key=f"{key_prefix}_{plot_loc}_ts_max_points",
                        )
                        if len(plot_df) > max_points:
                            plot_df = plot_df.iloc[:max_points].copy()
                            st.info(
                                f"Selected range contains more than {max_points} points. "
                                f"Only the first {max_points} continuous points are displayed."
                            )

                        st.write(
                            "Displayed time-series range:",
                            plot_df['timestamp'].min(),
                            "→",
                            plot_df['timestamp'].max(),
                            f"({len(plot_df)} points)",
                        )

                        plot_time_series(
                            df_plot=plot_df,
                            pollutant_unit=res['pollutant_unit'],
                            location_name=plot_loc,
                            model_name=res['model_name'],
                            display_unit=res['display_unit'],
                            plot_config=plot_config_for_display
                        )
        else:
            st.info(f"No test data available for location '{plot_loc}' to display charts.")

    
    with tab3:
        st.header("Model Insights")

        native_df = get_native_feature_importance_df(res.get('model'), res.get('feature_names_for_model'))
        permutation_df = get_permutation_importance_df(
            res.get('model'),
            res.get('X_test_processed_df'),
            res.get('y_test')
        )

        st.subheader("Model Explainability")
        if native_df is not None:
            st.markdown("**Native Feature Importance**")
            fig_native = plot_importance_bar(native_df, f"Native Feature Importance for {res['model_name']}")
            if fig_native is not None:
                st.plotly_chart(fig_native, use_container_width=True, config=PLOTLY_CONFIG, key=f'{key_prefix}_native_feature_importance_chart')
            st.dataframe(native_df, use_container_width=True)
        else:
            st.info("Native feature importance is not available for this model type.")

        st.markdown("---")
        st.subheader("Permutation Importance")
        if permutation_df is not None:
            st.caption("Permutation importance can be used across a wider range of models and is often more suitable for model-to-model comparison in a paper.")
            fig_perm = plot_importance_bar(permutation_df, f"Permutation Importance for {res['model_name']}")
            if fig_perm is not None:
                st.plotly_chart(fig_perm, use_container_width=True, config=PLOTLY_CONFIG, key=f'{key_prefix}_permutation_importance_chart')
            st.dataframe(permutation_df, use_container_width=True)
        else:
            st.warning("Permutation importance could not be computed for this run.")

        st.markdown("---")

        st.subheader("Hyperparameter Optimization Heatmap (if applicable)")
        if res.get('optimized', False) and res.get('cv_results_') and res.get('optimization_mode') not in ["Bayesian Optimization (skopt)", "Bayesian Optimization", "RandomizedSearchCV"]:
            cv_results_df = pd.DataFrame(res['cv_results_'])
            optimized_param_keys = [p.replace('param_', '') for p in cv_results_df.columns if p.startswith('param_')]

            if len(optimized_param_keys) == 2:
                param1 = optimized_param_keys[0]
                param2 = optimized_param_keys[1]
                heatmap_data = cv_results_df.pivot_table(
                    values='mean_test_score',
                    index=f'param_{param1}',
                    columns=f'param_{param2}'
                )
                fig_heatmap = go.Figure(data=go.Heatmap(
                       z=heatmap_data.values,
                       x=heatmap_data.columns.astype(str),
                       y=heatmap_data.index.astype(str),
                       colorscale='Viridis',
                       colorbar=dict(title='Mean R² Score')
                   ))
                fig_heatmap.update_layout(
                    title=f'Hyperparameter Optimization Heatmap (R² Score) for {res["model_name"]}',
                    xaxis_title=param2,
                    yaxis_title=param1,
                    xaxis_nticks=len(heatmap_data.columns),
                    yaxis_nticks=len(heatmap_data.index)
                )
                st.plotly_chart(fig_heatmap, use_container_width=True, config=PLOTLY_CONFIG, key=f'{key_prefix}_hyperparameter_heatmap_chart')
            elif len(optimized_param_keys) > 2:
                st.warning(f"Heatmap visualization is best for 2 optimized hyperparameters. {len(optimized_param_keys)} parameters were optimized. Please refer to performance metrics for details.")
            elif len(optimized_param_keys) < 2:
                 st.info("No heatmap generated: Less than 2 hyperparameters were selected for optimization.")
        elif res.get('optimization_mode') in ["Bayesian Optimization (skopt)", "Bayesian Optimization"]:
            st.info("Heatmap visualization is not applicable for Bayesian Optimization, as it does not search on a regular grid.")
        elif res.get('optimization_mode') == "RandomizedSearchCV":
             st.info("Heatmap visualization is not applicable for RandomizedSearchCV, as it does not search on a regular grid.")
        else:
            st.info("Hyperparameter optimization heatmap is available when GridSearch is enabled and at least 2 parameters are optimized.")

    with tab8:
        st.header("Model Summary Graphic")
        st.caption("Publication-style summary figure combining validation curves, hyperparameter interaction, optimization progress, iteration distribution, and sensitivity analysis.")

        if res.get('optimized', False) and res.get('cv_results_'):
            cv_results_df = pd.DataFrame(res['cv_results_'])
            plot_model_summary_2x3(
                cv_results_df,
                res['model_name'],
                format_optimization_mode_label(res.get('optimization_mode', 'Unknown')),
                AUTO_OPTIMIZE_PARAMS
            )
        else:
            st.info("This figure is available for optimized runs with cv_results data.")

    with tab4:
        st.header("Residuals and Location-Based Insights")
        plot_loc_options_res = ['Mean'] + sorted(list(res['df_processed']['location'].unique()))
        plot_loc_res = st.selectbox("Sensor Location for Residuals Plots", options=plot_loc_options_res, key=f"tab4_plot_loc_select_{res['model_name']}")
        plot_library_res = st.session_state.get("global_plot_library", "Plotly")

        if plot_loc_res != 'Mean':
            all_indices_for_plot_loc_res = res['df_processed'][res['df_processed']['location'] == plot_loc_res].index
            plot_indices_res = all_indices_for_plot_loc_res.intersection(res['y_test'].index)
        else:
            plot_indices_res = res['y_test'].index
        
        if not plot_indices_res.empty:
            y_test_plot_res = res['y_test'].loc[plot_indices_res]
            y_pred_plot_res = res['y_pred_series'].loc[plot_indices_res]

            st.subheader("Residuals Distribution (Histogram)")
            if plot_library_res == "Seaborn" and _SEABORN_AVAILABLE:
                plot_residuals_histogram_seaborn(
                    y_test=y_test_plot_res, 
                    y_pred=y_pred_plot_res, 
                    pollutant_unit=res['pollutant_unit'], 
                    location_name=plot_loc_res, 
                    model_name=res['model_name'], 
                    display_unit=res['display_unit'], 
                    plot_config=plot_config_for_display
                )
            else:
                plot_residuals_histogram(
                    y_test=y_test_plot_res, 
                    y_pred=y_pred_plot_res, 
                    pollutant_unit=res['pollutant_unit'], 
                    location_name=plot_loc_res, 
                    model_name=res['model_name'], 
                    display_unit=res['display_unit'], 
                    plot_config=plot_config_for_display
                )

            st.subheader("Residuals Density (KDE Plot)")
            if plot_library_res == "Seaborn" and _SEABORN_AVAILABLE:
                plot_residuals_kde_seaborn(
                    y_test=y_test_plot_res, 
                    y_pred=y_pred_plot_res, 
                    pollutant_unit=res['pollutant_unit'], 
                    location_name=plot_loc_res, 
                    model_name=res['model_name'], 
                    display_unit=res['display_unit'], 
                    plot_config=plot_config_for_display
                )
            else:
                plot_residuals_kde(
                    y_test=y_test_plot_res, 
                    y_pred=y_pred_plot_res, 
                    pollutant_unit=res['pollutant_unit'], 
                    location_name=plot_loc_res, 
                    model_name=res['model_name'], 
                    display_unit=res['display_unit'], 
                    plot_config=plot_config_for_display
                )
        else:
            st.info(f"No test data available for location '{plot_loc_res}' to display residuals plots.")
        
        st.markdown("---")
        st.subheader("Location-Based Metric Consistency Across Datasets")
        if res.get('detailed_loc_metrics_df') is not None and not res['detailed_loc_metrics_df'].empty:
            detailed_df = res['detailed_loc_metrics_df']
            
            metric_options = {'RMSE': 'rmse', 'MAE': 'mae', 'R²': 'r2', 'MAPE': 'mape'}
            selected_metric_label = st.selectbox(
                "Select Metric to Compare",
                options=list(metric_options.keys()),
                key=f"location_consistency_metric_select_{res['model_name']}"
            )
            selected_metric_col = metric_options[selected_metric_label]

            fig = px.box(
                detailed_df,
                x='Location',
                y=selected_metric_col,
                color='Location',
                title=f'Consistency of {selected_metric_label} Across Datasets for Each Location',
                labels={'Location': 'Location', selected_metric_col: selected_metric_label},
                points='all' # Show all 3 points
            )
            fig.update_layout(
                title=dict( # Yeni eklenen başlık ayarı
                    text=f'Consistency of {selected_metric_label} Across Datasets for Each Location',
                    font=dict(
                        size=plot_config_for_display["general"]["plot_title_font_size"],
                        family=plot_config_for_display["general"]["font_family"],
                        color=plot_config_for_display["general"]["font_color"]
                    )
                ),
                template=plot_config_for_display["general"]["template"],
                font=dict(family=plot_config_for_display["general"]["font_family"], color=plot_config_for_display["general"]["font_color"]),
                showlegend=False,
                xaxis=dict( # Eksen ayarları
                    title=dict( # Eksen başlığı font ayarı
                        text='Location',
                        font=dict(
                            size=plot_config_for_display["general"]["axis_title_font_size"],
                            color=plot_config_for_display["general"]["axis_title_font_color"],
                            family=plot_config_for_display["general"]["axis_title_font_family"]
                        )
                    ),
                    tickfont=dict( # Eksen tik etiketleri font ayarı
                        size=plot_config_for_display["general"]["axis_tick_font_size"],
                        color=plot_config_for_display["general"]["axis_tick_font_color"],
                        family=plot_config_for_display["general"]["axis_tick_font_family"]
                    )
                ),
                yaxis=dict( # Eksen ayarları
                    title=dict( # Eksen başlığı font ayarı
                        text=selected_metric_label,
                        font=dict(
                            size=plot_config_for_display["general"]["axis_title_font_size"],
                            color=plot_config_for_display["general"]["axis_title_font_color"],
                            family=plot_config_for_display["general"]["axis_title_font_family"]
                        )
                    ),
                    tickfont=dict( # Eksen tik etiketleri font ayarı
                        size=plot_config_for_display["general"]["axis_tick_font_size"],
                        color=plot_config_for_display["general"]["axis_tick_font_color"],
                        family=plot_config_for_display["general"]["axis_tick_font_family"]
                    )
                )
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=f'{key_prefix}_loc_consistency_chart_{selected_metric_label}')
        else:
            st.info("No detailed location-based metrics available to plot.")

    with tab5:
        st.header("Model Predictions vs. Reference Values Across Data Splits")
        if all(key in res for key in ['y_train', 'y_train_pred_series', 'y_val', 'y_val_pred_series', 
                                      'y_test', 'y_pred_series', 'X_train', 'X_val', 'X_test', 'df_processed']):

            plot_loc_options_tab5 = ['Mean'] + sorted(list(res['df_processed']['location'].unique()))
            plot_loc_tab5 = st.selectbox(
                "Sensor Location", 
                options=plot_loc_options_tab5, 
                key=f"tab5_plot_loc_select_{res['model_name']}"
            )

            if plot_loc_tab5 == 'Mean':
                y_train_plot = res['y_train']
                y_train_pred_plot = res['y_train_pred_series']
                y_val_plot = res['y_val']
                y_val_pred_plot = res['y_val_pred_series']
                y_test_plot = res['y_test']
                y_test_pred_plot = res['y_pred_series']
            else:
                train_indices = res['X_train'][res['X_train']['location'] == plot_loc_tab5].index
                y_train_plot = res['y_train'].loc[train_indices]
                y_train_pred_plot = res['y_train_pred_series'].loc[train_indices]

                val_indices = res['X_val'][res['X_val']['location'] == plot_loc_tab5].index
                y_val_plot = res['y_val'].loc[val_indices]
                y_val_pred_plot = res['y_val_pred_series'].loc[val_indices]

                test_indices = res['X_test'][res['X_test']['location'] == plot_loc_tab5].index
                y_test_plot = res['y_test'].loc[test_indices]
                y_test_pred_plot = res['y_pred_series'].loc[test_indices]
            
            plot_library_pred_comp = st.session_state.get("global_plot_library", "Plotly")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Training Set")
                if not y_train_plot.empty:
                    if plot_library_pred_comp == "Seaborn" and _SEABORN_AVAILABLE:
                         plot_scatter_seaborn(y_test=y_train_plot, y_pred=y_train_pred_plot, pollutant_unit=res['pollutant_unit'], location_name=plot_loc_tab5, model_name=res['model_name'], display_unit=res['display_unit'], plot_config=plot_config_for_display)
                    else:
                        plot_scatter(
                            y_test=y_train_plot, 
                            y_pred=y_train_pred_plot,
                            pollutant_unit=res['pollutant_unit'], 
                            location_name=plot_loc_tab5,
                            model_name=res['model_name'], 
                            display_unit=res['display_unit'],
                            plot_config=plot_config_for_display,
                            chart_key="scatter_tab5_train"
                        )
                else:
                    st.info(f"No training data available for location: {plot_loc_tab5}")

            with col2:
                st.subheader("Validation Set")
                if not y_val_plot.empty:
                    if plot_library_pred_comp == "Seaborn" and _SEABORN_AVAILABLE:
                         plot_scatter_seaborn(y_test=y_val_plot, y_pred=y_val_pred_plot, pollutant_unit=res['pollutant_unit'], location_name=plot_loc_tab5, model_name=res['model_name'], display_unit=res['display_unit'], plot_config=plot_config_for_display)
                    else:
                        plot_scatter(
                            y_test=y_val_plot, 
                            y_pred=y_val_pred_plot, 
                            pollutant_unit=res['pollutant_unit'], 
                            location_name=plot_loc_tab5,
                            model_name=res['model_name'], 
                            display_unit=res['display_unit'],
                            plot_config=plot_config_for_display,
                            chart_key="scatter_tab5_val"
                        )
                else:
                    st.info(f"No validation data available for location: {plot_loc_tab5}")

            with col3:
                st.subheader("Test Set")
                if not y_test_plot.empty:
                    if plot_library_pred_comp == "Seaborn" and _SEABORN_AVAILABLE:
                         plot_scatter_seaborn(y_test=y_test_plot, y_pred=y_test_pred_plot, pollutant_unit=res['pollutant_unit'], location_name=plot_loc_tab5, model_name=res['model_name'], display_unit=res['display_unit'], plot_config=plot_config_for_display)
                    else:
                        plot_scatter(
                            y_test=y_test_plot, 
                            y_pred=y_test_pred_plot,
                            pollutant_unit=res['pollutant_unit'], 
                            location_name=plot_loc_tab5,
                            model_name=res['model_name'], 
                            display_unit=res['display_unit'],
                            plot_config=plot_config_for_display,
                            chart_key="scatter_tab5_test"
                        )
                else:
                    st.info(f"No test data available for location: {plot_loc_tab5}")
        else:
            st.info("Training, Validation, or Test prediction data is not available to display comparison charts. Please run an analysis first.")

    with tab6:
        st.header("Hyperparameter Pair Plots")
        if res.get('optimized', False) and res.get('cv_results_'):
            cv_results_df = pd.DataFrame(res['cv_results_'])
            cv_results_df['r2_score'] = cv_results_df['mean_test_score']
            param_cols = [col for col in cv_results_df.columns if col.startswith('param_')]
            
            if len(param_cols) > 1:
                plot_df = cv_results_df[param_cols + ['r2_score']].rename(columns=lambda x: x.replace('param_', ''))
                
                if res.get('param_scope') == 'Top 3 Parameters' and res.get('optimization_mode') != "Manual Optimization":
                    top_3_keys = AUTO_OPTIMIZE_PARAMS.get(res['model_name'], [])
                    valid_top_3_params = [p for p in top_3_keys if p in plot_df.columns]
                    plot_df = plot_df[valid_top_3_params + ['r2_score']]
                
                if len(plot_df.columns) <= 6: # Heuristic limit for readability
                    if _SEABORN_AVAILABLE:
                        st.info("Using Seaborn for Pair Plot visualization for a different style.")
                        plot_pair_plots_seaborn(
                            plot_df,
                            title=f"Hyperparameter Pair Plots for {res['model_name']} (R² Score)",
                            plot_config=plot_config_for_display
                        )
                    else:
                        st.warning("Seaborn library not found. Falling back to default Plotly Pair Plots.")
                        plot_pair_plots(
                            plot_df,
                            title=f"Hyperparameter Pair Plots for {res['model_name']} (R² Score)",
                            plot_config=plot_config_for_display
                        )
                else:
                    st.warning(f"Pair plots can become difficult to read with more than 5 parameters. Your analysis has {len(plot_df.columns) - 1} parameters.")
                    
                    st.info("To improve readability, consider filtering parameters for the Pair Plot.")
                    all_params = [p for p in plot_df.columns if p != 'r2_score']
                    selected_params_pair = st.multiselect(
                        "Select parameters to display in Pair Plot:",
                        options=all_params,
                        default=all_params[:5] if len(all_params) > 5 else all_params,
                        key="pair_plot_params_multiselect"
                    )
                    
                    if selected_params_pair:
                        filtered_plot_df = plot_df[selected_params_pair + ['r2_score']]
                        if _SEABORN_AVAILABLE:
                            plot_pair_plots_seaborn(
                                filtered_plot_df,
                                title=f"Hyperparameter Pair Plots for {res['model_name']} (R² Score)",
                                plot_config=plot_config_for_display
                            )
                        else:
                            plot_pair_plots(
                                filtered_plot_df,
                                title=f"Hyperparameter Pair Plots for {res['model_name']} (R² Score)",
                                plot_config=plot_config_for_display
                            )

            else:
                st.info("Pair plots require at least 2 optimized hyperparameters.")
        else:
            st.info("No optimization results available to plot. Please run an analysis with optimization enabled.")

    with tab7:
        st.header("Hyperparameter Parallel Coordinates")
        if res.get('optimized', False) and res.get('cv_results_'):
            cv_results_df = pd.DataFrame(res['cv_results_'])
            cv_results_df['r2_score'] = cv_results_df['mean_test_score']
            param_cols = [col for col in cv_results_df.columns if col.startswith('param_')]
            
            if len(param_cols) > 0:
                plot_df = cv_results_df.rename(columns=lambda x: x.replace('param_', ''))
                plot_df['r2_score'] = plot_df['mean_test_score']

                valid_params = [col for col in plot_df.columns if plot_df[col].dtype in ['int64', 'float64', 'object'] and col != 'r2_score']
                
                st.info("Please select the parameters you want to view in the Parallel Coordinates Plot for better readability.")
                selected_params_parallel = st.multiselect(
                    "Select Parameters to Display:",
                    options=valid_params,
                    default=valid_params,
                    key="parallel_plot_params_multiselect"
                )
                
                if selected_params_parallel:
                    selected_plot_df = plot_df[selected_params_parallel + ['r2_score']]
                    plot_parallel_coordinates(
                        selected_plot_df,
                        title=f"Hyperparameter Parallel Coordinates for {res['model_name']} (R² Score)",
                        plot_config=plot_config_for_display
                    )
                else:
                    st.warning("Please select at least one parameter to display the plot.")
            else:
                st.info("Parallel Coordinates plot requires at least 1 optimized hyperparameter.")
        else:
            st.info("No optimization results available to plot. Please run an anaylsis with optimization enabled.")


def display_selected_model_results():
    selected_models = list(st.session_state.get("selected_models_to_run", []))
    results_by_model = st.session_state.get("results_by_model", {})

    valid_models = [m for m in selected_models if m in results_by_model]

    # Batch/all-model mode: only tables, no detailed plots or graphic downloads.
    if len(valid_models) > 1:
        display_batch_summary_results(valid_models, results_by_model)
        return

    # Individual mode keeps the full detailed graphics.
    if len(valid_models) == 1:
        display_results(results_by_model[valid_models[0]], key_prefix=f"single_{valid_models[0]}")
        return

    if st.session_state.get("analysis_results"):
        display_results(
            st.session_state.analysis_results,
            key_prefix=f"single_{st.session_state.analysis_results.get('model_name', 'model')}"
        )


# --- Main Model Run Function ---

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'analysis_duration' not in st.session_state:
    st.session_state.analysis_duration = None
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'plot_config' not in st.session_state:
    st.session_state.plot_config = deepcopy(DEFAULT_PLOT_STYLES)
if 'history' not in st.session_state:
    st.session_state.history = []
if 'results_by_model' not in st.session_state:
    st.session_state.results_by_model = {}
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0
if 'run_analysis_triggered' not in st.session_state:
    st.session_state.run_analysis_triggered = False
if 'run_all_models_triggered' not in st.session_state:
    st.session_state.run_all_models_triggered = False
if 'run_selected_models_triggered' not in st.session_state:
    st.session_state.run_selected_models_triggered = False
if 'selected_models_to_run' not in st.session_state:
    st.session_state.selected_models_to_run = []
if 'selected_models_multiselect_sidebar' not in st.session_state:
    st.session_state.selected_models_multiselect_sidebar = []
if 'optimization_target_model_sidebar' not in st.session_state:
    st.session_state.optimization_target_model_sidebar = None
if 'run_config' not in st.session_state:
    st.session_state.run_config = {}
if 'auto_optimize_param_keys_cache' not in st.session_state:
    st.session_state.auto_optimize_param_keys_cache = {}
if 'stop_auto_run' not in st.session_state:
    st.session_state.stop_auto_run = False
if 'run_optimization_comparison_triggered' not in st.session_state:
    st.session_state.run_optimization_comparison_triggered = False
if 'comparative_results' not in st.session_state:
    st.session_state['comparative_results'] = None
if 'comparative_model_name' not in st.session_state:
    st.session_state['comparative_model_name'] = None
if 'comparative_cv_results' not in st.session_state:
    st.session_state['comparative_cv_results'] = {}

MODELS_TO_RUN_FIRST = [
    "Linear Regression",
    "Ridge Regression",
    "Lasso Regression",
    "ElasticNet Regression",
    "Decision Tree",
    "k-Nearest Neighbors (kNN)",
    "LightGBM",
    "XGBoost",
    "CatBoost",
    "AdaBoost",
    "Gradient Boosting",
    "Random Forest"
]

MODELS_TO_RUN_LAST = [
]



def _safe_plot_config():
    base = deepcopy(DEFAULT_PLOT_STYLES)
    current = st.session_state.get("plot_config", {})
    if isinstance(current, dict):
        for section, values in current.items():
            if section in base and isinstance(values, dict):
                base[section].update(values)
    st.session_state.plot_config = base
    return st.session_state.plot_config

def render_plot_customization_sidebar():
    plot_config = _safe_plot_config()

    with st.expander("Plot Customization", expanded=False):
        st.caption("Configure visual styling for time-series, scatter, and residual diagnostics.")

        with st.container():
            st.markdown("**General Style**")
            plot_config["general"]["template"] = st.selectbox(
                "Plot Template",
                options=["simple_white", "plotly_white", "plotly", "ggplot2", "seaborn"],
                index=["simple_white", "plotly_white", "plotly", "ggplot2", "seaborn"].index(
                    plot_config["general"].get("template", "simple_white")
                ),
                key="plot_template_sidebar",
            )
            plot_config["general"]["font_family"] = st.selectbox(
                "Font Family",
                options=["Arial", "Times New Roman", "Calibri", "Courier New"],
                index=["Arial", "Times New Roman", "Calibri", "Courier New"].index(
                    plot_config["general"].get("font_family", "Arial")
                )
                if plot_config["general"].get("font_family", "Arial") in ["Arial", "Times New Roman", "Calibri", "Courier New"]
                else 0,
                key="plot_font_family_sidebar",
            )
            plot_config["general"]["plot_title_font_size"] = st.slider(
                "Title Font Size", 12, 32, int(plot_config["general"].get("plot_title_font_size", 20)), key="plot_title_font_size_sidebar"
            )
            plot_config["general"]["axis_title_font_size"] = st.slider(
                "Axis Title Font Size", 10, 24, int(plot_config["general"].get("axis_title_font_size", 14)), key="axis_title_font_size_sidebar"
            )
            plot_config["general"]["axis_tick_font_size"] = st.slider(
                "Axis Tick Font Size", 8, 20, int(plot_config["general"].get("axis_tick_font_size", 12)), key="axis_tick_font_size_sidebar"
            )

        st.markdown("---")
        st.markdown("**Plot Rendering**")
        st.radio(
            "Select Plotting Library",
            ["Plotly", "Seaborn"],
            index=0 if st.session_state.get("global_plot_library", "Plotly") == "Plotly" else 1,
            key="global_plot_library",
            help="Applies to distributions, analysis charts, residual diagnostics, prediction comparison, and other supported plots."
        )

        st.markdown("---")
        st.markdown("**Time Series**")
        c1, c2 = st.columns(2)
        with c1:
            plot_config["time_series"]["raw_color"] = st.color_picker(
                "Raw Sensor Color", value=plot_config["time_series"].get("raw_color", "#228B22"), key="raw_color_sidebar"
            )
            plot_config["time_series"]["calibrated_color"] = st.color_picker(
                "Calibrated Color", value=plot_config["time_series"].get("calibrated_color", "#FF7F0E"), key="calibrated_color_sidebar"
            )
        with c2:
            plot_config["time_series"]["reference_color"] = st.color_picker(
                "Reference Color", value=plot_config["time_series"].get("reference_color", "#000000"), key="reference_color_sidebar"
            )
            plot_config["time_series"]["legend_bgcolor"] = st.color_picker(
                "Legend Background", value=plot_config["time_series"].get("legend_bgcolor", "#FFFFFF"), key="legend_bgcolor_sidebar"
            )

        plot_config["time_series"]["raw_width"] = st.slider(
            "Raw Line Width", 1.0, 5.0, float(plot_config["time_series"].get("raw_width", 1.0)), 0.5, key="raw_width_sidebar"
        )
        plot_config["time_series"]["calibrated_width"] = st.slider(
            "Calibrated Line Width", 1.0, 6.0, float(plot_config["time_series"].get("calibrated_width", 2.0)), 0.5, key="calibrated_width_sidebar"
        )
        plot_config["time_series"]["reference_width"] = st.slider(
            "Reference Line Width", 1.0, 6.0, float(plot_config["time_series"].get("reference_width", 3.0)), 0.5, key="reference_width_sidebar"
        )
        plot_config["time_series"]["raw_opacity"] = st.slider(
            "Raw Opacity", 0.1, 1.0, float(plot_config["time_series"].get("raw_opacity", 0.5)), 0.1, key="raw_opacity_sidebar"
        )
        plot_config["time_series"]["calibrated_opacity"] = st.slider(
            "Calibrated Opacity", 0.1, 1.0, float(plot_config["time_series"].get("calibrated_opacity", 1.0)), 0.1, key="calibrated_opacity_sidebar"
        )
        line_style_options = ["solid", "dash", "dot", "dashdot"]
        plot_config["time_series"]["raw_style"] = st.selectbox(
            "Raw Line Style",
            options=line_style_options,
            index=line_style_options.index(plot_config["time_series"].get("raw_style", "solid")) if plot_config["time_series"].get("raw_style", "solid") in line_style_options else 0,
            key="raw_style_sidebar"
        )
        plot_config["time_series"]["calibrated_style"] = st.selectbox(
            "Calibrated Line Style",
            options=line_style_options,
            index=line_style_options.index(plot_config["time_series"].get("calibrated_style", "solid")) if plot_config["time_series"].get("calibrated_style", "solid") in line_style_options else 0,
            key="calibrated_style_sidebar"
        )
        plot_config["time_series"]["reference_style"] = st.selectbox(
            "Reference Line Style",
            options=line_style_options,
            index=line_style_options.index(plot_config["time_series"].get("reference_style", "dash")) if plot_config["time_series"].get("reference_style", "dash") in line_style_options else 1,
            key="reference_style_sidebar"
        )

        st.markdown("---")
        st.markdown("**Scatter / Residuals**")
        c3, c4 = st.columns(2)
        with c3:
            plot_config["scatter_plot"]["marker_color"] = st.color_picker(
                "Scatter Marker Color", value=plot_config["scatter_plot"].get("marker_color", "#228B22"), key="scatter_marker_color_sidebar"
            )
            plot_config["scatter_plot"]["trendline_color"] = st.color_picker(
                "Trendline Color", value=plot_config["scatter_plot"].get("trendline_color", "#DC143C"), key="scatter_trendline_color_sidebar"
            )
            plot_config["residuals_plot"]["marker_color"] = st.color_picker(
                "Residual Marker Color", value=plot_config["residuals_plot"].get("marker_color", "#228B22"), key="residual_marker_color_sidebar"
            )
        with c4:
            plot_config["residuals_plot"]["zeroline_color"] = st.color_picker(
                "Residual Zero Line Color", value=plot_config["residuals_plot"].get("zeroline_color", "#646464"), key="residual_zero_color_sidebar"
            )
            plot_config["residuals_hist"]["bar_color"] = st.color_picker(
                "Histogram Bar Color", value=plot_config["residuals_hist"].get("bar_color", "#1F77B4"), key="residual_hist_color_sidebar"
            )
            plot_config["residuals_kde"]["line_color"] = st.color_picker(
                "KDE Line Color", value=plot_config["residuals_kde"].get("line_color", "#9467BD"), key="residual_kde_color_sidebar"
            )

        plot_config["scatter_plot"]["marker_size"] = st.slider(
            "Scatter Marker Size", 3, 12, int(plot_config["scatter_plot"].get("marker_size", 5)), key="scatter_marker_size_sidebar"
        )
        plot_config["scatter_plot"]["marker_opacity"] = st.slider(
            "Scatter Marker Opacity", 0.1, 1.0, float(plot_config["scatter_plot"].get("marker_opacity", 0.5)), 0.1, key="scatter_marker_opacity_sidebar"
        )
        plot_config["residuals_plot"]["marker_size"] = st.slider(
            "Residual Marker Size", 3, 12, int(plot_config["residuals_plot"].get("marker_size", 5)), key="residual_marker_size_sidebar"
        )
        plot_config["residuals_plot"]["marker_opacity"] = st.slider(
            "Residual Marker Opacity", 0.1, 1.0, float(plot_config["residuals_plot"].get("marker_opacity", 0.5)), 0.1, key="residual_marker_opacity_sidebar"
        )

        if st.button("Reset Plot Styles", key="reset_plot_styles_sidebar", type="secondary"):
            st.session_state.plot_config = deepcopy(DEFAULT_PLOT_STYLES)
            st.rerun()

        st.session_state.plot_config = plot_config


config = {}

with st.sidebar:
    st.title("AQ-MultiCal: Air Quality Multi-Model Calibration")
    
    with st.expander("Data Management", expanded=True):
        uploaded_file_pollutant = st.file_uploader("1. Upload Main Pollutant Dataset (CO2, PM2.5 etc.)", type=['csv'], key=f"pollutant_{st.session_state.uploader_key}")
        uploaded_file_temp = st.file_uploader("2. Upload Temperature Dataset", type=['csv'], key=f"temp_{st.session_state.uploader_key}")
        uploaded_file_hum = st.file_uploader("3. Upload Humidity Dataset", type=['csv'], key=f"hum_{st.session_state.uploader_key}")

    with st.expander("Model Configuration", expanded=True):
        config['selected_model_names'] = st.multiselect(
            "Models to Run",
            options=list(MODELS.keys()),
            key="selected_models_multiselect_sidebar",
            help="Select one or more models to include in the analysis run."
        )
        selected_models_sidebar = list(st.session_state.get("selected_models_multiselect_sidebar", []))

        if selected_models_sidebar:
            current_target = st.session_state.get("optimization_target_model_sidebar")
            if current_target not in selected_models_sidebar:
                st.session_state.optimization_target_model_sidebar = selected_models_sidebar[0]

            config['model_name'] = st.selectbox(
                "Parameter Preview / Comparison Target Model",
                options=selected_models_sidebar,
                key="optimization_target_model_sidebar",
                help="Used only to preview parameter options and to run the separate optimization-method comparison. Batch optimization is applied to every selected model."
            )
            if len(selected_models_sidebar) > 1:
                st.caption("Batch optimization will be applied to every selected model. Each model uses its own parameter space.")
            else:
                st.caption("Optimization will be applied to the selected model.")
        else:
            config['model_name'] = None
        config['interval_label'] = st.selectbox("Sampling Period", options={'1 Minute (Original)': None, '2 Minutes': '2min','3 Minutes': '3min','5 Minutes': '5min', '10 Minutes': '10min', '15 Minutes': '15min', '30 Minutes': '30min','60 Minutes': '60min'}.keys(), key="interval_select_sidebar")
        st.subheader("Environmental Parameters")
        config['use_temp'] = st.checkbox("Temperature", value=True, key="use_temp_sidebar")
        config['use_hum'] = st.checkbox("Humidity", value=True, key="use_hum_sidebar")
        st.subheader("Data Splitting Strategy")
        config['split_method'] = st.radio("Splitting Method", ["Time-Based", "Random"], index=0, key="split_method_sidebar")
        config['train_perc'] = st.slider("Training Set Ratio (%)", 10, 90, 70, 5, key="train_perc_sidebar")
        config['val_perc'] = st.slider("Validation Set Ratio (%)", 5, (100 - config['train_perc']), 15, 5, key="val_perc_sidebar")

        test_perc_sidebar = 100 - config['train_perc'] - config['val_perc']
        st.metric(label="Test Set Ratio (%)", value=f"{test_perc_sidebar}%", delta_color="off")

        config['optimize'] = st.checkbox(
            "Enable Hyperparameter Optimization",
            help="Run a parameter search before final model fitting.",
            key="optimize_sidebar",
            disabled=(config.get('model_name') is None)
        )
        if config.get('model_name') is None:
            st.caption('Select at least one model in "Models to Run" to configure optimization settings.')

        config['manual_params'] = []
        config['n_iter_random'] = 18
        config['n_iter_bayes'] = 18

        optimization_label_map = {
            "Manual Optimization": "Grid Search",
            "RandomizedSearchCV": "Randomized Search",
            "Bayesian Optimization (skopt)": "Bayesian Optimization",
            "Bayesian Optimization": "Bayesian Optimization",
        }

        if config.get('model_name') is not None and config['optimize']:
            optimization_options = ["Manual Optimization"]
            if _SCIPY_AVAILABLE:
                optimization_options.append("RandomizedSearchCV")
            if _SKOPT_AVAILABLE:
                optimization_options.append("Bayesian Optimization (skopt)")

            config['optimization_mode'] = st.radio(
                "Optimization Method",
                optimization_options,
                index=0,
                key="optimization_mode_select",
                format_func=lambda x: optimization_label_map.get(x, x)
            )

            config['param_scope'] = st.radio(
                "Parameter Selection",
                options=["Top 3 Parameters", "Manual Selection", "Auto Search"],
                index=0,
                key="param_scope_radio"
            )

            if config['optimization_mode'] == "RandomizedSearchCV":
                if config['param_scope'] == "Manual Selection":
                    config['n_iter_random'] = st.slider(
                        "Random Search Iterations (Manual Selection)",
                        min_value=5,
                        max_value=200,
                        value=18,
                        step=1,
                        key="manual_random_n_iter_slider",
                        help="Manual Selection uses the full selected parameter space. This slider limits how many Random Search trials will be run."
                    )
                    st.caption(f"Manual Selection: Random Search will run {config['n_iter_random']} sampled trials.")
                else:
                    config['n_iter_random'] = 18
                    st.caption("Search Iterations: 18 (fixed for fair comparison)")
            elif config['optimization_mode'] == "Bayesian Optimization (skopt)" and config['model_name'] in BAYES_PARAM_SPACES:
                if config['param_scope'] == "Manual Selection":
                    config['n_iter_bayes'] = st.slider(
                        "Bayesian Optimization Iterations (Manual Selection)",
                        min_value=5,
                        max_value=200,
                        value=18,
                        step=1,
                        key="manual_bayes_n_iter_slider",
                        help="Manual Selection uses the full selected parameter space. This slider limits how many Bayesian trials will be run."
                    )
                    st.caption(f"Manual Selection: Bayesian Optimization will run {config['n_iter_bayes']} trials.")
                else:
                    config['n_iter_bayes'] = 18
                    st.caption("Search Iterations: 18 (fixed for fair comparison)")
            elif config['optimization_mode'] == "Manual Optimization" and config['param_scope'] == "Manual Selection":
                st.caption("Manual Selection + Grid Search: all selected parameter combinations will be evaluated; the 18-combination fair limit is not applied.")

            config['manual_params_by_model'] = {}
            st.markdown("**Optimization Parameters**")

            models_for_param_ui = selected_models_sidebar if selected_models_sidebar else [config['model_name']]

            if config['param_scope'] == "Top 3 Parameters":
                for m in models_for_param_ui:
                    config['manual_params_by_model'][m] = AUTO_OPTIMIZE_PARAMS.get(m, [])[:3]
                preview_params = config['manual_params_by_model'].get(config['model_name'], [])
                st.caption(f"Preview for {config['model_name']}: {', '.join(preview_params) if preview_params else 'No predefined parameters'}")

            elif config['param_scope'] == "Auto Search":
                for m in models_for_param_ui:
                    config['manual_params_by_model'][m] = AUTO_SEARCH_PARAMS.get(
                        m,
                        AUTO_OPTIMIZE_PARAMS.get(m, [])[:3]
                    )
                preview_params = config['manual_params_by_model'].get(config['model_name'], [])
                st.caption(f"Preview for {config['model_name']}: {', '.join(preview_params) if preview_params else 'No predefined parameters'}")

            else:
                st.caption("Manual Selection: choose parameters separately for each selected model.")
                for m in models_for_param_ui:
                    model_potential_params = EXTENDED_COMMON_PARAM_GRID.get(m, COMMON_PARAM_GRID.get(m, {}))
                    if not model_potential_params:
                        config['manual_params_by_model'][m] = []
                        st.warning(f"No predefined hyperparameter space is available for {m}.")
                        continue

                    with st.expander(f"{m} parameters", expanded=(m == config['model_name'])):
                        selected_optimization_params = st.multiselect(
                            "Select hyperparameters to include in the search",
                            options=list(model_potential_params.keys()),
                            default=list(model_potential_params.keys()),
                            key=f"optimize_params_multiselect_{m}"
                        )
                        config['manual_params_by_model'][m] = selected_optimization_params
                        if not selected_optimization_params:
                            st.warning("At least one hyperparameter must be selected for this model to run optimization.")

            config['manual_params'] = config['manual_params_by_model'].get(config['model_name'], [])
        else:
            config['optimization_mode'] = "None"
            config['param_scope'] = "None"
    render_plot_customization_sidebar()

    run_button_placeholder = st.empty()
    run_all_models_button_placeholder = st.empty()
    run_opt_comparison_button_placeholder = st.empty()

st.markdown("---")

if 'history' in st.session_state and st.session_state.history:
    with st.expander(" Analysis History (Comparison Table)", expanded=True):
        history_df = pd.DataFrame(st.session_state.history).sort_values(by="Analysis Time", ascending=False)
        
        column_order = [
            "Analysis Time", "Model Name", 
            "Analysis Duration (s)", "Time per Sample (ms)",
            "Training R²","Validation R²","Test R²",
            "Training RMSE","Validation RMSE", "Test RMSE", 
            "Training MAE","Validation MAE","Test MAE", 
            "Training MAPE","Validation MAPE","Test MAPE", 
            "Environmental Factors","Train %", "Val %", "Test %",
            "Opt. Status", "Opt. Mode", "Parameter Selection", "Optimized Parameters", "Opt. Param Values",
            "Splitting Method", "Sampling Period",
            "Processed Rows", "Processed Columns", "Model Input Features"
        ]
        
        existing_columns_in_order = [col for col in column_order if col in history_df.columns]
        history_df_display = history_df[existing_columns_in_order]

        formatter = {
            "Training RMSE": "{:.3f}", "Training MAE": "{:.3f}", "Training MAPE": "{:.2f}%", "Training R²": "{:.4f}",
            "Validation RMSE": "{:.3f}", "Validation MAE": "{:.3f}", "Validation MAPE": "{:.2f}%", "Validation R²": "{:.4f}",
            "Test RMSE": "{:.3f}", "Test MAE": "{:.3f}", "Test MAPE": "{:.2f}%", "Test R²": "{:.4f}",
            "Analysis Duration (s)": "{:.2f}",
            "Time per Sample (ms)": "{:.4f}"
        }
        
        st.dataframe(history_df_display.style.format(formatter), use_container_width=True)

        _download_table_as_csv_excel(
            history_df_display,
            base_filename="analysis_history_comparison_table",
            key_prefix="analysis_history_comparison_table",
            sheet_name="Analysis History",
        )
        
        if st.button("Clear Analysis History", key="clear_history", type="secondary"):
            st.session_state.history = []
            st.rerun()

if st.button("Start New Analysis (Reset All)", type="secondary"):
    st.session_state.pop('analysis_results', None)
    st.session_state.pop('analysis_duration', None)
    st.session_state.pop('start_time', None)
    st.session_state.uploader_key += 1
    st.session_state.run_analysis_triggered = False
    st.session_state.run_selected_models_triggered = False
    st.session_state.run_all_models_triggered = False
    st.session_state.run_optimization_comparison_triggered = False
    st.session_state['comparative_results'] = None
    st.session_state['comparative_model_name'] = None
    st.session_state['comparative_cv_results'] = {}
    st.session_state.selected_models_to_run = []
    st.rerun()

files_ready = bool(uploaded_file_pollutant and uploaded_file_temp and uploaded_file_hum)
selected_models_ready = bool(st.session_state.get('selected_models_multiselect_sidebar', []))

st.markdown("### Execution Controls")
st.caption("Run the selected model set, evaluate the full model set, or compare optimization strategies for the chosen target model.")

if run_button_placeholder.button("Run Analysis", type="primary", disabled=(not files_ready or not selected_models_ready), use_container_width=True):
    st.session_state.run_analysis_triggered = False
    st.session_state.run_selected_models_triggered = True
    st.session_state.run_all_models_triggered = False
    st.session_state.run_optimization_comparison_triggered = False
    st.session_state.stop_auto_run = False
    st.session_state.base_auto_run_config = config.copy()
    st.session_state.selected_models_to_run = list(dict.fromkeys(st.session_state.get('selected_models_multiselect_sidebar', [])))
    st.session_state.auto_optimize_param_keys_cache = AUTO_OPTIMIZE_PARAMS
    st.session_state['comparative_results'] = None
    st.session_state['comparative_cv_results'] = {}
    st.rerun()

if run_all_models_button_placeholder.button("Run All Models", type="secondary", disabled=not files_ready, use_container_width=True):
    st.session_state.run_analysis_triggered = False
    st.session_state.run_selected_models_triggered = False
    st.session_state.run_all_models_triggered = True
    st.session_state.run_optimization_comparison_triggered = False
    st.session_state.stop_auto_run = False
    st.session_state.base_auto_run_config = config.copy()
    st.session_state.selected_models_to_run = MODELS_TO_RUN_FIRST + MODELS_TO_RUN_LAST
    st.session_state.auto_optimize_param_keys_cache = AUTO_OPTIMIZE_PARAMS
    st.session_state['comparative_results'] = None
    st.session_state['comparative_cv_results'] = {}
    st.rerun()

optimization_target_ready = bool(st.session_state.get('optimization_target_model_sidebar'))
if run_opt_comparison_button_placeholder.button("Compare Optimization Methods", type="secondary", disabled=(not files_ready or not optimization_target_ready), use_container_width=True):
    st.session_state.run_analysis_triggered = False
    st.session_state.run_selected_models_triggered = False
    st.session_state.run_all_models_triggered = False
    st.session_state.run_optimization_comparison_triggered = True
    st.session_state.stop_auto_run = False
    st.session_state.opt_comp_config = config.copy()
    st.session_state.opt_comp_config['model_name'] = st.session_state.get('optimization_target_model_sidebar')
    # Use comparison-safe defaults to reduce hanging on large search spaces.
    st.session_state.opt_comp_config['param_scope'] = "Top 3 Parameters"
    st.session_state.opt_comp_config['manual_params'] = AUTO_OPTIMIZE_PARAMS.get(
        st.session_state.opt_comp_config['model_name'],
        []
    )[:3]
    st.session_state.opt_comp_config['n_iter_random'] = 18
    st.session_state.opt_comp_config['n_iter_bayes'] = 18
    st.session_state['comparative_results'] = None
    st.session_state['comparative_cv_results'] = {}
    st.rerun()

if not files_ready:
    st.info("Upload the required pollutant, temperature, and humidity datasets from the Control Panel to enable analysis.")
elif not selected_models_ready:
    st.caption('Select at least one model in "Models to Run" to enable analysis.')
elif not st.session_state.get("optimization_target_model_sidebar"):
    st.caption("Select an Optimization Target Model to run optimization comparison.")

main_content_area = st.container() 
progress_status_container = main_content_area.empty()
stop_button_container = main_content_area.empty()

if st.session_state.run_analysis_triggered:
    with progress_status_container:
        status_text_placeholder = st.empty()
        progress_bar_placeholder = st.empty()

    status_text_placeholder.info("Single model analysis in progress...")
    
    analysis_results = run_model_analysis(
        st.session_state.run_config, 
        uploaded_file_pollutant, 
        uploaded_file_temp, 
        uploaded_file_hum,
        progress_bar_placeholder,
        status_text_placeholder
    )
    
    if analysis_results:
        st.session_state['analysis_results'] = analysis_results
        st.session_state.results_by_model[analysis_results['model_name']] = analysis_results
        
        res = analysis_results
        env_factors = [f for f in res['features'] if f != "Raw Pollutant"]
        env_factors_str = ", ".join(env_factors) if env_factors else "None"
        
        opt_param_values_str = "N/A"
        if res.get('optimized', False) and res.get('best_params'):
            if isinstance(res['best_params'], dict):
                params_to_display = dict(res['best_params'])
            else:
                params_to_display = res['best_params']
            opt_param_values_str = ", ".join([f"{k}: {v}" for k, v in params_to_display.items()])

        
        run_summary_detailed = {
            "Analysis Time": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            "Model Name": res['model_name'],
            "Opt. Status": "Yes" if res['optimized'] else "No",
            "Opt. Mode": format_optimization_mode_label(res.get('optimization_mode', 'N/A')),
            "Parameter Selection": res.get('param_scope', 'N/A'),
            "Optimized Parameters": ", ".join(res['best_params'].keys()) if res.get('optimized', False) and res.get('best_params') else "None",
            "Opt. Param Values": opt_param_values_str,
            "Training R²": res['train_metrics']['r2'],
            "Validation R²": res['val_metrics']['r2'],
            "Test R²": res['test_metrics']['r2'],
            "Training RMSE": res['train_metrics']['rmse'],
            "Validation RMSE": res['val_metrics']['rmse'],
            "Test RMSE": res['test_metrics']['rmse'],
            "Training MAE": res['train_metrics']['mae'],
            "Validation MAE": res['val_metrics']['mae'],
            "Test MAE": res['test_metrics']['mae'],
            "Training MAPE": res['train_metrics']['mape'],
            "Validation MAPE": res['val_metrics']['mape'],
            "Test MAPE": res['test_metrics']['mape'],
            "Environmental Factors": env_factors_str,
            "Train %": res['train_perc'],
            "Val %": res['val_perc'],
            "Test %": res['test_perc'],
            "Analysis Duration (s)": res['analysis_duration'],
            "Time per Sample (ms)": (res['analysis_duration'] / res['df_processed'].shape[0]) * 1000 if res['df_processed'].shape[0] > 0 else np.nan,
            "Sampling Period": res['interval'],
            "Splitting Method": res['split_method'],
            "Processed Rows": res['df_processed'].shape[0],
            "Processed Columns": res['df_processed'].shape[1],
            "Model Input Features": ", ".join(res.get('feature_names_for_model', []))
        }
        st.session_state.history.append(run_summary_detailed)
        status_text_placeholder.success(f"Analysis for {st.session_state.run_config['model_name']} completed in {st.session_state.analysis_duration:.2f} seconds.")
    
    st.session_state.run_analysis_triggered = False
    st.rerun()

elif st.session_state.run_selected_models_triggered:
    selected_model_order = list(dict.fromkeys(st.session_state.get('selected_models_to_run', [])))
    total_models = len(selected_model_order)

    with progress_status_container:
        status_text_placeholder = st.empty()
        progress_bar_placeholder = st.empty()

    if stop_button_container.button("Stop Analysis", key="stop_selected_models_button", type="secondary"):
        st.session_state.stop_auto_run = True
        stop_button_container.empty()
        status_text_placeholder.warning("Stop signal received. Finishing the current model, then stopping the analysis.")

    analysis_results = None
    for i, model_name in enumerate(selected_model_order):
        if st.session_state.stop_auto_run:
            status_text_placeholder.info("Analysis interrupted by user.")
            break

        current_progress_percent = int((i / total_models) * 100) if total_models else 0
        status_text_placeholder.info(f"Running model {i+1}/{total_models}: {model_name}...")
        progress_bar_obj = progress_bar_placeholder.progress(current_progress_percent, text=f"Starting analysis for {model_name}...")
        st.session_state.start_time = time.time()

        temp_config = st.session_state.base_auto_run_config.copy()
        temp_config['model_name'] = model_name
        # Batch/selected-model optimization: keep the selected optimization method
        # and apply it to every selected model with that model's own parameters.
        if temp_config.get('optimize', False):
            param_scope = temp_config.get('param_scope', 'Top 3 Parameters')
            if param_scope != 'Manual Selection':
                temp_config['n_iter_random'] = 18
                temp_config['n_iter_bayes'] = 18
            else:
                temp_config['n_iter_random'] = temp_config.get('n_iter_random', 18)
                temp_config['n_iter_bayes'] = temp_config.get('n_iter_bayes', 18)
            manual_params_by_model = temp_config.get('manual_params_by_model', {}) or {}

            if param_scope == 'Top 3 Parameters':
                temp_config['manual_params'] = AUTO_OPTIMIZE_PARAMS.get(model_name, [])[:3]
            elif param_scope == 'Auto Search':
                temp_config['manual_params'] = AUTO_SEARCH_PARAMS.get(
                    model_name,
                    AUTO_OPTIMIZE_PARAMS.get(model_name, [])[:3]
                )
            elif param_scope == 'Manual Selection':
                full_params = EXTENDED_COMMON_PARAM_GRID.get(model_name, COMMON_PARAM_GRID.get(model_name, {}))
                temp_config['manual_params'] = manual_params_by_model.get(model_name, list(full_params.keys()))
            else:
                temp_config['manual_params'] = AUTO_OPTIMIZE_PARAMS.get(model_name, [])[:3]

        analysis_results = run_model_analysis(
            temp_config,
            uploaded_file_pollutant,
            uploaded_file_temp,
            uploaded_file_hum,
            progress_bar_obj,
            status_text_placeholder
        )

        if analysis_results and not st.session_state.stop_auto_run:
            st.session_state['analysis_results'] = analysis_results
            st.session_state.results_by_model[analysis_results['model_name']] = analysis_results

            res = analysis_results
            env_factors = [f for f in res['features'] if f != "Raw Pollutant"]
            env_factors_str = ", ".join(env_factors) if env_factors else "None"

            opt_param_values_str = "N/A"
            if res.get('optimized', False) and res.get('best_params'):
                params_to_display = dict(res['best_params']) if isinstance(res['best_params'], dict) else res['best_params']
                opt_param_values_str = ", ".join([f"{k}: {v}" for k, v in params_to_display.items()])

            run_summary_detailed = {
                "Analysis Time": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                "Model Name": res['model_name'],
                "Opt. Status": "Yes" if res['optimized'] else "No",
                "Opt. Mode": format_optimization_mode_label(res.get('optimization_mode', 'N/A')),
            "Parameter Selection": res.get('param_scope', 'N/A'),
                "Optimized Parameters": ", ".join(res['best_params'].keys()) if res.get('optimized', False) and res.get('best_params') else "None",
                "Opt. Param Values": opt_param_values_str,
                "Training R²": res['train_metrics']['r2'],
                "Validation R²": res['val_metrics']['r2'],
                "Test R²": res['test_metrics']['r2'],
                "Training RMSE": res['train_metrics']['rmse'],
                "Validation RMSE": res['val_metrics']['rmse'],
                "Test RMSE": res['test_metrics']['rmse'],
                "Training MAE": res['train_metrics']['mae'],
                "Validation MAE": res['val_metrics']['mae'],
                "Test MAE": res['test_metrics']['mae'],
                "Training MAPE": res['train_metrics']['mape'],
                "Validation MAPE": res['val_metrics']['mape'],
                "Test MAPE": res['test_metrics']['mape'],
                "Environmental Factors": env_factors_str,
                "Train %": res['train_perc'],
                "Val %": res['val_perc'],
                "Test %": res['test_perc'],
                "Analysis Duration (s)": res['analysis_duration'],
                "Time per Sample (ms)": (res['analysis_duration'] / res['df_processed'].shape[0]) * 1000 if res['df_processed'].shape[0] > 0 else np.nan,
                "Sampling Period": res['interval'],
                "Splitting Method": res['split_method'],
                "Processed Rows": res['df_processed'].shape[0],
                "Processed Columns": res['df_processed'].shape[1],
                "Model Input Features": ", ".join(res.get('feature_names_for_model', []))
            }
            st.session_state.history.append(run_summary_detailed)

    progress_bar_placeholder.empty()
    status_text_placeholder.empty()

    if analysis_results and not st.session_state.stop_auto_run:
        status_text_placeholder.success("Analysis completed successfully!")
    else:
        status_text_placeholder.info("Analysis completed or was interrupted by the user.")

    st.session_state.run_selected_models_triggered = False
    st.rerun()

if st.session_state.run_all_models_triggered:
    total_models = len(MODELS_TO_RUN_FIRST) + len(MODELS_TO_RUN_LAST)
    combined_model_order = MODELS_TO_RUN_FIRST + MODELS_TO_RUN_LAST

    with progress_status_container:
        status_text_placeholder = st.empty()
        progress_bar_placeholder = st.empty()

    if stop_button_container.button("Stop All-Model Analysis", key="stop_auto_run_batch_button", type="secondary"):
        st.session_state.stop_auto_run = True
        stop_button_container.empty()
        status_text_placeholder.warning("Stop signal received. Finishing current model, then stopping batch analysis.")

    for i, model_name in enumerate(combined_model_order):
        if st.session_state.stop_auto_run:
            status_text_placeholder.info("Batch analysis interrupted by user.")
            break

        current_progress_percent = int(((i) / total_models) * 100)
        status_text_placeholder.info(f"Running model {i+1}/{total_models}: {model_name}...")
        
        progress_bar_obj = progress_bar_placeholder.progress(current_progress_percent, text=f"Starting analysis for {model_name}...")
        
        st.session_state.start_time = time.time()

        temp_config = st.session_state.base_auto_run_config.copy()
        temp_config['model_name'] = model_name
        # Batch/selected-model optimization: keep the selected optimization method
        # and apply it to every selected model with that model's own parameters.
        if temp_config.get('optimize', False):
            param_scope = temp_config.get('param_scope', 'Top 3 Parameters')
            if param_scope != 'Manual Selection':
                temp_config['n_iter_random'] = 18
                temp_config['n_iter_bayes'] = 18
            else:
                temp_config['n_iter_random'] = temp_config.get('n_iter_random', 18)
                temp_config['n_iter_bayes'] = temp_config.get('n_iter_bayes', 18)
            manual_params_by_model = temp_config.get('manual_params_by_model', {}) or {}

            if param_scope == 'Top 3 Parameters':
                temp_config['manual_params'] = AUTO_OPTIMIZE_PARAMS.get(model_name, [])[:3]
            elif param_scope == 'Auto Search':
                temp_config['manual_params'] = AUTO_SEARCH_PARAMS.get(
                    model_name,
                    AUTO_OPTIMIZE_PARAMS.get(model_name, [])[:3]
                )
            elif param_scope == 'Manual Selection':
                full_params = EXTENDED_COMMON_PARAM_GRID.get(model_name, COMMON_PARAM_GRID.get(model_name, {}))
                temp_config['manual_params'] = manual_params_by_model.get(model_name, list(full_params.keys()))
            else:
                temp_config['manual_params'] = AUTO_OPTIMIZE_PARAMS.get(model_name, [])[:3]
        

        analysis_results = run_model_analysis(
            temp_config, 
            uploaded_file_pollutant, 
            uploaded_file_temp, 
            uploaded_file_hum,
            progress_bar_obj,
            status_text_placeholder
        )
        
        if analysis_results and not st.session_state.stop_auto_run:
            st.session_state['analysis_results'] = analysis_results
            st.session_state.results_by_model[analysis_results['model_name']] = analysis_results
            res = analysis_results
            
            env_factors = [f for f in res['features'] if f != "Raw Pollutant"]
            env_factors_str = ", ".join(env_factors) if env_factors else "None"

            opt_param_values_str = "N/A"
            if res.get('optimized', False) and res.get('best_params'):
                if isinstance(res['best_params'], dict):
                    params_to_display = dict(res['best_params'])
                else:
                    params_to_display = res['best_params']
                opt_param_values_str = ", ".join([f"{k}: {v}" for k, v in params_to_display.items()])

            run_summary_detailed = {
                "Analysis Time": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                "Model Name": res['model_name'],
                "Opt. Status": "Yes" if res['optimized'] else "No",
                "Opt. Mode": format_optimization_mode_label(res.get('optimization_mode', 'N/A')),
            "Parameter Selection": res.get('param_scope', 'N/A'),
                "Optimized Parameters": ", ".join(res['best_params'].keys()) if res.get('optimized', False) and res.get('best_params') else "None",
                "Opt. Param Values": opt_param_values_str,
                "Training R²": res['train_metrics']['r2'],
                "Validation R²": res['val_metrics']['r2'],
                "Test R²": res['test_metrics']['r2'],
                "Training RMSE": res['train_metrics']['rmse'],
                "Validation RMSE": res['val_metrics']['rmse'],
                "Test RMSE": res['test_metrics']['rmse'],
                "Training MAE": res['train_metrics']['mae'],
                "Validation MAE": res['val_metrics']['mae'],
                "Test MAE": res['test_metrics']['mae'],
                "Training MAPE": res['train_metrics']['mape'],
                "Validation MAPE": res['val_metrics']['mape'],
                "Test MAPE": res['test_metrics']['mape'],
                "Environmental Factors": env_factors_str,
                "Train %": res['train_perc'],
                "Val %": res['val_perc'],
                "Test %": res['test_perc'],
                "Analysis Duration (s)": res['analysis_duration'],
                "Time per Sample (ms)": (res['analysis_duration'] / res['df_processed'].shape[0]) * 1000 if res['df_processed'].shape[0] > 0 else np.nan,
                "Sampling Period": res['interval'],
                "Splitting Method": res['split_method'],
                "Processed Rows": res['df_processed'].shape[0],
                "Processed Columns": res['df_processed'].shape[1],
                "Model Input Features": ", ".join(res.get('feature_names_for_model', []))
            }
            st.session_state.history.append(run_summary_detailed)

    progress_bar_placeholder.empty()
    status_text_placeholder.empty()

    if i == total_models - 1 and analysis_results:
        status_text_placeholder.success("Batch analysis completed successfully!")
    else:
        status_text_placeholder.info("Batch analysis completed or interrupted by user.")

    st.session_state.run_all_models_triggered = False
    st.rerun()
    
elif st.session_state.run_optimization_comparison_triggered:
    with progress_status_container:
        status_text_placeholder = st.empty()
        progress_bar_placeholder = st.empty()
        stop_button_container.button(" Stop Comparison", key="stop_opt_comp_button", type="secondary")
    
    model_to_compare = st.session_state.opt_comp_config['model_name']
    optimizers_to_compare = ["GridSearchCV", "RandomizedSearchCV"]
    if _SKOPT_AVAILABLE:
        optimizers_to_compare.append("Bayesian Optimization")
        
    results_df, cv_results_dict, pollutant_unit, display_unit = run_comparative_optimization_analysis(
        model_to_compare,
        optimizers_to_compare,
        st.session_state.opt_comp_config,
        uploaded_file_pollutant,
        uploaded_file_temp,
        uploaded_file_hum,
        progress_bar_placeholder,
        status_text_placeholder
    )
    
    if results_df is not None:
        st.session_state['comparative_results'] = results_df
        st.session_state['comparative_cv_results'] = cv_results_dict
        st.session_state['comparative_model_name'] = model_to_compare
        st.session_state['comparative_display_unit'] = display_unit
        
    st.session_state.run_optimization_comparison_triggered = False
    st.rerun()

with main_content_area:
    if 'analysis_results' in st.session_state and st.session_state.analysis_results and not st.session_state.run_analysis_triggered and not st.session_state.run_selected_models_triggered and not st.session_state.run_all_models_triggered and not st.session_state.run_optimization_comparison_triggered:
        display_selected_model_results()
        if st.session_state.analysis_duration is not None:
            st.success(f"Last analysis completed in {st.session_state.analysis_duration:.2f} seconds.")

    render_results_summary_panel()
    render_explainability_summary_panel()

    if 'comparative_results' in st.session_state and st.session_state.comparative_results is not None:
        if not st.session_state.comparative_results.empty:
            display_optimization_comparison_results(
                st.session_state.comparative_results, 
                st.session_state.comparative_cv_results,
                st.session_state.comparative_model_name,
                st.session_state.comparative_display_unit
            )
