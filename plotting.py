import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas.plotting import parallel_coordinates
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

from config import DEFAULT_PLOT_STYLES, PLOTLY_CONFIG

try:
    import scipy.stats as stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    _SEABORN_AVAILABLE = True
except ImportError:
    _SEABORN_AVAILABLE = False

def _unique_plotly_key(prefix="plotly_chart"):
    if "_plotly_key_counter" not in st.session_state:
        st.session_state["_plotly_key_counter"] = 0
    st.session_state["_plotly_key_counter"] += 1
    return f"{prefix}_{st.session_state['_plotly_key_counter']}"


def _render_plotly(fig, use_container_width=True, config=None, key=None):
    st.plotly_chart(
        fig,
        use_container_width=use_container_width,
        config=config if config is not None else PLOTLY_CONFIG,
        key=(f"{key}_{_unique_plotly_key()}" if key else _unique_plotly_key()),
    )


def plot_time_series(df_plot, pollutant_unit, location_name, model_name, display_unit, plot_config):
    # Dynamic Titles and Labels
    graph_title = plot_config["time_series"]["title"] or (f"{model_name} Model Time Series - {location_name.capitalize()} {pollutant_unit}" if location_name != 'Mean' else f"{model_name} Model Time Series - Mean {pollutant_unit}")
    xaxis_title = plot_config["time_series"]["xaxis_title"]
    yaxis_title = plot_config["time_series"]["yaxis_title"] or (f"{location_name.capitalize()} {pollutant_unit} [{display_unit}]" if location_name != 'Mean' else f"Mean {pollutant_unit} [{display_unit}]")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_plot['timestamp'], y=df_plot['raw_pollutant'], mode='lines', name='Raw Sensor',
        line=dict(color=plot_config["time_series"]["raw_color"], width=plot_config["time_series"]["raw_width"], dash=plot_config["time_series"]["raw_style"]),
        opacity=plot_config["time_series"]["raw_opacity"]
    ))
    fig.add_trace(go.Scatter(
        x=df_plot['timestamp'], y=df_plot['calibrated_pollutant'], mode='lines', name='Calibrated (Model)',
        line=dict(color=plot_config["time_series"]["calibrated_color"], width=plot_config["time_series"]["calibrated_width"], dash=plot_config["time_series"]["calibrated_style"]),
        opacity=plot_config["time_series"]["calibrated_opacity"]
    ))
    fig.add_trace(go.Scatter(
        x=df_plot['timestamp'], y=df_plot['reference_pollutant'], mode='lines', name='Reference Device',
        line=dict(color=plot_config["time_series"]["reference_color"], width=plot_config["time_series"]["reference_width"], dash=plot_config["time_series"]["reference_style"])
    ))
    
    fig.update_layout(
        title=dict( # Başlık ayarı
            text=graph_title,
            font=dict(
                size=plot_config["general"]["plot_title_font_size"], # Yeni eklenen başlık font boyutu
                family=plot_config["general"]["font_family"],
                color=plot_config["general"]["font_color"]
            )
        ),
        xaxis_title=xaxis_title, 
        yaxis_title=yaxis_title, 
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor=plot_config["time_series"]["legend_bgcolor"], bordercolor="Black", borderwidth=1),
        template=plot_config["general"]["template"], # Apply general template
        font=dict(family=plot_config["general"]["font_family"], color=plot_config["general"]["font_color"]), # Apply general font
        xaxis=dict( # Eksen ayarları
            title=dict( # Eksen başlığı font ayarı
                text=xaxis_title,
                font=dict(
                    size=plot_config["general"]["axis_title_font_size"],
                    color=plot_config["general"]["axis_title_font_color"],
                    family=plot_config["general"]["axis_title_font_family"]
                )
            ),
            tickfont=dict( # Eksen tik etiketleri font ayarı
                size=plot_config["general"]["axis_tick_font_size"],
                color=plot_config["general"]["axis_tick_font_color"],
                family=plot_config["general"]["axis_tick_font_family"]
            )
        ),
        yaxis=dict( # Eksen ayarları
            title=dict( # Eksen başlığı font ayarı
                text=yaxis_title,
                font=dict(
                    size=plot_config["general"]["axis_title_font_size"],
                    color=plot_config["general"]["axis_title_font_color"],
                    family=plot_config["general"]["axis_title_font_family"]
                )
            ),
            tickfont=dict( # Eksen tik etiketleri font ayarı
                size=plot_config["general"]["axis_tick_font_size"],
                color=plot_config["general"]["axis_tick_font_color"],
                family=plot_config["general"]["axis_tick_font_family"]
            )
        )
    )
    _render_plotly(fig, use_container_width=True, config=PLOTLY_CONFIG)

def plot_scatter(y_test, y_pred, pollutant_unit, location_name, model_name, display_unit, plot_config, chart_key=None):
    r2 = r2_score(y_test, y_pred); rmse = np.sqrt(mean_squared_error(y_test, y_pred)); mae = mean_absolute_error(y_test, y_pred)
    N = len(y_test) # Data point count

    # Dynamic Titles and Labels
    graph_title = plot_config["scatter_plot"]["title"] or (f"{model_name} Predictions for {location_name.capitalize()} {pollutant_unit}" if location_name != 'Mean' else f"{model_name} Predictions for Mean {pollutant_unit}")
    xaxis_title = plot_config["scatter_plot"]["xaxis_title"] or f"Reference Values [{display_unit}]"
    yaxis_title = plot_config["scatter_plot"]["yaxis_title"] or (f"{model_name} Predictions for {location_name.capitalize()} {pollutant_unit} [{display_unit}]" if location_name != 'Mean' else f"{model_name} Predictions for Mean {pollutant_unit} [{display_unit}]")

    # --- DENSITY CALCULATION AND DATAFRAME CREATION START ---
    scatter_df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})

    range_min = min(scatter_df['y_test'].min(), scatter_df['y_pred'].min())
    range_max = max(scatter_df['y_test'].max(), scatter_df['y_pred'].max())
    bins = 50 # Number of bins for density, adjustable

    H, xedges, yedges = np.histogram2d(scatter_df['y_test'], scatter_df['y_pred'], bins=bins, 
                                        range=[[range_min, range_max], [range_min, range_max]])

    x_bin_indices = np.digitize(scatter_df['y_test'], xedges) - 1
    y_bin_indices = np.digitize(scatter_df['y_pred'], yedges) - 1

    x_bin_indices = np.clip(x_bin_indices, 0, bins - 1)
    y_bin_indices = np.clip(y_bin_indices, 0, bins - 1)

    density = np.array([H[x_idx, y_idx] 
                        for x_idx, y_idx in zip(x_bin_indices, y_bin_indices)])
    scatter_df['density'] = density
    # --- DENSITY CALCULATION AND DATAFRAME CREATION END ---

    fig = px.scatter(
        scatter_df, 
        x='y_test', 
        y='y_pred', 
        color='density', # Color by density
        color_continuous_scale=px.colors.sequential.Jet, # Color scale for density (as in example)
        labels={'y_test': xaxis_title, 'y_pred': yaxis_title, 'density': 'Point Density'}, # Legend title for density
        opacity=plot_config["scatter_plot"]["marker_opacity"],
        trendline="ols",
        trendline_color_override=plot_config["scatter_plot"]["trendline_color"]
    )
    
    # --- ADD IDEAL 1:1 LINE START ---
    min_val_plot = min(y_test.min(), y_pred.min())
    max_val_plot = max(y_test.max(), y_pred.max())
    ideal_line_range = [min_val_plot, max_val_plot]

    fig.add_trace(go.Scatter(
        x=ideal_line_range,
        y=ideal_line_range,
        mode='lines',
        name='Ideal 1:1 Line',
        line=dict(color='black', dash='dash', width=2),
        showlegend=True
    ))

    # Ensure the OLS trendline is labeled clearly in the legend
    for trace in fig.data:
        if getattr(trace, "mode", None) == "lines" and trace.name != "Ideal 1:1 Line":
            trace.name = "Regression Line"
            trace.showlegend = True

    fig.update_traces(
        marker=dict(size=plot_config["scatter_plot"]["marker_size"]), # Color now comes from density, removed marker_color
        selector=dict(mode='markers')
    )
    fig.update_traces(
        line=dict(width=plot_config["scatter_plot"]["trendline_width"], dash=plot_config["scatter_plot"]["trendline_style"]),
        selector=lambda tr: getattr(tr, "mode", None) == "lines" and tr.name == "Regression Line"
    )
    
    # --- UPDATE ANNOTATION (add N value) ---
    fig.add_annotation(x=0.05, y=0.95, xref="paper", yref="paper", 
                       text=f"N = {N}<br>R² = {r2:.4f}<br>RMSE = {rmse:.2f}<br>MAE = {mae:.2f}", # N added
                       showarrow=False, align="left", bordercolor="black", borderwidth=1, bgcolor="#FFFFFF")
    
    fig.update_layout(
        title=dict( # Başlık ayarı
            text=graph_title,
            font=dict(
                size=plot_config["general"]["plot_title_font_size"], # Yeni eklenen başlık font boyutu
                family=plot_config["general"]["font_family"],
                color=plot_config["general"]["font_color"]
            )
        ),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template=plot_config["general"]["template"],
        font=dict(family=plot_config["general"]["font_family"], color=plot_config["general"]["font_color"]),
        coloraxis_colorbar=dict( # Settings for density scale
            title=dict(text="Point Density"), 
            len=1,   # Extend along Y-axis (normalized from 0 to 1)
            y=0.5,   # Vertically center
            x=1.08 # Adjust color bar position (to avoid being too far right)
        ),
        legend=dict( # Move legend to bottom right
            x=0.90,  # Close to right edge
            y=0.01,  # Close to bottom
            xanchor="right", # Align to right edge
            yanchor="bottom", # Align to bottom edge
            bgcolor="rgba(255,255,255,0.7)", # Background for readability
            bordercolor="Black",
            borderwidth=1
        ),
        # --- Axis tick label settings ---
        xaxis=dict(
            tickfont=dict( # Eksen tik etiketleri font ayarı (artık genel ayarları kullanıyor)
                size=plot_config["general"]["axis_tick_font_size"],
                color=plot_config["general"]["axis_tick_font_color"],
                family=plot_config["general"]["axis_tick_font_family"]
            ),
            title=dict( # Eksen başlığı font ayarı
                text=xaxis_title,
                font=dict(
                    size=plot_config["general"]["axis_title_font_size"],
                    color=plot_config["general"]["axis_title_font_color"],
                    family=plot_config["general"]["axis_title_font_family"]
                )
            )
        ),
        yaxis=dict(
            tickfont=dict( # Eksen tik etiketleri font ayarı (artık genel ayarları kullanıyor)
                size=plot_config["general"]["axis_tick_font_size"],
                color=plot_config["general"]["axis_tick_font_color"],
                family=plot_config["general"]["axis_tick_font_family"]
            ),
            title=dict( # Eksen başlığı font ayarı
                text=yaxis_title,
                font=dict(
                    size=plot_config["general"]["axis_title_font_size"],
                    color=plot_config["general"]["axis_title_font_color"],
                    family=plot_config["general"]["axis_title_font_family"]
                )
            )
        )
        # --- End of Axis tick label settings ---
    )
    _render_plotly(fig, use_container_width=True, config=PLOTLY_CONFIG, key=chart_key)

def plot_scatter_seaborn(y_test, y_pred, pollutant_unit, location_name, model_name, display_unit, plot_config, chart_key=None):
    if not _SEABORN_AVAILABLE:
        st.warning("Seaborn or Matplotlib library not installed. Cannot plot with this function.")
        return
    
    r2 = r2_score(y_test, y_pred); rmse = np.sqrt(mean_squared_error(y_test, y_pred)); mae = mean_absolute_error(y_test, y_pred)
    N = len(y_test)
    
    sns.set_style("whitegrid", {"axes.facecolor": "#FFFFFF"})
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.regplot(
        x=y_test, y=y_pred, 
        scatter_kws={'alpha': 0.5},
        line_kws={'color': 'red', 'label': 'Regression Line'},
        ax=ax
    )
    
    min_val_plot = min(y_test.min(), y_pred.min())
    max_val_plot = max(y_test.max(), y_pred.max())
    ax.plot([min_val_plot, max_val_plot], [min_val_plot, max_val_plot], linestyle='--', color='black', label='Ideal 1:1 Line')

    ax.set_title(f"{model_name} Predictions for {location_name.capitalize()} {pollutant_unit}")
    ax.set_xlabel(f"Reference Values [{display_unit}]")
    ax.set_ylabel(f"Predictions [{display_unit}]")
    ax.text(0.05, 0.95, f"N = {N}\n$R^2$ = {r2:.4f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}",
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_residuals(y_test, y_pred, pollutant_unit, location_name, model_name, display_unit, plot_config):
    residuals = y_test - y_pred
    
    # Dynamic Titles and Labels
    graph_title = plot_config["residuals_plot"]["title"] or (f"{model_name} Model Residuals Plot - {location_name.capitalize()} {pollutant_unit}" if location_name != 'Mean' else f"{model_name} Model Residuals Plot - Mean {pollutant_unit}")
    xaxis_title = plot_config["residuals_plot"]["xaxis_title"] or (f'{model_name} Model Predictions for {location_name.capitalize()} {pollutant_unit} [{display_unit}]' if location_name != 'Mean' else f'{model_name} Model Predictions for Mean {pollutant_unit} [{display_unit}]')
    yaxis_title = plot_config["residuals_plot"]["yaxis_title"]

    fig = px.scatter(
        x=y_pred, y=residuals, 
        labels={'x': xaxis_title, 'y': yaxis_title},
        opacity=plot_config["residuals_plot"]["marker_opacity"]
    )
    fig.update_traces(
        marker=dict(color=plot_config["residuals_plot"]["marker_color"], size=plot_config["residuals_plot"]["marker_size"])
    )
    fig.add_hline(
        y=0, 
        line_color=plot_config["residuals_plot"]["zeroline_color"], 
        line_dash=plot_config["residuals_plot"]["zeroline_style"],
        line_width=plot_config["residuals_plot"]["zeroline_width"]
    );
    
    fig.update_layout(
        title=dict( # Başlık ayarı
            text=graph_title,
            font=dict(
                size=plot_config["general"]["plot_title_font_size"], # Yeni eklenen başlık font boyutu
                family=plot_config["general"]["font_family"],
                color=plot_config["general"]["font_color"]
            )
        ),
        template=plot_config["general"]["template"], # Apply general template
        font=dict(family=plot_config["general"]["font_family"], color=plot_config["general"]["font_color"]), # Apply general font
        xaxis=dict( # Eksen ayarları
            title=dict( # Eksen başlığı font ayarı
                text=xaxis_title,
                font=dict(
                    size=plot_config["general"]["axis_title_font_size"],
                    color=plot_config["general"]["axis_title_font_color"],
                    family=plot_config["general"]["axis_title_font_family"]
                )
            ),
            tickfont=dict( # Eksen tik etiketleri font ayarı
                size=plot_config["general"]["axis_tick_font_size"],
                color=plot_config["general"]["axis_tick_font_color"],
                family=plot_config["general"]["axis_tick_font_family"]
            )
        ),
        yaxis=dict( # Eksen ayarları
            title=dict( # Eksen başlığı font ayarı
                text=yaxis_title,
                font=dict(
                    size=plot_config["general"]["axis_title_font_size"],
                    color=plot_config["general"]["axis_title_font_color"],
                    family=plot_config["general"]["axis_title_font_family"]
                )
            ),
            tickfont=dict( # Eksen tik etiketleri font ayarı
                size=plot_config["general"]["axis_tick_font_size"],
                color=plot_config["general"]["axis_tick_font_color"],
                family=plot_config["general"]["axis_tick_font_family"]
            )
        )
    )
    _render_plotly(fig, use_container_width=True, config=PLOTLY_CONFIG)

def plot_residuals_seaborn(y_test, y_pred, pollutant_unit, location_name, model_name, display_unit, plot_config):
    if not _SEABORN_AVAILABLE:
        st.warning("Seaborn or Matplotlib library not installed. Cannot plot with this function.")
        return

    sns.set_style("whitegrid", {"axes.facecolor": "#FFFFFF"})
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.residplot(
        x=y_pred, y=y_test,
        scatter_kws={'alpha': 0.5},
        line_kws={'color': 'red', 'lw': 2},
        ax=ax
    )

    ax.set_title(f"{model_name} Model Residuals Plot - {location_name.capitalize()} {pollutant_unit}")
    ax.set_xlabel(f"Predictions [{display_unit}]")
    ax.set_ylabel(f"Residuals [{display_unit}]")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# --- NEWLY ADDED GRAPH FUNCTIONS ---
def plot_residuals_histogram(y_test, y_pred, pollutant_unit, location_name, model_name, display_unit, plot_config):
    residuals = y_test - y_pred
    
    graph_title = plot_config["residuals_hist"]["title"] or (f"{model_name} Model Residuals Distribution (Histogram) - {location_name.capitalize()} {pollutant_unit}" if location_name != 'Mean' else f"{model_name} Model Residuals Distribution (Histogram) - Mean {pollutant_unit}")
    xaxis_title = plot_config["residuals_hist"]["xaxis_title"] or f"Residuals [{display_unit}]"
    yaxis_title = plot_config["residuals_hist"]["yaxis_title"]
    
    fig = px.histogram(
        x=residuals,
        nbins=50, # Default bin count, adjustable
        title=graph_title,
        labels={'x': xaxis_title, 'y': yaxis_title}
    )
    fig.update_traces(marker_color=plot_config["residuals_hist"]["bar_color"], opacity=plot_config["residuals_hist"]["bar_opacity"])
    
    # Mean and Median Lines
    fig.add_vline(x=np.mean(residuals), line_width=plot_config["residuals_hist"]["line_width"], line_dash="dash", line_color=plot_config["residuals_hist"]["line_color"], annotation_text=f"Mean: {np.mean(residuals):.2f}", annotation_position="top right")
    fig.add_vline(x=np.median(residuals), line_width=plot_config["residuals_hist"]["line_width"], line_dash="dot", line_color=plot_config["residuals_hist"]["line_color"], annotation_text=f"Median: {np.median(residuals):.2f}", annotation_position="bottom right")

    fig.update_layout(
        title=dict( # Başlık ayarı
            text=graph_title,
            font=dict(
                size=plot_config["general"]["plot_title_font_size"], # Yeni eklenen başlık font boyutu
                family=plot_config["general"]["font_family"],
                color=plot_config["general"]["font_color"]
            )
        ),
        template=plot_config["general"]["template"],
        font=dict(family=plot_config["general"]["font_family"], color=plot_config["general"]["font_color"]),
        xaxis=dict( # Eksen ayarları
            title=dict( # Eksen başlığı font ayarı
                text=xaxis_title,
                font=dict(
                    size=plot_config["general"]["axis_title_font_size"],
                    color=plot_config["general"]["axis_title_font_color"],
                    family=plot_config["general"]["axis_title_font_family"]
                )
            ),
            tickfont=dict( # Eksen tik etiketleri font ayarı
                size=plot_config["general"]["axis_tick_font_size"],
                color=plot_config["general"]["axis_tick_font_color"],
                family=plot_config["general"]["axis_tick_font_family"]
            )
        ),
        yaxis=dict( # Eksen ayarları
            title=dict( # Eksen başlığı font ayarı
                text=yaxis_title,
                font=dict(
                    size=plot_config["general"]["axis_title_font_size"],
                    color=plot_config["general"]["axis_title_font_color"],
                    family=plot_config["general"]["axis_title_font_family"]
                )
            ),
            tickfont=dict( # Eksen tik etiketleri font ayarı
                size=plot_config["general"]["axis_tick_font_size"],
                color=plot_config["general"]["axis_tick_font_color"],
                family=plot_config["general"]["axis_tick_font_family"]
            )
        )
    )
    _render_plotly(fig, use_container_width=True, config=PLOTLY_CONFIG)
    
def plot_residuals_histogram_seaborn(y_test, y_pred, pollutant_unit, location_name, model_name, display_unit, plot_config):
    if not _SEABORN_AVAILABLE:
        st.warning("Seaborn or Matplotlib library not installed. Cannot plot with this function.")
        return
    
    residuals = y_test - y_pred
    sns.set_style("whitegrid", {"axes.facecolor": "#FFFFFF"})
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.histplot(residuals, kde=False, bins=50, ax=ax, color='skyblue')
    ax.axvline(x=np.mean(residuals), color='red', linestyle='--', label=f'Mean: {np.mean(residuals):.2f}')
    ax.axvline(x=np.median(residuals), color='red', linestyle=':', label=f'Median: {np.median(residuals):.2f}')
    
    ax.set_title(f"{model_name} Model Residuals Distribution (Histogram) - {location_name.capitalize()} {pollutant_unit}")
    ax.set_xlabel(f"Residuals [{display_unit}]")
    ax.set_ylabel("Frequency")
    ax.legend(frameon=True, fancybox=False, framealpha=1.0, edgecolor='black')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_residuals_kde(y_test, y_pred, pollutant_unit, location_name, model_name, display_unit, plot_config):
    residuals = y_test - y_pred
    
    graph_title = plot_config["residuals_kde"]["title"] or (f"{model_name} Model Residuals Density (KDE) - {location_name.capitalize()} {pollutant_unit}" if location_name != 'Mean' else f"{model_name} Model Residuals Density (KDE) - Mean {pollutant_unit}")
    xaxis_title = plot_config["residuals_kde"]["xaxis_title"] or f"Residuals [{display_unit}]"
    yaxis_title = plot_config["residuals_kde"]["yaxis_title"]
    
    # Using px.histogram to create a density plot
    fig = px.histogram(
        x=residuals,
        histnorm='density', # Important parameter for density plot
        nbins=50, # Desired number of bins, adjustable
        title=graph_title,
        labels={'x': xaxis_title, 'y': yaxis_title}
    )
    
    # Adjusting the appearance of histogram bars
    fig.update_traces(
        marker_color=plot_config["residuals_kde"]["fill_color"], # Color of histogram bars
        opacity=plot_config["residuals_kde"]["fill_opacity"], # Opacity of histogram bars
        selector=dict(type='histogram') # Select only histogram traces
    )
    
    # Optionally, we can add a KDE line to show the distribution more smoothly.
    # This may require the scipy library.
    if _SCIPY_AVAILABLE:
        try:
            kde = stats.gaussian_kde(residuals)
            x_vals = np.linspace(min(residuals), max(residuals), 500)
            y_vals = kde(x_vals)
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', 
                                     line=dict(color=plot_config["residuals_kde"]["line_color"], width=2), 
                                     name='KDE Curve', showlegend=False))
        except Exception as e:
            st.warning(f"Error plotting KDE curve: {e}. Please check residuals data.")
    else:
        st.info("Scipy library not installed. KDE curve cannot be plotted. Density distribution will be shown as histogram.")

    fig.update_layout(
        title=dict( # Başlık ayarı
            text=graph_title,
            font=dict(
                size=plot_config["general"]["plot_title_font_size"], # Yeni eklenen başlık font boyutu
                family=plot_config["general"]["font_family"],
                color=plot_config["general"]["font_color"]
            )
        ),
        template=plot_config["general"]["template"],
        font=dict(family=plot_config["general"]["font_family"], color=plot_config["general"]["font_color"]),
        xaxis=dict( # Eksen ayarları
            title=dict( # Eksen başlığı font ayarı
                text=xaxis_title,
                font=dict(
                    size=plot_config["general"]["axis_title_font_size"],
                    color=plot_config["general"]["axis_title_font_color"],
                    family=plot_config["general"]["axis_title_font_family"]
                )
            ),
            tickfont=dict( # Eksen tik etiketleri font ayarı
                size=plot_config["general"]["axis_tick_font_size"],
                color=plot_config["general"]["axis_tick_font_color"],
                family=plot_config["general"]["axis_tick_font_family"]
            )
        ),
        yaxis=dict( # Eksen ayarları
            title=dict( # Eksen başlığı font ayarı
                text=yaxis_title,
                font=dict(
                    size=plot_config["general"]["axis_title_font_size"],
                    color=plot_config["general"]["axis_title_font_color"],
                    family=plot_config["general"]["axis_title_font_family"]
                )
            ),
            tickfont=dict( # Eksen tik etiketleri font ayarı
                size=plot_config["general"]["axis_tick_font_size"],
                color=plot_config["general"]["axis_tick_font_color"],
                family=plot_config["general"]["axis_tick_font_family"]
            )
        )
    )
    _render_plotly(fig, use_container_width=True, config=PLOTLY_CONFIG)

def plot_residuals_kde_seaborn(y_test, y_pred, pollutant_unit, location_name, model_name, display_unit, plot_config):
    if not _SEABORN_AVAILABLE:
        st.warning("Seaborn or Matplotlib library not installed. Cannot plot with this function.")
        return
    
    residuals = y_test - y_pred
    sns.set_style("whitegrid", {"axes.facecolor": "#FFFFFF"})
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.kdeplot(residuals, fill=True, color='purple', ax=ax, alpha=0.3)
    ax.axvline(x=0, color='red', linestyle='--', label='Zero Residuals')
    
    ax.set_title(f"{model_name} Model Residuals Density (KDE) - {location_name.capitalize()} {pollutant_unit}")
    ax.set_xlabel(f"Residuals [{display_unit}]")
    ax.set_ylabel("Density")
    ax.legend(frameon=True, fancybox=False, framealpha=1.0, edgecolor='black')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_dataset_distributions(y_train, y_val, y_test, display_unit, plot_config):
    """
    Draws a box plot comparing the value distributions of
    Training, Validation, and Test sets.
    """
    # Create a long-form DataFrame suitable for Plotly Express
    df_train = pd.DataFrame({'Value': y_train, 'Set': 'Training'})
    df_val = pd.DataFrame({'Value': y_val, 'Set': 'Validation'})
    df_test = pd.DataFrame({'Value': y_test, 'Set': 'Test'})
    df_plot = pd.concat([df_train, df_val, df_test])

    fig = px.box(
        df_plot,
        x='Set',
        y='Value',
        color='Set',
        title="Train/Validation/Test Set Value Distributions",
        labels={'Value': f'Target Value [{display_unit}]', 'Set': 'Dataset'}
    )
    
    fig.update_layout(
        title=dict( # Başlık ayarı
            text="Train/Validation/Test Set Value Distributions",
            font=dict(
                size=plot_config["general"]["plot_title_font_size"], # Yeni eklenen başlık font boyutu
                family=plot_config["general"]["font_family"],
                color=plot_config["general"]["font_color"]
            )
        ),
        template=plot_config["general"]["template"],
        font=dict(family=plot_config["general"]["font_family"], color=plot_config["general"]["font_color"]),
        xaxis=dict( # Eksen ayarları
            title=dict( # Eksen başlığı font ayarı
                text="Dataset",
                font=dict(
                    size=plot_config["general"]["axis_title_font_size"],
                    color=plot_config["general"]["axis_title_font_color"],
                    family=plot_config["general"]["axis_title_font_family"]
                )
            ),
            tickfont=dict( # Eksen tik etiketleri font ayarı
                size=plot_config["general"]["axis_tick_font_size"],
                color=plot_config["general"]["axis_tick_font_color"],
                family=plot_config["general"]["axis_tick_font_family"]
            )
        ),
        yaxis=dict( # Eksen ayarları
            title=dict( # Eksen başlığı font ayarı
                text=f'Target Value [{display_unit}]',
                font=dict(
                    size=plot_config["general"]["axis_title_font_size"],
                    color=plot_config["general"]["axis_title_font_color"],
                    family=plot_config["general"]["axis_title_font_family"]
                )
            ),
            tickfont=dict( # Eksen tik etiketleri font ayarı
                size=plot_config["general"]["axis_tick_font_size"],
                color=plot_config["general"]["axis_tick_font_color"],
                family=plot_config["general"]["axis_tick_font_family"]
            )
        )
    )
    _render_plotly(fig, use_container_width=True, config=PLOTLY_CONFIG)

def plot_dataset_distributions_seaborn(y_train, y_val, y_test, display_unit, plot_config):
    if not _SEABORN_AVAILABLE:
        st.warning("Seaborn or Matplotlib library not installed. Cannot plot with this function.")
        return
    
    df_train = pd.DataFrame({'Value': y_train, 'Set': 'Training'})
    df_val = pd.DataFrame({'Value': y_val, 'Set': 'Validation'})
    df_test = pd.DataFrame({'Value': y_test, 'Set': 'Test'})
    df_plot = pd.concat([df_train, df_val, df_test])

    sns.set_style("whitegrid", {"axes.facecolor": "#FFFFFF"})
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(x='Set', y='Value', data=df_plot, ax=ax, palette='Set2')
    
    ax.set_title("Train/Validation/Test Set Value Distributions")
    ax.set_xlabel("Dataset")
    ax.set_ylabel(f"Target Value [{display_unit}]")
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_pair_plots_seaborn(df, title, plot_config):
    """
    Draws a pair plot using Seaborn for better customization,
    including scatter plots and KDE plots on the diagonal.
    """
    if not _SEABORN_AVAILABLE:
        st.warning("Seaborn or Matplotlib library not installed. Cannot plot with this function.")
        return

    # Create the pair plot
    sns.set_style("whitegrid", {"axes.facecolor": "#FFFFFF"})
    fig = sns.pairplot(df, diag_kind="kde", corner=True, plot_kws={'alpha': 0.6})

    # Set the title for the plot (this requires a bit of matplotlib magic)
    fig.fig.suptitle(title, y=1.02)
    
    # Customize the axes and labels if needed
    for ax in fig.axes.flatten():
        if ax:
            ax.xaxis.label.set_size(plot_config["general"]["axis_title_font_size"])
            ax.yaxis.label.set_size(plot_config["general"]["axis_title_font_size"])
            ax.tick_params(labelsize=plot_config["general"]["axis_tick_font_size"])
    
    st.pyplot(fig.figure)
    plt.close(fig.figure) # Close the plot to free memory

def plot_pair_plots(df, title, plot_config):
    # This function is now a fallback, but we will keep it as it's part of the original code.
    fig = px.scatter_matrix(df, dimensions=[col for col in df.columns if col != 'r2_score'], color='r2_score')
    fig.update_traces(diagonal_visible=False)
    fig.update_layout(title=title)
    _render_plotly(fig, use_container_width=True, config=PLOTLY_CONFIG)

def plot_parallel_coordinates(df, title, plot_config):
    # --- MODIFIED: ADDING CUSTOM COLORSCALE ---
    fig = px.parallel_coordinates(
        df, 
        dimensions=[col for col in df.columns if col != 'r2_score'], 
        color='r2_score',
        color_continuous_scale=px.colors.diverging.Spectral, # Using a more dynamic color scale
        color_continuous_midpoint=df['r2_score'].mean() # Center the color scale on the mean R2 score
    )
    fig.update_layout(title=title)
    fig.update_layout(
        title=title,
        title_x=0.25, # Başlığı yatayda ortalar
        title_y=0.99, # Başlığı grafiğin üstüne taşıyarak metin çakışmasını önler
        font_color="black",
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=100, r=100, t=150, b=50)
    )
    _render_plotly(fig, use_container_width=True, config=PLOTLY_CONFIG)

def plot_optimization_progress(cv_results_dict, model_name):
    """
    Plots R² scores vs. iterations for each optimization method, marking the best score with a star.
    """
    st.subheader(f"R² Score Change During Optimization Process for {model_name}")

    for opt_method, cv_df in cv_results_dict.items():
        if cv_df is not None and 'mean_test_score' in cv_df.columns:
            fig = go.Figure()
            
            # Find the best score and its index
            best_score = cv_df['mean_test_score'].max()
            best_score_index = cv_df['mean_test_score'].idxmax()

            # Plot the main line
            fig.add_trace(go.Scatter(
                x=np.arange(len(cv_df)),
                y=cv_df['mean_test_score'],
                mode='lines+markers',
                name=f'{model_name} ({opt_method})',
                line=dict(color='green', width=2),
                marker=dict(color='green', size=6)
            ))
            
            # Add the red star marker for the best score
            fig.add_trace(go.Scatter(
                x=[best_score_index],
                y=[best_score],
                mode='markers',
                marker=dict(symbol='star', size=15, color='red', line=dict(width=1, color='red')),
                name=f'Best R² Score ({best_score:.4f})'
            ))

            fig.update_layout(
                title=f"Optimization Progress for {model_name} ({opt_method}) - R² Score",
                xaxis_title="Combination Number",
                yaxis_title="Validation R² Score",
                template="plotly_white",
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="right",
                    x=1
                )
            )
            _render_plotly(fig, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info(f"Iteration data not available for {opt_method}. It might not be applicable or the optimization did not run successfully.")


EXPLAINABLE_MODEL_NAMES = [
    "Random Forest",
    "Gradient Boosting",
    "Decision Tree",
    "AdaBoost",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "k-Nearest Neighbors (kNN)",
    "Linear Regression",
    "Ridge Regression",
    "Lasso Regression",
    "ElasticNet Regression",
]

def get_native_feature_importance_df(model, feature_names):
    if model is None or not hasattr(model, 'feature_importances_') or feature_names is None:
        return None
    try:
        importances = np.asarray(model.feature_importances_, dtype=float)
        if len(importances) != len(feature_names):
            return None
        df = pd.DataFrame({
            'Feature': list(feature_names),
            'Importance': importances
        }).sort_values('Importance', ascending=False).reset_index(drop=True)
        if df['Importance'].abs().sum() > 0:
            df['Normalized Importance'] = df['Importance'] / df['Importance'].abs().sum()
        else:
            df['Normalized Importance'] = 0.0
        df['Method'] = 'Native Feature Importance'
        return df
    except Exception:
        return None

def get_permutation_importance_df(model, X_processed_df, y_true, scoring='neg_root_mean_squared_error', n_repeats=10, random_state=42):
    if model is None or X_processed_df is None or y_true is None:
        return None
    try:
        result = permutation_importance(
            model, X_processed_df, y_true,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=1
        )
        df = pd.DataFrame({
            'Feature': list(X_processed_df.columns),
            'Importance': result.importances_mean,
            'Importance Std': result.importances_std,
        }).sort_values('Importance', ascending=False).reset_index(drop=True)
        denom = df['Importance'].abs().sum()
        if denom > 0:
            df['Normalized Importance'] = df['Importance'] / denom
        else:
            df['Normalized Importance'] = 0.0
        df['Method'] = 'Permutation Importance'
        return df
    except Exception:
        return None

def plot_importance_bar(importance_df, title, normalized=False):
    if importance_df is None or importance_df.empty:
        return None
    value_col = 'Normalized Importance' if normalized and 'Normalized Importance' in importance_df.columns else 'Importance'
    df_plot = importance_df.copy().sort_values(value_col, ascending=True)
    fig = px.bar(
        df_plot,
        x=value_col,
        y='Feature',
        orientation='h',
        title=title,
        text_auto='.3f'
    )
    fig.update_layout(template='simple_white')
    return fig

def build_cross_model_importance_table(results_by_model):
    rows = []
    for model_name, res in (results_by_model or {}).items():
        if not isinstance(res, dict):
            continue
        feature_names = res.get('feature_names_for_model')
        native_df = get_native_feature_importance_df(res.get('model'), feature_names)
        perm_df = get_permutation_importance_df(
            res.get('model'),
            res.get('X_test_processed_df'),
            res.get('y_test')
        )
        use_df = native_df if native_df is not None else perm_df
        if use_df is None or use_df.empty:
            continue
        row = {'Model': model_name, 'Method': use_df['Method'].iloc[0]}
        for _, r in use_df.iterrows():
            row[r['Feature']] = r.get('Normalized Importance', r.get('Importance', np.nan))
        rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows).fillna(0.0)
    metric_cols = [c for c in df.columns if c not in ['Model', 'Method']]
    ordered_cols = ['Model', 'Method'] + sorted(metric_cols)
    return df[ordered_cols]


def _select_summary_params(cv_results_df, model_name, auto_optimize_params):
    available = [c.replace("param_", "") for c in cv_results_df.columns if c.startswith("param_")]
    preferred = auto_optimize_params.get(model_name, []) if auto_optimize_params else []
    selected = [p for p in preferred if p in available]
    if len(selected) >= 3:
        return selected[:3]
    if len(selected) >= 2:
        extra = [p for p in available if p not in selected]
        return (selected + extra)[:3]
    return available[:3]


def _safe_numeric_series(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == len(series):
        return numeric, None
    cats = {k: i for i, k in enumerate(sorted(series.astype(str).unique()))}
    return series.astype(str).map(cats), cats




def _method_display_label(method):
    mapping = {
        "GridSearchCV": "Grid Search",
        "RandomizedSearchCV": "Random Search",
        "Bayesian Optimization": "Bayesian Optimization",
        "Bayesian Optimization (skopt)": "Bayesian Optimization",
        "grid": "Grid Search",
        "random": "Random Search",
        "bayesian": "Bayesian Optimization",
    }
    return mapping.get(method, str(method))




def _safe_streamlit_key(value):
    """Return a stable Streamlit-safe key fragment."""
    text = str(value) if value is not None else "item"
    text = text.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in text)


def add_download_buttons(fig, filename_prefix="figure", key_prefix=None):
    """Add PNG/PDF download buttons with unique keys for batch rendering."""
    safe_filename_prefix = _safe_streamlit_key(filename_prefix)
    safe_key_prefix = _safe_streamlit_key(key_prefix or filename_prefix)

    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", dpi=1000, bbox_inches="tight")
    buf_png.seek(0)
    st.download_button(
        label=f"Download {filename_prefix} (PNG - 1000 DPI)",
        data=buf_png,
        file_name=f"{safe_filename_prefix}.png",
        mime="image/png",
        key=f"{safe_key_prefix}_png_download",
    )

    buf_pdf = io.BytesIO()
    fig.savefig(buf_pdf, format="pdf", bbox_inches="tight")
    buf_pdf.seek(0)
    st.download_button(
        label=f"Download {filename_prefix} (PDF - Vector)",
        data=buf_pdf,
        file_name=f"{safe_filename_prefix}.pdf",
        mime="application/pdf",
        key=f"{safe_key_prefix}_pdf_download",
    )

def plot_model_summary_2x3(cv_results_df, model_name, opt_method, auto_optimize_params=None):
    """Publication-ready single-optimizer summary figure.

    Renders one clean 1x3 figure instead of the old 2x3 dashboard:
    1) Validation curve for the primary hyperparameter,
    2) Heatmap for the two main hyperparameters,
    3) Hyperparameter sensitivity/importance.

    Only validation/CV scores are used here; test metrics are kept for final evaluation.
    """
    if cv_results_df is None or cv_results_df.empty:
        st.info("No cv_results data available for summary figure.")
        return

    params = _select_summary_params(cv_results_df, model_name, auto_optimize_params)
    if len(params) < 2:
        st.info("At least two hyperparameters are required to build the summary figure.")
        return

    param1, param2 = params[0], params[1]
    df = cv_results_df.copy().reset_index(drop=True)
    df['iteration'] = np.arange(1, len(df) + 1)
    score_col = 'mean_test_score' if 'mean_test_score' in df.columns else None
    train_col = 'mean_train_score' if 'mean_train_score' in df.columns else None

    if score_col is None:
        st.info("Validation/CV score data are unavailable for this summary figure.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    fig.suptitle(f"{model_name} - Hyperparameter Sensitivity Analysis ({_method_display_label(opt_method)})", fontsize=16, y=1.04)

    # 1. Validation curve
    ax = axes[0]
    x1_raw = df[f'param_{param1}']
    x1_num, cats1 = _safe_numeric_series(x1_raw)
    tmp = pd.DataFrame({'x': x1_num, 'Validation R²': pd.to_numeric(df[score_col], errors='coerce')})
    grp = tmp.groupby('x', as_index=False)['Validation R²'].mean().sort_values('x')
    ax.plot(grp['x'], grp['Validation R²'], marker='o', linewidth=2.0, label='Validation R²')
    if train_col:
        tmp_train = pd.DataFrame({'x': x1_num, 'Training R²': pd.to_numeric(df[train_col], errors='coerce')})
        grp_train = tmp_train.groupby('x', as_index=False)['Training R²'].mean().sort_values('x')
        ax.plot(grp_train['x'], grp_train['Training R²'], marker='o', linewidth=1.8, label='Training R²')
        ax.legend(frameon=True, fancybox=False, framealpha=1.0, edgecolor='black', fontsize=9)
    ax.set_title(f"1. Validation Curve: {param1} vs R²")
    ax.set_xlabel(param1)
    ax.set_ylabel('R² Score')
    ax.grid(True, alpha=0.3)
    if cats1:
        ax.set_xticks(list(cats1.values()))
        ax.set_xticklabels(list(cats1.keys()), rotation=30, ha='right')

    # 2. Heatmap
    ax = axes[1]
    try:
        pivot = df.pivot_table(index=f'param_{param2}', columns=f'param_{param1}', values=score_col, aggfunc='max')
        heat = ax.imshow(pivot.values, aspect='auto', cmap='viridis')
        ax.set_title(f"2. Heatmap: {param1} vs {param2} on Validation R²")
        ax.set_xlabel(param1)
        ax.set_ylabel(param2)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns], rotation=30, ha='right')
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(i) for i in pivot.index])
        fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
    except Exception:
        ax.text(0.5, 0.5, 'Not enough data for heatmap', ha='center', va='center')
        ax.set_title('2. Hyperparameter Heatmap')

    # 3. Hyperparameter sensitivity / importance
    ax = axes[2]
    importances = {}
    for p in params:
        col = f'param_{p}'
        if col not in df.columns:
            continue
        grp = df.groupby(col)[score_col].mean()
        imp = grp.max() - grp.min()
        if pd.notna(imp):
            importances[p] = float(imp)
    if importances:
        imp_series = pd.Series(importances).sort_values()
        ax.barh(imp_series.index, imp_series.values, color='#1f77b4')
        ax.set_title('3. Hyperparameter Sensitivity\n(range of mean Validation R²)')
        ax.set_xlabel('Sensitivity (Δ mean Validation R²)')
    else:
        ax.text(0.5, 0.5, 'Not enough data for sensitivity summary', ha='center', va='center')
        ax.set_title('3. Hyperparameter Sensitivity')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    st.pyplot(fig)
    summary_key = f"figure1_optimization_{model_name}_{opt_method}"
    add_download_buttons(fig, "figure1_optimization", key_prefix=summary_key)
    plt.close(fig)

def _select_comparison_params(cv_results_dict, model_name, auto_optimize_params=None):
    available = set()
    for df in (cv_results_dict or {}).values():
        if df is None or getattr(df, 'empty', True):
            continue
        for c in df.columns:
            if str(c).startswith('param_'):
                available.add(str(c).replace('param_', ''))
    preferred = auto_optimize_params.get(model_name, []) if auto_optimize_params else []
    selected = [p for p in preferred if p in available]
    if len(selected) >= 2:
        return selected[:2]
    available = sorted(available)
    return available[:2]


def plot_comparative_optimization_graphic(results_df, cv_results_dict, model_name, auto_optimize_params=None):
    """Publication-ready comparative optimization figures.

    Renders two separate figures instead of the old 2x3 dashboard:
    Figure 2: GS-RS-BO convergence behavior using validation/CV metrics.
    Figure 3: Method comparison + final test-set performance.
    """
    if results_df is None or results_df.empty or not cv_results_dict:
        st.info("No comparative optimization data available.")
        return

    method_label_map = {
        "GridSearchCV": "Grid Search",
        "RandomizedSearchCV": "Random Search",
        "Bayesian Optimization": "Bayesian Optimization",
        "Bayesian Optimization (skopt)": "Bayesian Optimization",
    }
    color_map = {
        "GridSearchCV": "#66C2A5",
        "RandomizedSearchCV": "#FC8D62",
        "Bayesian Optimization": "#8DA0CB",
        "Bayesian Optimization (skopt)": "#8DA0CB",
    }
    label_color_map = {
        "Grid Search": "#66C2A5",
        "Random Search": "#FC8D62",
        "Bayesian Optimization": "#8DA0CB",
    }
    line_style_map = {
        "GridSearchCV": {"linestyle": "--", "marker": "s", "linewidth": 2.8, "zorder": 6},
        "RandomizedSearchCV": {"linestyle": "-", "marker": "o", "linewidth": 2.2, "zorder": 4},
        "Bayesian Optimization": {"linestyle": "-.", "marker": "^", "linewidth": 2.4, "zorder": 5},
        "Bayesian Optimization (skopt)": {"linestyle": "-.", "marker": "^", "linewidth": 2.4, "zorder": 5},
    }
    order_keys = ["GridSearchCV", "RandomizedSearchCV", "Bayesian Optimization", "Bayesian Optimization (skopt)"]

    def _label(method):
        try:
            return _method_display_label(method)
        except Exception:
            return method_label_map.get(method, str(method))

    available_methods = [
        m for m in order_keys
        if m in cv_results_dict and cv_results_dict.get(m) is not None and not cv_results_dict.get(m).empty
    ]

    results_plot = results_df.copy()
    if "Optimization_Method_Label" not in results_plot.columns:
        results_plot["Optimization_Method_Label"] = results_plot["Optimization_Method"].map(method_label_map).fillna(results_plot["Optimization_Method"])

    def _format_iteration_axis(ax):
        try:
            max_iter = max([len(cv_results_dict[m]) for m in available_methods] or [18])
            ax.set_xlim(1, max_iter)
            ax.set_xticks(list(range(1, max_iter + 1)))
        except Exception:
            pass

    def _plot_method_line(ax, method, x, y):
        style = line_style_map.get(method, {"linestyle": "-", "marker": "o", "linewidth": 2.2, "zorder": 3})
        ax.plot(
            x, y,
            label=_label(method),
            color=color_map.get(method, "#333333"),
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=4.2,
            linewidth=style["linewidth"],
            zorder=style["zorder"],
            alpha=0.96,
        )

    # ------------------------------------------------------------------
    # Figure 2: Comparative optimization convergence behavior
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(13.5, 5.2))
    fig2.suptitle(f"{model_name} - GS, RS and BO Convergence Behavior", fontsize=16, y=1.04)

    ax = axes2[0]
    plotted = False
    for method in available_methods:
        df = cv_results_dict.get(method)
        if "mean_test_score" not in df.columns:
            continue
        work = df.reset_index(drop=True).copy()
        work["iteration"] = np.arange(1, len(work) + 1)
        y = pd.to_numeric(work["mean_test_score"], errors="coerce").cummax()
        _plot_method_line(ax, method, work["iteration"], y)
        plotted = True
    ax.set_title("A. Best Validation R² vs Iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best-so-far Validation R²")
    _format_iteration_axis(ax)
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(fontsize=9, frameon=True, fancybox=False, framealpha=1.0, edgecolor="black", loc="best")

    ax = axes2[1]
    plotted = False
    for method in available_methods:
        df = cv_results_dict.get(method)
        if "val_rmse" not in df.columns:
            continue
        work = df.reset_index(drop=True).copy()
        work["iteration"] = np.arange(1, len(work) + 1)
        y = pd.to_numeric(work["val_rmse"], errors="coerce").cummin()
        _plot_method_line(ax, method, work["iteration"], y)
        plotted = True
    ax.set_title("B. Best Validation RMSE vs Iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best-so-far Validation RMSE")
    _format_iteration_axis(ax)
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(fontsize=9, frameon=True, fancybox=False, framealpha=1.0, edgecolor="black", loc="best")
    else:
        ax.text(0.5, 0.5, "Validation RMSE unavailable", ha="center", va="center")

    plt.tight_layout()
    st.pyplot(fig2)
    fig2_key = f"figure2_comparison_{model_name}"
    add_download_buttons(fig2, "figure2_comparison", key_prefix=fig2_key)
    plt.close(fig2)

    # ------------------------------------------------------------------
    # Figure 3: Method comparison and final test-set performance
    # ------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, 3, figsize=(19, 5.4))
    fig3.suptitle(f"{model_name} - Optimization Method Efficiency and Final Test Performance", fontsize=16, y=1.04)

    ax = axes3[0]
    dist_data, dist_labels, dist_colors = [], [], []
    for method in available_methods:
        df = cv_results_dict.get(method)
        if "mean_test_score" not in df.columns:
            continue
        vals = pd.to_numeric(df["mean_test_score"], errors="coerce").dropna().values
        if len(vals) == 0:
            continue
        dist_data.append(vals)
        dist_labels.append(_label(method))
        dist_colors.append(color_map.get(method, "#333333"))
    if dist_data:
        bp = ax.boxplot(dist_data, patch_artist=True, labels=dist_labels, showfliers=False)
        for patch, color in zip(bp["boxes"], dist_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
            patch.set_edgecolor("black")
            patch.set_linewidth(1.0)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.6)
        ax.set_title("A. Validation R² Distribution Across Methods")
        ax.set_ylabel("Validation R²")
        ax.grid(True, alpha=0.25, axis="y")
        ax.tick_params(axis="x", rotation=10)
    else:
        ax.text(0.5, 0.5, "No validation R² data available", ha="center", va="center")
        ax.set_title("A. Validation R² Distribution")

    ax = axes3[1]
    ordered_labels = ["Grid Search", "Random Search", "Bayesian Optimization"]
    runtime_labels, runtime_values, runtime_colors = [], [], []
    for label in ordered_labels:
        sub = results_plot[results_plot["Optimization_Method_Label"] == label]
        if sub.empty or "Duration (s)" not in sub.columns:
            continue
        runtime_labels.append(label)
        runtime_values.append(float(sub["Duration (s)"].iloc[0]))
        runtime_colors.append(label_color_map.get(label, "#999999"))
    x = np.arange(len(runtime_labels))
    bars = ax.bar(x, runtime_values, color=runtime_colors, edgecolor="black", linewidth=0.8, alpha=0.88)
    ax.set_title("B. Analysis Runtime Comparison")
    ax.set_ylabel("Analysis Runtime (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(runtime_labels, rotation=10)
    ax.grid(True, alpha=0.25, axis="y")
    for bar, val in zip(bars, runtime_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.1f}s", ha="center", va="bottom", fontsize=8)
    ax.text(
        0.02, 0.98, "Runtime excludes plotting/download preparation.",
        transform=ax.transAxes, va="top", ha="left", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.85),
    )

    ax = axes3[2]
    best_row = results_plot.sort_values("Test_R2", ascending=False).iloc[0] if "Test_R2" in results_plot.columns and not results_plot.empty else None
    if best_row is None or "Y_Test_Values" not in results_plot.columns or "Y_Pred_Values" not in results_plot.columns:
        ax.text(0.5, 0.5, "Final test prediction data unavailable", ha="center", va="center")
        ax.set_title("C. Predicted vs True Calibration (Test Set)")
    else:
        y_true = np.asarray(best_row["Y_Test_Values"], dtype=float)
        y_pred = np.asarray(best_row["Y_Pred_Values"], dtype=float)
        if y_true.size == 0 or y_pred.size == 0:
            ax.text(0.5, 0.5, "Final test prediction data unavailable", ha="center", va="center")
            ax.set_title("C. Predicted vs True Calibration (Test Set)")
        else:
            label = best_row["Optimization_Method_Label"]
            color = label_color_map.get(label, "#666666")
            max_points = 2500
            if len(y_true) > max_points:
                idx = np.linspace(0, len(y_true) - 1, max_points).astype(int)
                y_true_plot = y_true[idx]
                y_pred_plot = y_pred[idx]
            else:
                y_true_plot = y_true
                y_pred_plot = y_pred
            ax.scatter(y_true_plot, y_pred_plot, s=12, alpha=0.35, color=color, edgecolor="none", label=label)
            min_val = float(np.nanmin([np.nanmin(y_true_plot), np.nanmin(y_pred_plot)]))
            max_val = float(np.nanmax([np.nanmax(y_true_plot), np.nanmax(y_pred_plot)]))
            ax.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--", linewidth=1.4, label="Ideal 1:1")
            ax.set_title("C. Predicted vs True Calibration (Test Set)")
            ax.set_xlabel("Reference / True Value")
            ax.set_ylabel("Predicted Value")
            ax.grid(True, alpha=0.25)
            text = f"Best method: {label}\nFinal Test R²={best_row['Test_R2']:.4f}\nRMSE={best_row['Test_RMSE']:.2f}\nMAE={best_row['Test_MAE']:.2f}"
            ax.text(
                0.03, 0.97, text, transform=ax.transAxes,
                va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="black", alpha=0.88),
            )
            ax.legend(fontsize=8, frameon=True, fancybox=False, framealpha=1.0, edgecolor="black", loc="lower right")

    plt.tight_layout()
    st.pyplot(fig3)
    fig3_key = f"figure3_performance_{model_name}"
    add_download_buttons(fig3, "figure3_performance", key_prefix=fig3_key)
    plt.close(fig3)
