POLLUTANT_DISPLAY_LABELS = {
    'CO2': 'CO₂',
    'PM1': 'PM₁',
    'PM25': 'PM₂.₅',
    'PM10': 'PM₁₀',
    'TEMPERATURE': 'Temperature',
    'HUMIDITY': 'Humidity'
}

POLLUTANT_DISPLAY_UNITS = {
    'CO2': 'ppm',
    'PM1': 'µg/m³',
    'PM25': 'µg/m³',
    'PM10': 'µg/m³',
    'TEMPERATURE': '°C',
    'HUMIDITY': '%RH'
}

PLOTLY_CONFIG = {'toImageButtonOptions': {'format': 'png', 'filename': 'newplot', 'height': 800, 'width': 1200, 'scale': 5}}

DEFAULT_PLOT_STYLES = {
    "general": {
        "template": "simple_white",
        "font_family": "Arial",
        "font_color": "#000000",
        "plot_title_font_size": 20,
        "axis_title_font_size": 14,
        "axis_title_font_color": "#333333",
        "axis_title_font_family": "Arial",
        "axis_tick_font_size": 12,
        "axis_tick_font_color": "#000000",
        "axis_tick_font_family": "Arial"
    },
    "time_series": {
        "raw_color": "#228B22",
        "raw_width": 1.0,
        "raw_style": "solid", "raw_opacity": 0.5,
        "calibrated_color": "#FF7F0E",
        "calibrated_width": 2.0,
        "calibrated_style": "solid", "calibrated_opacity": 1.0,
        "reference_color": "#000000",
        "reference_width": 3.0,
        "reference_style": "dash",
        "title": None,
        "xaxis_title": "Time",
        "yaxis_title": None,
        "legend_bgcolor": "#FFFFFF"
    },
    "scatter_plot": {
        "marker_color": "#228B22",
        "marker_size": 5, "marker_opacity": 0.5,
        "trendline_color": "#DC143C",
        "trendline_width": 2.5,
        "trendline_style": "solid",
        "title": None,
        "xaxis_title": None,
        "yaxis_title": None,
    },
    "residuals_plot": {
        "marker_color": "#228B22",
        "marker_size": 5, "marker_opacity": 0.5,
        "zeroline_color": "#646464",
        "zeroline_width": 2.0,
        "zeroline_style": "dash",
        "title": None,
        "xaxis_title": None,
        "yaxis_title": "Residuals"
    },
    "residuals_hist": {
        "bar_color": "#1F77B4",
        "bar_opacity": 0.7,
        "line_color": "#FF0000",
        "line_width": 2.0,
        "title": None,
        "xaxis_title": None,
        "yaxis_title": "Frequency"
    },
    "residuals_kde": {
        "line_color": "#9467BD",
        "fill_color": "#9467BD",
        "fill_opacity": 0.3,
        "title": None,
        "xaxis_title": None,
        "yaxis_title": "Density"
    }
}
