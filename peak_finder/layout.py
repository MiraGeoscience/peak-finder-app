#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import dash_daq as daq
from dash import dcc, html
from geoapps_utils.application.layout import export_layout

data_selection_layout = html.Div(
    [
        dcc.Markdown("Data"),
        dcc.Dropdown(
            id="data",
        ),
    ]
)

group_settings_layout = html.Div(
    [
        dcc.Markdown("Group Name"),
        dcc.Dropdown(id="group_name"),
        dcc.Markdown("Color"),
        daq.ColorPicker(
            id="color_picker",
            value=dict(hex="#000000"),
        ),
        #dcc.Store(id="property_groups"),
    ]
)

line_selection_layout = html.Div(
    [
        dcc.Markdown("Lines Field"),
        dcc.Dropdown(id="line_field"),
        dcc.Markdown("Select Line"),
        dcc.Dropdown(id="line_id"),
    ]
)

plot_layout = html.Div(
    [
        dcc.Graph(id="plot"),
    ]
)

visual_params_layout = html.Div(
    [
        dcc.Markdown("Visual Parameters"),
        #dcc.Markdown("Select Peak"),
        #dcc.Dropdown(id="peak"),
        dcc.Markdown("Window Center"),
        dcc.Slider(
            id="center",
            min=0,
            max=5000,
            step=1,
        ),
        dcc.Markdown("Window Width"),
        dcc.Slider(id="width"),
        dcc.Markdown("X-axis Label"),
        dcc.Dropdown(
            id="x_label",
            options=["Distance", "Easting", "Northing"],
        ),
        dcc.Markdown("Y-axis Scaling"),
        dcc.Dropdown(
            id="y_axis_scaling",
            options=["linear", "symlog"],
        ),
        dcc.Markdown("Linear threshold"),
        dcc.Slider(id="linear_threshold"),
        dcc.Checklist(
            id="show_markers",
            options=[{"label": "Show Markers", "value": True}],
        ),
    ]
)
"""
linear_threshold
min=-18,
max=10,
step=0.1,
base=10,
"""
detection_params_layout = html.Div(
    [
        dcc.Markdown("Detection Parameters"),
        dcc.Markdown("Smoothing"),
        dcc.Slider(id="smoothing", min=0, max=64, step=1),
        dcc.Markdown("Minimum Amplitude (%)"),
        dcc.Slider(id="min_amplitude", min=0, max=100, step=1),
        dcc.Markdown("Minimum Data Value"),
        dcc.Input(id="min_value", type="number"),
        dcc.Markdown("Minimum Width (m)"),
        dcc.Slider(id="min_width", min=1, max=1000, step=1),
        dcc.Markdown("Mac Peak Migration"),
        dcc.Slider(id="max_migration", min=1, max=1000, step=1),
        dcc.Markdown("Minimum # Channels"),
        dcc.Slider(id="min_channels", min=1, max=10, step=1),
        dcc.Checklist(
            id="show_residual",
            options=[{"label": "Show Residual", "value": True}],
        )
    ]
)


peak_finder_layout = html.Div(
    [
        data_selection_layout,
        group_settings_layout,
        line_selection_layout,
        plot_layout,
        visual_params_layout,
        detection_params_layout,
        export_layout,
    ]
)
