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
        dcc.Markdown(children="**Data Selection**", style={"margin-bottom": "20px"}),
        html.Div(
            [
                dcc.Markdown(
                    children="Lines Field",
                    style={
                        "width": "30%",
                        "display": "inline-block",
                        "vertical-align": "middle",
                    },
                ),
                dcc.Dropdown(
                    id="line_field",
                    style={
                        "width": "70%",
                        "display": "inline-block",
                        "vertical-align": "middle",
                    },
                ),
            ],
            style={"margin-bottom": "20px"},
        ),
        html.Div(
            [
                dcc.Markdown(
                    children="Select Line",
                    style={
                        "width": "30%",
                        "display": "inline-block",
                        "vertical-align": "middle",
                    },
                ),
                dcc.Dropdown(
                    id="line_id",
                    style={
                        "width": "70%",
                        "display": "inline-block",
                        "vertical-align": "middle",
                    },
                ),
            ],
            style={"margin-bottom": "20px"},
        ),
        html.Div(
            [
                dcc.Markdown(
                    children="Masking Data",
                    style={
                        "width": "30%",
                        "display": "inline-block",
                        "vertical-align": "middle",
                    },
                ),
                dcc.Dropdown(
                    id="masking_data",
                    style={
                        "width": "70%",
                        "display": "inline-block",
                        "vertical-align": "middle",
                    },
                ),
            ],
            style={"margin-bottom": "20px"},
        ),
        html.Div(
            [
                dcc.Markdown(
                    children="N outward lines",
                    style={
                        "width": "30%",
                        "display": "inline-block",
                    },
                ),
                dcc.Input(
                    id="n_lines",
                    type="number",
                    min=0,
                    step=1,
                    value=1,
                    style={
                        "width": "49%",
                        "display": "inline-block",
                    },
                ),
            ]
        ),
        dcc.Checklist(
            id="flip_sign",
            options=[{"label": "Flip Y (-1x)", "value": True}],
        ),
        dcc.Checklist(
            id="group_settings_visibility",
            options=[{"label": "Select group colours", "value": True}],
        ),
    ],
    style={
        "width": "45%",
        "display": "inline-block",
        "vertical-align": "top",
        "margin-bottom": "20px",
    },
)
group_settings_layout = html.Div(
    [
        html.Div(
            id="group_settings",
            children=[
                dcc.Markdown(
                    children="Group Name",
                    style={
                        "width": "30%",
                        "display": "inline-block",
                        "vertical-align": "middle",
                        "horizontal-align": "right",
                    },
                ),
                dcc.Dropdown(
                    id="group_name",
                    style={
                        "width": "70%",
                        "display": "inline-block",
                        "vertical-align": "middle",
                        "margin-bottom": "5px",
                    },
                ),
                daq.ColorPicker(  # pylint: disable=not-callable
                    id="color_picker",
                    value={"hex": "#000000"},
                    style={
                        "width": "225px",
                        "margin-left": "22%",
                    },
                ),
            ],
        ),
    ],
    style={
        "width": "45%",
        "vertical-align": "top",
        "display": "inline-block",
    },
)

figure_layout = html.Div(
    [
        dcc.Loading(
            id="line_loading",
            type="default",
            children=html.Div(dcc.Graph(id="line_figure")),
        ),
        dcc.Loading(
            id="full_lines_loading",
            type="default",
            children=html.Div(dcc.Graph(id="full_lines_figure")),
        ),
    ],
)

visual_params_layout = html.Div(
    [
        dcc.Markdown(children="**Visual Parameters**", style={"margin-bottom": "20px"}),
        html.Div(
            [
                dcc.Markdown(
                    children="X-axis Label",
                    style={
                        "width": "30%",
                        "display": "inline-block",
                        "vertical-align": "middle",
                    },
                ),
                dcc.Dropdown(
                    id="x_label",
                    options=["Distance", "Easting", "Northing"],
                    style={
                        "width": "70%",
                        "display": "inline-block",
                        "vertical-align": "middle",
                    },
                    value="Distance",
                ),
            ],
            style={"margin-bottom": "10px"},
        ),
        html.Div(
            [
                dcc.Markdown(
                    children="Y-axis Scaling",
                    style={
                        "width": "30%",
                        "display": "inline-block",
                        "vertical-align": "middle",
                    },
                ),
                dcc.Dropdown(
                    id="y_scale",
                    options=["linear", "symlog"],
                    style={
                        "width": "70%",
                        "display": "inline-block",
                        "vertical-align": "middle",
                    },
                    value="symlog",
                ),
            ],
            style={"margin-bottom": "10px"},
        ),
        dcc.Markdown(
            children="Linear threshold",
            style={"width": "30%", "display": "inline-block", "vertical-align": "top"},
        ),
        html.Div(
            [
                dcc.Slider(
                    id="linear_threshold",
                    min=-10,
                    max=10,
                    step=0.1,
                    marks={
                        -10: "10E-10",
                        -5: "10E-5",
                        0: "1",
                        5: "10E5",
                        10: "10E10",
                    },
                    value=-3.8,
                ),
            ],
            style={
                "width": "70%",
                "display": "inline-block",
                "vertical-align": "top",
                "margin-bottom": "10px",
            },
        ),
        dcc.Checklist(
            id="show_residuals",
            options=[{"label": "Plot residuals", "value": True}],
        ),
    ],
    style={
        "width": "45%",
        "display": "inline-block",
        "vertical-align": "top",
        "margin-right": "5%",
    },
)

detection_params_layout = html.Div(
    [
        dcc.Markdown(
            children="**Detection Parameters**", style={"margin-bottom": "20px"}
        ),
        dcc.Markdown(
            children="Smoothing",
            style={"width": "30%", "display": "inline-block", "vertical-align": "top"},
        ),
        html.Div(
            [
                dcc.Slider(
                    id="smoothing",
                    min=0,
                    max=64,
                    step=1,
                    marks=None,
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
            ],
            style={
                "width": "70%",
                "display": "inline-block",
                "vertical-align": "top",
                "margin-bottom": "10px",
            },
        ),
        dcc.Markdown(
            children="Minimum Amplitude (%)",
            style={"width": "30%", "display": "inline-block", "vertical-align": "top"},
        ),
        html.Div(
            [
                dcc.Slider(
                    id="min_amplitude",
                    min=0.0,
                    max=100.0,
                    step=0.1,
                    marks=None,
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
            ],
            style={
                "width": "70%",
                "display": "inline-block",
                "vertical-align": "top",
                "margin-bottom": "10px",
            },
        ),
        html.Div(
            [
                dcc.Markdown(
                    children="Minimum Data Value",
                    style={
                        "width": "30%",
                        "display": "inline-block",
                        "vertical-align": "middle",
                    },
                ),
                dcc.Input(
                    id="min_value",
                    type="number",
                    style={
                        "width": "70%",
                        "display": "inline-block",
                        "vertical-align": "middle",
                    },
                ),
            ],
            style={"margin-bottom": "10px"},
        ),
        dcc.Markdown(
            children="Minimum Width (m)",
            style={"width": "30%", "display": "inline-block", "vertical-align": "top"},
        ),
        html.Div(
            [
                dcc.Slider(
                    id="min_width",
                    min=1,
                    max=1000,
                    step=1,
                    marks=None,
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
            ],
            style={
                "width": "70%",
                "display": "inline-block",
                "vertical-align": "top",
                "margin-bottom": "10px",
            },
        ),
        dcc.Markdown(
            children="Max Peak Migration",
            style={"width": "30%", "display": "inline-block", "vertical-align": "top"},
        ),
        html.Div(
            [
                dcc.Slider(
                    id="max_migration",
                    min=1,
                    max=1000,
                    step=1,
                    marks=None,
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
            ],
            style={
                "width": "70%",
                "display": "inline-block",
                "vertical-align": "top",
                "margin-bottom": "10px",
            },
        ),
        dcc.Markdown(
            children="Minimum # Channels",
            style={"width": "30%", "display": "inline-block", "vertical-align": "top"},
        ),
        html.Div(
            [
                dcc.Slider(
                    id="min_channels",
                    min=1,
                    max=10,
                    step=1,
                    marks=None,
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
            ],
            style={
                "width": "70%",
                "display": "inline-block",
                "vertical-align": "top",
                "margin-bottom": "10px",
            },
        ),
        html.Div(
            [
                dcc.Markdown(
                    children="Merge N Peaks",
                    style={
                        "width": "30%",
                        "display": "inline-block",
                    },
                ),
                dcc.Input(
                    id="n_groups",
                    value=1,
                    type="number",
                    style={
                        "width": "70%",
                        "display": "inline-block",
                    },
                ),
                dcc.Markdown(
                    children="Max Group Separation",
                    style={
                        "width": "30%",
                        "display": "inline-block",
                    },
                ),
                dcc.Input(
                    id="max_separation",
                    value=100,
                    type="number",
                    style={
                        "width": "70%",
                        "display": "inline-block",
                    },
                ),
            ]
        ),
    ],
    style={
        "width": "45%",
        "display": "inline-block",
        "vertical-align": "top",
        "margin-right": "5%",
    },
)


peak_finder_layout = html.Div(
    [
        dcc.Markdown(
            children="#### Peak Finder",
            style={"margin-bottom": "20px"},
        ),
        figure_layout,
        visual_params_layout,
        data_selection_layout,
        detection_params_layout,
        group_settings_layout,
        html.Div(
            [
                dcc.Checklist(
                    id="structural_markers",
                    options=[{"label": "Export all markers", "value": True}],
                ),
                export_layout,
            ],
            style={"width": "70%", "display": "inline-block", "vertical-align": "top"},
        ),
        dcc.Store(id="objects"),
        dcc.Store(id="active_channels"),
        dcc.Store(id="line_indices"),
        dcc.Store(id="line_ids"),
        dcc.Store(id="trace_map"),
        dcc.Store(id="update_line", data=0),
    ],
)
