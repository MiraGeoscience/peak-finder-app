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
        html.Div(
            [
                dcc.Markdown(
                    children="Data",
                    style={
                        "width": "30%",
                        "display": "inline-block",
                        "vertical-align": "middle",
                    },
                ),
                dcc.Dropdown(
                    id="data",
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
        "width": "50%",
        "vertical-align": "top",
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
                    },
                ),
                dcc.Dropdown(
                    id="group_name",
                    style={
                        "width": "70%",
                        "display": "inline-block",
                        "vertical-align": "middle",
                    },
                ),
                daq.ColorPicker(  # pylint: disable=not-callable
                    id="color_picker",
                    value={"hex": "#000000"},
                    style={
                        "width": "225px",
                    },
                ),
            ],
        ),
    ],
    style={"width": "50%", "vertical-align": "top"},
)

figure_layout = html.Div(
    [
        dcc.Loading(
            id="loading", type="default", children=html.Div(dcc.Graph(id="figure"))
        ),
    ]
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
            ]
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
            ]
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
                    value=-2,
                ),
            ],
            style={"width": "70%", "display": "inline-block", "vertical-align": "top"},
        ),
    ],
    style={"width": "50%", "display": "inline-block", "vertical-align": "top"},
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
            style={"width": "70%", "display": "inline-block", "vertical-align": "top"},
        ),
        dcc.Markdown(
            children="Minimum Amplitude (%)",
            style={"width": "30%", "display": "inline-block", "vertical-align": "top"},
        ),
        html.Div(
            [
                dcc.Slider(
                    id="min_amplitude",
                    min=0,
                    max=100,
                    step=1,
                    marks=None,
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
            ],
            style={"width": "70%", "display": "inline-block", "vertical-align": "top"},
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
            ]
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
            style={"width": "70%", "display": "inline-block", "vertical-align": "top"},
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
            style={"width": "70%", "display": "inline-block", "vertical-align": "top"},
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
            style={"width": "70%", "display": "inline-block", "vertical-align": "top"},
        ),
    ],
    style={"width": "50%", "display": "inline-block", "vertical-align": "top"},
)


peak_finder_layout = html.Div(
    [
        data_selection_layout,
        group_settings_layout,
        figure_layout,
        visual_params_layout,
        detection_params_layout,
        html.Div(
            [
                export_layout,
            ],
            style={"width": "70%", "display": "inline-block", "vertical-align": "top"},
        ),
        dcc.Store(id="objects"),
        dcc.Store(id="active_channels"),
    ]
)
