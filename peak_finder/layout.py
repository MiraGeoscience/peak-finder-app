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
    ],
    style={
        "width": "50%",
        "display": "inline-block",
        "vertical-align": "top",
        "margin-right": "5%",
    },
)
group_settings_layout = html.Div(
    [
        html.Div(
            [
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
            ],
        ),
        daq.ColorPicker(
            id="color_picker",
            value=dict(hex="#000000"),
        ),
    ],
    style={"width": "45%", "display": "inline-block", "vertical-align": "top"},
)

plot_layout = html.Div(
    [
        dcc.Graph(id="plot"),
    ]
)

visual_params_layout = html.Div(
    [
        dcc.Markdown(children="**Visual Parameters**", style={"margin-bottom": "20px"}),
        dcc.Markdown(
            children="Window Center",
            style={"width": "30%", "display": "inline-block", "vertical-align": "top"},
        ),
        html.Div(
            [
                dcc.Slider(
                    id="center",
                    min=0,
                    max=5000,
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
            children="Window Width",
            style={"width": "30%", "display": "inline-block", "vertical-align": "top"},
        ),
        html.Div(
            [
                dcc.Slider(
                    id="width",
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
                    min=-18,
                    max=10,
                    step=0.1,
                    marks={
                        -18: "10E-18",
                        -15: "10E-15",
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
        dcc.Checklist(
            id="show_markers",
            options=[{"label": "Show Markers", "value": True}],
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
        dcc.Checklist(
            id="show_residual",
            options=[{"label": "Show Residual", "value": True}],
        ),
    ],
    style={"width": "50%", "display": "inline-block", "vertical-align": "top"},
)


peak_finder_layout = html.Div(
    [
        data_selection_layout,
        group_settings_layout,
        plot_layout,
        visual_params_layout,
        detection_params_layout,
        export_layout,
        dcc.Store(id="objects"),
        dcc.Store(id="system"),
        dcc.Store(id="active_channels"),
    ]
)
