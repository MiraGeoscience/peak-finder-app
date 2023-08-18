#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

# pylint: disable=W0613

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from dash import Dash, callback_context, ctx, dcc, no_update
from dash.dependencies import Input, Output, State
from flask import Flask
from geoapps_utils import geophysical_systems
from geoapps_utils.application.application import get_output_workspace
from geoapps_utils.application.dash_application import (
    BaseDashApplication,
    ObjectSelection,
)
from geoh5py.data import ReferencedData
from geoh5py.groups import PropertyGroup
from geoh5py.objects import ObjectBase
from geoh5py.shared.utils import fetch_active_workspace
from geoh5py.ui_json import InputFile

from peak_finder.anomaly_group import AnomalyGroup
from peak_finder.driver import PeakFinderDriver
from peak_finder.layout import peak_finder_layout
from peak_finder.line_anomaly import LineAnomaly
from peak_finder.line_position import LinePosition
from peak_finder.params import PeakFinderParams


class PeakFinder(BaseDashApplication):
    """
    Dash app to make a scatter plot.
    """

    _param_class = PeakFinderParams
    _driver_class = PeakFinderDriver

    _lines_position = None
    _lines_anomalies = None

    def __init__(self, ui_json=None, ui_json_data=None, params=None):
        if params is not None:
            # Launched from notebook
            # Params for initialization are coming from params
            # ui_json_data is provided
            self.params = params
        elif ui_json is not None and Path(ui_json.path).exists():
            # Launched from terminal
            # Params for initialization are coming from ui_json
            # ui_json_data starts as None
            self.params = self._param_class(ui_json)
            ui_json_data = self.params.input_file.demote(self.params.to_dict())

        super().__init__()

        # Start flask server
        external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
        server = Flask(__name__)
        self.app = Dash(
            server=server,
            url_base_pathname=os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "/"),
            external_stylesheets=external_stylesheets,
        )

        # Getting app layout
        self.set_initialized_layout(ui_json_data)

        # Set up callbacks
        self.app.callback(
            Output(component_id="linear_threshold", component_property="disabled"),
            Input(component_id="y_scale", component_property="value"),
        )(PeakFinder.disable_linear_threshold)
        self.app.callback(
            Output(component_id="data", component_property="options"),
            Output(component_id="group_name", component_property="options"),
            Input(component_id="objects", component_property="data"),
        )(self.update_data_options)
        self.app.callback(
            Output(component_id="line_field", component_property="options"),
            Input(component_id="objects", component_property="data"),
        )(self.update_lines_field_list)
        self.app.callback(
            Output(component_id="line_id", component_property="options"),
            Input(component_id="line_field", component_property="value"),
        )(self.update_lines_list)
        self.app.callback(
            Output(component_id="property_groups", component_property="data"),
            Output(component_id="color_picker", component_property="value"),
            Input(component_id="group_name", component_property="value"),
            Input(component_id="color_picker", component_property="value"),
            State(component_id="property_groups", component_property="data"),
        )(PeakFinder.update_property_groups)
        self.app.callback(
            Output(component_id="active_channels", component_property="data"),
            Output(component_id="min_value", component_property="value"),
            Output(component_id="linear_threshold", component_property="value"),
            Input(component_id="property_groups", component_property="data"),
            Input(component_id="flip_sign", component_property="value"),
        )(self.update_active_channels)
        self.app.callback(
            Output(component_id="center", component_property="value"),
            Output(component_id="center", component_property="max"),
            Output(component_id="width", component_property="value"),
            Output(component_id="width", component_property="max"),
            Input(component_id="plot", component_property="relayoutData"),
            State(component_id="plot", component_property="figure"),
            Input(component_id="objects", component_property="data"),
            Input(component_id="property_groups", component_property="data"),
            Input(component_id="smoothing", component_property="value"),
            Input(component_id="max_migration", component_property="value"),
            Input(component_id="min_channels", component_property="value"),
            Input(component_id="min_amplitude", component_property="value"),
            Input(component_id="min_value", component_property="value"),
            Input(component_id="min_width", component_property="value"),
            Input(component_id="line_field", component_property="value"),
            Input(component_id="line_id", component_property="value"),
            Input(component_id="system", component_property="data"),
            Input(component_id="center", component_property="value"),
            Input(component_id="width", component_property="max"),
        )(self.update_window_params)
        self.app.callback(
            Output(component_id="figure_layout", component_property="data"),
            Input(component_id="center", component_property="value"),
            Input(component_id="width", component_property="value"),
            Input(component_id="y_scale", component_property="value"),
            Input(component_id="linear_threshold", component_property="value"),
            Input(component_id="min_value", component_property="value"),
            Input(component_id="x_label", component_property="value"),
        )(self.update_figure_layout)
        self.app.callback(
            Output(component_id="figure_data", component_property="data"),
            Input(component_id="objects", component_property="data"),
            Input(component_id="property_groups", component_property="data"),
            Input(component_id="active_channels", component_property="data"),
            Input(component_id="show_residual", component_property="value"),
            Input(component_id="show_markers", component_property="value"),
            Input(component_id="center", component_property="value"),
            Input(component_id="width", component_property="value"),
        )(self.update_figure_data)
        self.app.callback(
            Output(component_id="plot", component_property="figure"),
            Input(component_id="figure_data", component_property="data"),
            Input(component_id="figure_layout", component_property="data"),
        )(PeakFinder.update_plot)
        self.app.callback(
            Output(component_id="export", component_property="n_clicks"),
            Input(component_id="export", component_property="n_clicks"),
            Input(component_id="monitoring_directory", component_property="value"),
            prevent_initial_call=True,
        )(self.trigger_click)

    @property
    def lines_position(self) -> LinePosition | None:
        return self._lines_position

    @lines_position.setter
    def lines_position(self, value):
        self._lines_position = value

    @property
    def lines_anomalies(self) -> list[AnomalyGroup] | None:
        return self._lines_anomalies

    @lines_anomalies.setter
    def lines_anomalies(self, value):
        self._lines_anomalies = value

    def set_initialized_layout(self, ui_json_data):
        self.app.layout = peak_finder_layout
        BaseDashApplication.init_vals(self.app.layout.children, ui_json_data)

        # Assemble property groups
        property_groups = self.params.get_property_groups()
        peak_finder_layout.children.append(
            dcc.Store(id="property_groups", data=property_groups)
        )

    @staticmethod
    def disable_linear_threshold(y_scale):
        if y_scale == "symlog":
            return False
        return True

    @staticmethod
    def update_property_groups(group_name, color_picker, property_groups):
        property_groups_out, color_picker_out = no_update, no_update
        trigger = ctx.triggered_id

        if trigger == "group_name":
            color_picker_out = dict(hex=property_groups[group_name]["color"])
        elif trigger == "color_picker":
            property_groups[group_name]["color"] = color_picker
        return property_groups_out, color_picker_out

    def update_data_options(self, objects: str):
        data_options = []
        group_name_options = []
        for child in self.workspace.get_entity(uuid.UUID(objects))[0].property_groups:
            data_options.append(
                {"label": child.name, "value": "{" + str(child.uid) + "}"}
            )
            group_name_options.append(child.name)
        return data_options, group_name_options

    def update_lines_field_list(self, object_uid: str | None):
        obj = self.workspace.get_entity(uuid.UUID(object_uid))[0]
        options = []
        for child in obj.children:
            if isinstance(child, ReferencedData):
                options.append(
                    {"label": child.name, "value": "{" + str(child.uid) + "}"}
                )
        return options

    def update_lines_list(
        self,
        line_field: str | None,
    ):
        line_field = self.workspace.get_entity(uuid.UUID(line_field))[0]
        options = []
        for key, value in line_field.value_map.map.items():
            options.append({"label": value, "value": key})
        return options

    def get_line_indices(self, line_field, line_id):
        """
        Find the vertices for a given line ID
        """
        line_data = self.workspace.get_entity(uuid.UUID(line_field))[0]
        indices = np.where(np.asarray(line_data.values) == line_id)[0]

        if len(indices) == 0:
            return None

        return indices

    def update_active_channels(self, property_groups_dict, flip_sign):
        if flip_sign:
            flip_sign = -1
        else:
            flip_sign = 1

        active_channels = {}
        for group in property_groups_dict.values():
            for channel in group["properties"]:
                chan = self.workspace.get_entity(uuid.UUID(channel))[0]
                if getattr(chan, "values", None) is not None:
                    active_channels[channel] = {"name": chan.name}

        d_min, d_max = np.inf, -np.inf
        thresh_value = np.inf
        # if self.tem_checkbox.value:
        #    system = self.em_system_specs[self.system.value]

        for uid, params in active_channels.copy().items():
            chan = self.workspace.get_entity(uuid.UUID(uid))[0]
            try:
                # if self.tem_checkbox.value:
                #    channel = [
                #        ch for ch in system["channels"] if ch in params["name"]
                #    ]
                #    if any(channel):
                #        self.active_channels[uid]["time"] = system["channels"][
                #            channel[0]
                #        ]
                #    else:
                #        del self.active_channels[uid]

                active_channels[uid]["values"] = flip_sign * chan.values.copy()
                thresh_value = np.min(
                    [
                        thresh_value,
                        np.percentile(np.abs(active_channels[uid]["values"]), 95),
                    ]
                )
                d_min = np.nanmin([d_min, active_channels[uid]["values"].min()])
                d_max = np.nanmax([d_max, active_channels[uid]["values"].max()])
            except KeyError:
                continue

        min_value, linear_threshold = no_update, no_update
        if d_max > -np.inf:
            min_value = d_min
            linear_threshold = thresh_value

        return active_channels, min_value, linear_threshold

    @staticmethod
    def update_plot(
        figure_data,
        figure_layout,
    ):
        return go.Figure(data=figure_data, layout=figure_layout)

    def update_figure_layout(
        self,
        center,
        width,
        y_scale,
        linear_threshold,
        min_value,
        x_label,
    ):
        linear_threshold = np.float_power(10, linear_threshold)

        if y_scale == "symlog":
            plt.yscale("symlog", linthresh=linear_threshold)

        x_lims = [
            center - width / 2.0,
            center + width / 2.0,
        ]
        # y_lims = [np.nanmax([y_min, min_value]), y_max]

        # ticks_loc = axs.get_xticks().tolist()
        # axs.set_xticks(ticks_loc)

        if x_label == "Easting":
            """axs.text(
                center,
                y_lims[0],
                f"{self.lines_position.interp_x(center):.0f} m E",
                va="top",
                ha="center",
                bbox={"edgecolor": "r"},
            )
            axs.set_xticklabels(
                [f"{self.lines_position.interp_x(label):.0f}" for label in ticks_loc]
            )"""
            # axs.set_xlabel("Easting (m)")
            xaxis_title = "Easting (m)"

        elif x_label == "Northing":
            """axs.text(
                center,
                y_lims[0],
                f"{self.lines_position.interp_y(center):.0f} m N",
                va="top",
                ha="center",
                bbox={"edgecolor": "r"},
            )
            axs.set_xticklabels(
                [f"{self.lines_position.interp_y(label):.0f}" for label in ticks_loc]
            )"""
            # axs.set_xlabel("Northing (m)")
            xaxis_title = "Northing (m)"
        else:
            """axs.text(
                center,
                y_min,
                f"{center:.0f} m",
                va="top",
                ha="center",
                bbox={"edgecolor": "r"},
            )"""
            # fig.add_annotation()
            # axs.set_xlabel("Distance (m)")
            xaxis_title = "Distance (m)"

        fig_layout = go.Layout(
            xaxis_title=xaxis_title,
            xaxis_range=x_lims,
            # yaxis_range=y_lims,
            yaxis_title="Data",
        )
        return fig_layout

    def update_window_params(
        self,
        figure_zoom_trigger,
        figure,
        objects,
        property_groups_dict,
        smoothing,
        max_migration,
        min_channels,
        min_amplitude,
        min_value,
        min_width,
        line_field,
        line_id,
        system,
        center,
        width_max,
    ):
        center_out, center_max_out, width_out, width_max_out = (
            no_update,
            no_update,
            no_update,
            no_update,
        )
        triggers = [t["prop_id"].split(".")[0] for t in callback_context.triggered]

        if "plot" in triggers and figure is not None:
            if (
                "autorange" in figure["layout"]
                and figure["layout"]["xaxis"]["autorange"]
            ):
                width_out = width_max
                center_out = width_max / 2
            else:
                width_out = (
                    figure["layout"]["xaxis"]["range"][1]
                    - figure["layout"]["xaxis"]["range"][0]
                )
                center_out = figure["layout"]["xaxis"]["range"][0] + width_out / 2.0
        else:
            # Update self.lines_position and self.lines_anomalies and use to update center, width
            self.line_update(
                objects,
                property_groups_dict,
                smoothing,
                max_migration,
                min_channels,
                min_amplitude,
                min_value,
                min_width,
                line_field,
                line_id,
                system,
            )
            end = self.lines_position.locations_resampled[-1]
            mid = self.lines_position.locations_resampled[-1] * 0.5

            if center >= end:
                center_out = 0
                center_max_out = end
                width_max_out = end
                width_out = mid
            else:
                center_max_out = end
                width_max_out = end

        return center_out, center_max_out, width_out, width_max_out

    def line_update(
        self,
        objects,
        property_groups_dict,
        smoothing,
        max_migration,
        min_channels,
        min_amplitude,
        min_value,
        min_width,
        line_field,
        line_id,
        system,
    ):
        """
        Re-compute derivatives
        """
        obj = self.workspace.get_entity(uuid.UUID(objects))[0]
        if (
            obj is None
            or len(self.workspace.get_entity(uuid.UUID(line_field))) == 0
            or line_id == ""
            or len(property_groups_dict) == 0
        ):
            return

        line_indices = self.get_line_indices(line_field, line_id)
        if line_indices is None:
            return

        obj.line_indices = line_indices
        property_groups = [
            obj.find_or_create_property_group(name=name)
            for name in property_groups_dict
        ]

        em_system_specs = geophysical_systems.parameters()
        line_anomaly = LineAnomaly(
            entity=obj,
            line_indices=line_indices,
            property_groups=property_groups,
            smoothing=smoothing,
            data_normalization=em_system_specs[system]["normalization"],
            min_amplitude=min_amplitude,
            min_value=min_value,
            min_width=min_width,
            max_migration=max_migration,
            min_channels=min_channels,
        )

        if line_anomaly is None:
            return
        self.lines_position = line_anomaly.position

        line_groups = line_anomaly.anomalies
        anomalies = []
        if line_groups is not None:
            for line_group in line_groups:
                anomalies += line_group.groups
        self.lines_anomalies = anomalies

        """
        line_groups = line_anomaly.anomalies
        anomalies = []
        if line_groups is not None:
            for line_group in line_groups:
                anomalies += line_group.groups
            self.lines.anomalies = anomalies
            self.lines.position = line_anomaly.position
        else:
            # self.group_display.disabled = True
            return center_out, center_max_out, width_out, width_max_out
        """
        # if self.previous_line != line_id:

        """
        if self.lines.anomalies is not None and len(self.lines.anomalies) > 0:
            peaks = np.sort(
                self.lines.position.locations_resampled[
                    [group.anomalies[0].peak for group in self.lines.anomalies]
                ]
            )
            current = self.center.value
            self.group_display.options = np.round(peaks, decimals=1)
            self.group_display.value = self.group_display.options[
                np.argmin(np.abs(peaks - current))
            ]
        """
        # self.previous_line = self.lines.lines.value

    def update_figure_data(
        self,
        objects,
        property_groups,
        active_channels,
        show_residual,
        show_markers,
        center,
        width,
    ):
        obj = self.workspace.get_entity(uuid.UUID(objects))[0]

        fig_data = []

        if (
            obj is None
            or getattr(obj, "line_indices", None) is None
            or len(obj.line_indices) < 2
            or len(active_channels) == 0
        ):
            return fig_data

        lims = np.searchsorted(
            self.lines_position.locations_resampled,
            [
                (center - width / 2.0),
                (center + width / 2.0),
            ],
        )
        sub_ind = np.arange(lims[0], lims[1])
        if len(sub_ind) == 0:
            return fig_data

        y_min, y_max = np.inf, -np.inf
        locs = self.lines_position.locations_resampled
        peak_markers_x, peak_markers_y, peak_markers_c = [], [], []
        end_markers_x, end_markers_y = [], []
        start_markers_x, start_markers_y = [], []
        up_markers_x, up_markers_y = [], []
        dwn_markers_x, dwn_markers_y = [], []

        fig_data.append(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name="full lines",
                line_color="lightgrey",
                showlegend=False,
            )
        )
        trace_map = {"lines": 0}
        for channel_dict in list(active_channels.values()):
            if "values" not in channel_dict:
                continue

            values = np.array(channel_dict["values"])[obj.line_indices]
            values, raw = self.lines_position.resample_values(values)

            y_min = np.nanmin([values[sub_ind].min(), y_min])

            y_max = np.nanmax([values[sub_ind].max(), y_max])
            lines_trace = fig_data[trace_map["lines"]]
            lines_trace.x += tuple(locs) + tuple([None])
            lines_trace.y += tuple(values) + tuple([None])

            for anomaly_group in self.lines_anomalies:
                channels = np.array(
                    [a.parent.data_entity.name for a in anomaly_group.anomalies]
                )
                group_name = anomaly_group.property_group.name
                color = property_groups[group_name]["color"]
                peaks = anomaly_group.get_list_attr("peak")
                query = np.where(np.array(channels) == channel_dict["name"])[0]

                if (
                    len(query) == 0
                    or peaks[query[0]] < lims[0]
                    or peaks[query[0]] > lims[1]
                ):
                    continue

                i = query[0]
                start = anomaly_group.anomalies[i].start
                end = anomaly_group.anomalies[i].end
                if group_name not in trace_map:
                    trace_map[group_name] = len(fig_data)
                    fig_data.append(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="lines",
                            line_color=color,
                            name=group_name,
                        )
                    )
                # else:
                lines_trace = fig_data[trace_map[group_name]]
                lines_trace.x += tuple(locs[start:end]) + tuple([None])
                lines_trace.y += tuple(values[start:end]) + tuple([None])

                if anomaly_group.azimuth < 180:
                    ori = "right"
                else:
                    ori = "left"

                if show_markers:
                    if i == 0:
                        fig_data.append(
                            go.Scatter(
                                x=[locs[peaks[i]]],
                                y=[values[peaks[i]]],
                                mode="markers",
                                # color=color,
                                marker={
                                    # "size": 200,
                                    "color": "black",  # [0, 1, 2, 3],
                                    "symbol": "arrow-" + ori,
                                },
                                showlegend=False,
                            )
                        )
                    peak_markers_x += [locs[peaks[i]]]
                    peak_markers_y += [values[peaks[i]]]
                    peak_markers_c += [color]
                    start_markers_x += [locs[anomaly_group.anomalies[i].start]]
                    start_markers_y += [values[anomaly_group.anomalies[i].start]]
                    end_markers_x += [locs[anomaly_group.anomalies[i].end]]
                    end_markers_y += [values[anomaly_group.anomalies[i].end]]
                    up_markers_x += [locs[anomaly_group.anomalies[i].inflect_up]]
                    up_markers_y += [values[anomaly_group.anomalies[i].inflect_up]]
                    dwn_markers_x += [locs[anomaly_group.anomalies[i].inflect_down]]
                    dwn_markers_y += [values[anomaly_group.anomalies[i].inflect_down]]

            if show_residual:
                # axs.fill_between(
                #    locs, values, raw, where=raw > values, color=[1, 0, 0, 0.5]
                # )
                # axs.fill_between(
                #    locs, values, raw, where=raw < values, color=[0, 0, 1, 0.5]
                # )
                pass

        if np.isinf(y_min):
            return fig_data

        if show_markers:
            fig_data.append(
                go.Scatter(
                    x=peak_markers_x,
                    y=peak_markers_y,
                    mode="markers",
                    marker={
                        "color": peak_markers_c,
                        "symbol": "circle",
                    },
                    showlegend=False,
                )
            )
            fig_data.append(
                go.Scatter(
                    x=start_markers_x,
                    y=start_markers_y,
                    mode="markers",
                    marker={
                        "color": "black",
                        "symbol": "y-right",
                    },
                    showlegend=False,
                )
            )
            fig_data.append(
                go.Scatter(
                    x=end_markers_x,
                    y=end_markers_y,
                    mode="markers",
                    marker={
                        "color": "black",
                        "symbol": "y-left",
                    },
                    showlegend=False,
                )
            )
            fig_data.append(
                go.Scatter(
                    x=up_markers_x,
                    y=up_markers_y,
                    mode="markers",
                    marker={
                        "color": "black",
                        "symbol": "y-down",
                    },
                    showlegend=False,
                )
            )
            fig_data.append(
                go.Scatter(
                    x=dwn_markers_x,
                    y=dwn_markers_y,
                    mode="markers",
                    marker={
                        "color": "black",
                        "symbol": "y-up",
                    },
                )
            )

        return fig_data

    def trigger_click(
        self,
        n_clicks: int,
        objects,
        data,
        flip_sign,
        line_field,
        system,
        smoothing,
        min_amplitude,
        min_value,
        min_width,
        max_migration,
        min_channels,
        line_id,
        center,
        width,
        property_groups,
        ga_group_name,
        monitoring_directory: str,
    ):
        """
        Save the plot as html, write out ui.json.

        :param n_clicks: Trigger export from button.
        :param monitoring_directory: Output path.
        :param figure: Figure created by update_plots.
        """

        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        if trigger == "export":
            # Update self.params from dash component values
            param_dict = self.get_params_dict(locals())

            # Get output path.
            if (
                monitoring_directory is not None
                and monitoring_directory != ""
                and Path(monitoring_directory).is_dir()
            ):
                param_dict["monitoring_directory"] = str(
                    Path(monitoring_directory).resolve()
                )
                temp_geoh5 = f"{ga_group_name}_{time():.0f}.geoh5"

                # Get output workspace.
                ws, _ = get_output_workspace(
                    False, param_dict["monitoring_directory"], temp_geoh5
                )

                p_g_uid = {
                    p_g.uid: p_g.name for p_g in param_dict["objects"].property_groups
                }
                print(p_g_uid)
                with fetch_active_workspace(ws, mode="r+") as new_workspace:
                    # Put entities in output workspace.
                    param_dict["geoh5"] = new_workspace
                    for key, value in param_dict.items():
                        if isinstance(value, ObjectBase):
                            if new_workspace.get_entity(value.uid)[0] is None:
                                param_dict[key] = value.copy(
                                    parent=new_workspace, copy_children=True
                                )
                                line_field = [
                                    c
                                    for c in param_dict[key].children
                                    if c.name == "Line"
                                ]
                                if line_field:
                                    param_dict["line_field"] = line_field[0]
                        elif isinstance(value, uuid.UUID) and value in p_g_uid:
                            print(value)
                            param_dict[key] = param_dict[
                                "objects"
                            ].find_or_create_property_group(name=p_g_uid[value])
                            print(param_dict[key])

                    # Write output uijson.
                    new_params = PeakFinderParams(**param_dict)
                    new_params.write_input_file(
                        name=temp_geoh5.replace(".geoh5", ".ui.json"),
                        path=param_dict["monitoring_directory"],
                        validate=False,
                    )
                    print(new_params.geoh5)
                    print(new_params.geoh5.h5file)
                    driver = PeakFinderDriver(new_params)
                    # with new_params.geoh5.open(mode="r+"):
                    driver.run()

                print("Saved to " + param_dict["monitoring_directory"])
            else:
                print("Invalid output path.")

        return no_update


if __name__ == "__main__":
    print("Loading geoh5 file . . .")
    file = sys.argv[1]
    ifile = InputFile.read_ui_json(file)
    ifile.workspace.open("r")
    print("Loaded. Launching peak finder app . . .")
    ObjectSelection.run("Peak Finder", PeakFinder, ifile)
    print("Done")
