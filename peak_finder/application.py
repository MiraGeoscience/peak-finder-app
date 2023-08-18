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
import time
import uuid
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, callback_context, ctx, dcc, no_update
from dash.dependencies import Input, Output
from flask import Flask
from geoapps_utils.application.application import get_output_workspace
from geoapps_utils.application.dash_application import (
    BaseDashApplication,
    ObjectSelection,
)
from geoh5py.data import ReferencedData
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
        # self.app.callback(
        #
        #    Input(component_id="figure", component_property="figure")
        # )(self.loading_figure)
        self.app.callback(
            Output(component_id="loading", component_property="children"),
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
            Input(component_id="active_channels", component_property="data"),
            Input(component_id="show_residual", component_property="value"),
            Input(component_id="y_scale", component_property="value"),
            Input(component_id="linear_threshold", component_property="value"),
            Input(component_id="x_label", component_property="value"),
        )(PeakFinder.loading_figure)
        self.app.callback(
            Output(component_id="linear_threshold", component_property="disabled"),
            Input(component_id="y_scale", component_property="value"),
        )(PeakFinder.disable_linear_threshold)
        self.app.callback(
            Output(component_id="group_settings", component_property="style"),
            Input(component_id="group_settings_visibility", component_property="value"),
        )(BaseDashApplication.update_visibility_from_checklist)
        self.app.callback(
            Output(component_id="data", component_property="options"),
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
            Output(component_id="group_name", component_property="options"),
            Input(component_id="group_name", component_property="value"),
            Input(component_id="color_picker", component_property="value"),
            Input(component_id="property_groups", component_property="data"),
        )(PeakFinder.update_property_groups)
        self.app.callback(
            Output(component_id="active_channels", component_property="data"),
            Output(component_id="min_value", component_property="value"),
            Output(component_id="linear_threshold", component_property="value"),
            Input(component_id="property_groups", component_property="data"),
            Input(component_id="flip_sign", component_property="value"),
        )(self.update_active_channels)
        self.app.callback(
            Output(component_id="figure", component_property="figure"),
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
            Input(component_id="active_channels", component_property="data"),
            Input(component_id="show_residual", component_property="value"),
            Input(component_id="y_scale", component_property="value"),
            Input(component_id="linear_threshold", component_property="value"),
            Input(component_id="x_label", component_property="value"),
        )(self.update_figure)
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
    def loading_figure(
        objects,
        property_groups,
        smoothing,
        max_migration,
        min_channels,
        min_amplitude,
        min_value,
        min_width,
        line_field,
        line_id,
        active_channels,
        show_residual,
        y_scale,
        linear_threshold,
        x_label,
    ):
        time.sleep(1)
        return no_update

    @staticmethod
    def disable_linear_threshold(y_scale):
        if y_scale == "symlog":
            return False
        return True

    @staticmethod
    def update_property_groups(group_name, color_picker, property_groups):
        property_groups_out, color_picker_out, group_name_options = (
            no_update,
            no_update,
            no_update,
        )
        trigger = ctx.triggered_id

        if trigger == "group_name":
            color_picker_out = {"hex": property_groups[group_name]["color"]}
        elif trigger == "color_picker":
            property_groups_out = property_groups
            property_groups_out[group_name]["color"] = str(color_picker["hex"])
        elif trigger == "property_groups" or trigger is None:
            group_name_options = list(property_groups.keys())

        return property_groups_out, color_picker_out, group_name_options

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
        property_groups_dict = dict(property_groups_dict)
        for group in property_groups_dict.values():
            for channel in group["properties"]:
                chan = self.workspace.get_entity(uuid.UUID(channel))[0]
                if getattr(chan, "values", None) is not None:
                    active_channels[channel] = {"name": chan.name}

        d_min, d_max = np.inf, -np.inf
        thresh_value = np.inf

        for uid, params in active_channels.copy().items():
            chan = self.workspace.get_entity(uuid.UUID(uid))[0]
            try:
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
    ):
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

        line_anomaly = LineAnomaly(
            entity=obj,
            line_indices=line_indices,
            property_groups=property_groups,
            smoothing=smoothing,
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

    def update_figure(
        self,
        objects,
        property_groups,
        smoothing,
        max_migration,
        min_channels,
        min_amplitude,
        min_value,
        min_width,
        line_field,
        line_id,
        active_channels,
        show_residual,
        y_scale,
        linear_threshold,
        x_label,
    ):
        triggers = [t["prop_id"].split(".")[0] for t in callback_context.triggered]
        update_line_triggers = [
            "objects",
            "property_groups",
            "smoothing",
            "max_migration",
            "min_channels",
            "min_amplitude",
            "min_value",
            "min_width",
            "line_field",
            "line_id",
        ]
        if any([t in triggers for t in update_line_triggers]):
            self.line_update(
                objects,
                property_groups,
                smoothing,
                max_migration,
                min_channels,
                min_amplitude,
                min_value,
                min_width,
                line_field,
                line_id,
            )

        figure_data = self.update_figure_data(
            objects,
            property_groups,
            active_channels,
            show_residual,
        )
        figure_layout = PeakFinder.update_figure_layout(
            y_scale,
            linear_threshold,
            min_value,
            x_label,
        )

        return go.Figure(data=figure_data, layout=figure_layout)

    @staticmethod
    def update_figure_layout(
        y_scale,
        linear_threshold,
        min_value,
        x_label,
    ):
        linear_threshold = np.float_power(10, linear_threshold)

        if y_scale == "symlog":
            # plt.yscale("symlog", linthresh=linear_threshold)
            pass

        # y_lims = [np.nanmax([y_min, min_value]), y_max]

        # ticks_loc = axs.get_xticks().tolist()
        # axs.set_xticks(ticks_loc)
        xaxis_title = x_label + " (m)"

        fig_layout = go.Layout(
            xaxis_title=xaxis_title,
            # xaxis_range=x_lims,
            # yaxis_range=y_lims,
            yaxis_title="Data",
        )
        return fig_layout

    def add_markers(
        self,
        trace_dict,
        peak_markers_x,
        peak_markers_y,
        peak_markers_c,
        start_markers_x,
        start_markers_y,
        end_markers_x,
        end_markers_y,
        up_markers_x,
        up_markers_y,
        dwn_markers_x,
        dwn_markers_y,
    ):
        # Add markers
        if "peaks" not in trace_dict:
            trace_dict["peaks"] = {
                "x": [None],
                "y": [None],
                "mode": "markers",
                "marker_color": ["black"],
                "marker_symbol": "circle",
                "name": "peaks",
            }
        trace_dict["peaks"]["x"] += peak_markers_x
        trace_dict["peaks"]["y"] += peak_markers_y
        trace_dict["peaks"]["marker_color"] += peak_markers_c

        if "start markers" not in trace_dict:
            trace_dict["start markers"] = {
                "x": [None],
                "y": [None],
                "mode": "markers",
                "marker_color": "black",
                "marker_symbol": "y-right",
                "name": "start markers",
            }
        trace_dict["start markers"]["x"] += start_markers_x
        trace_dict["start markers"]["y"] += start_markers_y

        if "end markers" not in trace_dict:
            trace_dict["end markers"] = {
                "x": [None],
                "y": [None],
                "mode": "markers",
                "marker_color": "black",
                "marker_symbol": "y-left",
                "name": "end markers",
            }
        trace_dict["end markers"]["x"] += end_markers_x
        trace_dict["end markers"]["y"] += end_markers_y

        if "up markers" not in trace_dict:
            trace_dict["up markers"] = {
                "x": [None],
                "y": [None],
                "mode": "markers",
                "marker_color": "black",
                "marker_symbol": "y-down",
                "name": "up markers",
            }
        trace_dict["up markers"]["x"] += up_markers_x
        trace_dict["up markers"]["y"] += up_markers_y

        if "down markers" not in trace_dict:
            trace_dict["down markers"] = {
                "x": [None],
                "y": [None],
                "mode": "markers",
                "marker_color": "black",
                "marker_symbol": "y-up",
                "name": "down markers",
            }
        trace_dict["down markers"]["x"] += dwn_markers_x
        trace_dict["down markers"]["y"] += dwn_markers_y

        return trace_dict

    def update_figure_data(
        self,
        objects,
        property_groups,
        active_channels,
        show_residual,
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

        y_min, y_max = np.inf, -np.inf
        locs = self.lines_position.locations_resampled
        peak_markers_x, peak_markers_y, peak_markers_c = [], [], []
        end_markers_x, end_markers_y = [], []
        start_markers_x, start_markers_y = [], []
        up_markers_x, up_markers_y = [], []
        dwn_markers_x, dwn_markers_y = [], []

        trace_dict = {
            "lines": {
                "x": [None],
                "y": [None],
                "mode": "lines",
                "name": "full lines",
                "line_color": "lightgrey",
                "showlegend": False,
                "hoverinfo": "skip",
            }
        }
        for channel_dict in list(active_channels.values()):
            if "values" not in channel_dict:
                continue
            values = np.array(channel_dict["values"])[obj.line_indices]
            values, raw = self.lines_position.resample_values(values)

            y_min = np.nanmin([values.min(), y_min])
            y_max = np.nanmax([values.max(), y_max])

            trace_dict["lines"]["x"] += list(locs) + [None]
            trace_dict["lines"]["y"] += list(values) + [None]

            for anomaly_group in self.lines_anomalies:
                channels = np.array(
                    [a.parent.data_entity.name for a in anomaly_group.anomalies]
                )
                group_name = anomaly_group.property_group.name
                color = property_groups[group_name]["color"]
                peaks = anomaly_group.get_list_attr("peak")
                query = np.where(np.array(channels) == channel_dict["name"])[0]
                if len(query) == 0:
                    continue

                i = query[0]
                start = anomaly_group.anomalies[i].start
                end = anomaly_group.anomalies[i].end

                if group_name not in trace_dict:
                    trace_dict[group_name] = {
                        "x": [None],
                        "y": [None],
                        "mode": "lines",
                        "line_color": color,
                        "name": group_name,
                    }
                trace_dict[group_name]["x"] += list(locs[start:end]) + [None]
                trace_dict[group_name]["y"] += list(values[start:end]) + [None]

                if anomaly_group.azimuth < 180:
                    ori = "right"
                else:
                    ori = "left"

                # Add markers
                if i == 0:
                    if ori + " azimuth" not in trace_dict:
                        trace_dict[ori + " azimuth"] = {
                            "x": [None],
                            "y": [None],
                            "mode": "markers",
                            "marker_color": "black",
                            "marker_symbol": "arrow-" + ori,
                            "name": "peaks start",
                        }
                    trace_dict[ori + " azimuth"]["x"] += [locs[peaks[i]]]
                    trace_dict[ori + " azimuth"]["y"] += [values[peaks[i]]]

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

        trace_dict = self.add_markers(
            trace_dict,
            peak_markers_x,
            peak_markers_y,
            peak_markers_c,
            start_markers_x,
            start_markers_y,
            end_markers_x,
            end_markers_y,
            up_markers_x,
            up_markers_y,
            dwn_markers_x,
            dwn_markers_y,
        )

        for trace in list(trace_dict.values()):
            fig_data.append(go.Scatter(**trace))

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
                temp_geoh5 = f"{ga_group_name}_{time.time():.0f}.geoh5"

                # Get output workspace.
                ws, _ = get_output_workspace(
                    False, param_dict["monitoring_directory"], temp_geoh5
                )

                p_g_uid = {
                    p_g.uid: p_g.name for p_g in param_dict["objects"].property_groups
                }
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
