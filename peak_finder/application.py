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

import numpy as np
import plotly.graph_objects as go
from dash import Dash, callback_context, dcc, no_update
from dash.dependencies import Input, Output
from flask import Flask
from geoapps_utils import geophysical_systems
from geoh5py.data import ReferencedData
from geoh5py.objects import ObjectBase
from geoh5py.shared.utils import fetch_active_workspace
from geoh5py.ui_json import InputFile
import matplotlib.pyplot as plt

from geoapps_utils.application.dash_application import BaseDashApplication, ObjectSelection
from peak_finder.constants import app_initializer, default_ui_json
from peak_finder.driver import PeakFinderDriver
from peak_finder.line_anomaly import LineAnomaly
from peak_finder.params import PeakFinderParams
from peak_finder.utils import default_groups_from_property_group
from peak_finder.layout import peak_finder_layout


class PeakFinder(BaseDashApplication):
    """
    Dash app to make a scatter plot.
    """

    _param_class = PeakFinderParams
    _driver_class = PeakFinderDriver

    _lines = None

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
        )(self.disable_linear_threshold)
        self.app.callback(
            Output(component_id="objects", component_property="data"),
            Output(component_id="ui_json_data", component_property="data"),
            Input(component_id="ui_json_data", component_property="data"),
        )(self.set_objects_value)
        self.app.callback(
            Output(component_id="data", component_property="value"),
            Output(component_id="data", component_property="options"),
            Output(component_id="group_name", component_property="options"),
            Input(component_id="objects", component_property="value"),
        )(self.update_data_options)
        self.app.callback(
            Output(component_id="line_field", component_property="value"),
            Output(component_id="line_field", component_property="options"),
            Input(component_id="objects", component_property="value"),
        )(self.update_lines_field_list)
        self.app.callback(
            Output(component_id="line_id", component_property="value"),
            Output(component_id="line_id", component_property="options"),
            Input(component_id="line_field", component_property="value"),
        )(self.update_lines_list)
        self.app.callback(
            Output(component_id="smoothing", component_property="value"),
            Output(component_id="min_amplitude", component_property="value"),
            Output(component_id="min_value", component_property="value"),
            Output(component_id="min_width", component_property="value"),
            Output(component_id="max_migration", component_property="value"),
            Output(component_id="min_channels", component_property="value"),
            Output(component_id="ga_group_name", component_property="value"),
            Output(component_id="structural_markers", component_property="value"),
            Output(component_id="monitoring_directory", component_property="value"),
            Input(component_id="ui_json_data", component_property="data"),
        )(self.update_remainder_from_ui_json)
        self.app.callback(
            Output(component_id="property_groups", component_property="data"),
            Input(component_id="group_name", component_property="value"),
            Input(component_id="color_picker", component_property="value"),
        )(self.update_property_groups)
        self.app.callback(
            Output(component_id="center", component_property="value"),
            Output(component_id="center", component_property="max"),
            Output(component_id="width", component_property="value"),
            Output(component_id="width", component_property="max"),
            Input(component_id="objects", component_property="value"),
            Input(component_id="property_groups", component_property="data"),
            Input(component_id="smoothing", component_property="value"),
            Input(component_id="max_migration", component_property="value"),
            Input(component_id="min_channels", component_property="value"),
            Input(component_id="min_amplitude", component_property="value"),
            Input(component_id="min_value", component_property="value"),
            Input(component_id="min_width", component_property="value"),
            Input(component_id="line_field", component_property="value"),
            Input(component_id="line_id", component_property="value"),
            Input(component_id="system", component_property="value"),
            Input(component_id="center", component_property="value"),
            Input(component_id="width", component_property="value"),
        )(self.line_update)
        self.app.callback(
            Output(component_id="plot", component_property="figure"),
            Input(component_id="objects", component_property="value"),
            Input(component_id="property_groups", component_property="data"),
            Input(component_id="show_residual", component_property="value"),
            Input(component_id="show_markers", component_property="value"),
            Input(component_id="y_scale", component_property="value"),
            Input(component_id="linear_threshold", component_property="value"),
            Input(component_id="min_value", component_property="value"),
            Input(component_id="center", component_property="value"),
            Input(component_id="width", component_property="value"),
            Input(component_id="x_label", component_property="value"),
        )(self.plot_data_selection)
        self.app.callback(
            Output(component_id="export", component_property="n_clicks"),
            Input(component_id="export", component_property="n_clicks"),
            Input(component_id="monitoring_directory", component_property="value"),
        )(self.trigger_click)

    @property
    def lines(self) -> LineAnomaly | None:
        return self._lines

    @lines.setter
    def lines(self, value):
        self._lines = value

    def set_initialized_layout(self, ui_json_data):
        self.app.layout = peak_finder_layout
        # Adding ui_json_data to layout here, so it can be initialized properly
        peak_finder_layout.children.append(dcc.Store(id="ui_json_data", data=ui_json_data))
        # Assemble property groups
        property_groups = self.params.get_property_groups()
        peak_finder_layout.children.append(dcc.Store(id="property_groups", data=property_groups))

    def disable_linear_threshold(self, y_scale):
        if y_scale == "symlog":
            return False
        return True

    def update_property_groups(self, group_name, color_picker):
        property_groups = {}
        return property_groups

    def set_objects_value(self, ui_json_data: dict):
        """
        Initializing objects from the ObjectSelection ui_json_data. Setting ui_json_data to trigger the other functions'
        initialization.

        :param ui_json_data: Dict of input ui.json params.
        """
        return ui_json_data.get("objects", None), ui_json_data

    def update_data_options(self, objects: str):
        """
        Get data dropdown options from a given object.
        """
        data_value = None
        data_options = self.get_data_options(None, objects)

        return data_value, data_options, data_options

    def update_data_list(self, val: str | None):
        """
        Update dropdown data options.

        :param val: object uuid.
        """
        """
        refresh = self.refresh.value
        self.refresh.value = False
        if self._workspace is not None:
            obj: ObjectBase | None = self._workspace.get_entity(self.objects.value)[0]
            if obj is None or getattr(obj, "get_data_list", None) is None:
                self.data.options = [["", None]]
                self.refresh.value = refresh
                return

            options = [["", None]]  # type: ignore
            if (self.add_groups or self.add_groups == "only") and obj.property_groups:
                options = (
                    options
                    + [["-- Groups --", None]]
                    + [[p_g.name, p_g.uid] for p_g in obj.property_groups]
                )

            if self.add_groups != "only":
                options += [["--- Channels ---", None]]

                children = sorted_children_dict(obj)
                excl = ["visual parameter"]
                if children is not None:
                    options += [
                        [k, v] for k, v in children.items() if k.lower() not in excl  # type: ignore
                    ]

                if self.add_xyz:
                    options += [["X", "X"], ["Y", "Y"], ["Z", "Z"]]

            value = self.data.value
            self.data.options = options

            self.update_uid_name_map()

            if self.select_multiple and any(val in options for val in value):
                self.data.value = [val for val in value if val in options]
            elif value in dict(options).values():  # type: ignore
                self.data.value = value
            elif self.find_label:
                self.data.value = find_value(self.data.options, self.find_label)
        else:
            self.data.options = []
            self.data.uid_name_map = {}

        self.refresh.value = refresh
        """
        pass

    def update_lines_field_list(self, object_uid: str | None):
        options = [["", None]]
        obj = self.workspace.get_entity(uuid.UUID(object_uid))
        options += [
            [k, v]  # type: ignore
            for k, v in obj.children.items()
            if isinstance(obj.get_entity(v)[0], ReferencedData)
        ]
        #value = self.data.value
        #self.data.options = options

        #if value in dict(options).values():  # type: ignore
        #    self.data.value = value
        #elif self.find_label:
        #    self.data.value = find_value(self.data.options, self.find_label)
        lines_field_value = None

        return lines_field_value, options

    def update_lines_list(
        self,
        lines_field: str | None,
    ):
        """
        Update dropdown lines options.
        """
        # From application\selection
        # Updates Select line, self.lines
        options = [["", None]]
        if lines_field and getattr(lines_field[0], "values", None) is not None:
            if isinstance(lines_field[0], ReferencedData):
                options += [
                    [v, k] for k, v in lines_field[0].value_map.map.items()
                ]
        value = None
        return value, options

    def get_line_indices(self, line_field, line_id):
        """
        Find the vertices for a given line ID
        """
        line_data = self.workspace.get_entity(line_field)[0]

        if not isinstance(line_data, ReferencedData):
            return None

        indices = np.where(np.asarray(line_data.values) == line_id)[0]

        if len(indices) == 0:
            return None

        return indices

    def get_active_channels(self, obj, channel_groups):
        active_channels = {}
        for group in channel_groups.values():
            for channel in group["properties"]:
                if getattr(obj, "values", None) is not None:
                    active_channels[channel] = {"name": obj.name}
        return active_channels

    def line_update(
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
        system,
        center,
        width,
    ):
        """
        Re-compute derivatives
        """
        obj = self.workspace.get_entity(uuid.UUID(objects))[0]

        if (
            obj is None
            or len(self.workspace.get_entity(uuid.UUID(line_field))) == 0
            or line_id == ""
            or len(property_groups) == 0
        ):
            return

        line_indices = self.get_line_indices(line_field, line_id)

        if line_indices is None:
            return

        obj.line_indices = line_indices
        #property_groups = [
        #    obj.find_or_create_property_group(name=name)
        #    for name in channel_groups
        #]
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

        line_groups = line_anomaly.anomalies
        anomalies = []
        if line_groups is not None:
            for line_group in line_groups:
                anomalies += line_group.groups
            self.lines.anomalies = anomalies
            self.lines.position = line_anomaly.position
        else:
            #self.group_display.disabled = True
            return

        center_max, width_max = no_update, no_update
        #if self.previous_line != line_id:
        end = self.lines.position.locations_resampled[-1]
        mid = self.lines.position.locations_resampled[-1] * 0.5

        if center >= end:
            center = 0
            center_max = end
            #width = 0
            width_max = end
            width = mid
        else:
            center_max = end
            width_max = end

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
        #self.previous_line = self.lines.lines.value

        return center, center_max, width, width_max

    def plot_data_selection(
        self,
        objects,
        property_groups,
        show_residual,
        show_markers,
        y_scale,
        linear_threshold,
        min_value,
        center,
        width,
        x_label,
    ):
        obj = self.workspace.get_entity(uuid.UUID(objects))[0]
        active_channels = self.get_active_channels(obj, property_groups)

        fig = go.Figure()

        figure = plt.figure(figsize=(12, 6))
        axs = plt.subplot()

        if (
            obj is None
            or getattr(obj, "line_indices", None) is None
            or len(obj.line_indices) < 2
            or len(active_channels) == 0
        ):
            return

        lims = np.searchsorted(
            self.lines.position.locations_resampled,
            [
                (center - width / 2.0),
                (center + width / 2.0),
            ],
        )
        sub_ind = np.arange(lims[0], lims[1])
        if len(sub_ind) == 0:
            return

        y_min, y_max = np.inf, -np.inf
        locs = self.lines.position.locations_resampled
        peak_markers_x, peak_markers_y, peak_markers_c = [], [], []
        end_markers_x, end_markers_y = [], []
        start_markers_x, start_markers_y = [], []
        up_markers_x, up_markers_y = [], []
        dwn_markers_x, dwn_markers_y = [], []

        for channel_dict in active_channels.values():
            if "values" not in channel_dict:
                continue

            values = channel_dict["values"][obj.line_indices]
            values, raw = self.lines.position.resample_values(values)

            y_min = np.nanmin([values[sub_ind].min(), y_min])
            y_max = np.nanmax([values[sub_ind].max(), y_max])
            axs.plot(locs, values, color=[0.5, 0.5, 0.5, 1])
            for group in self.lines.anomalies:
                channels = np.array(
                    [a.parent.data_entity.name for a in group.anomalies]
                )
                color = property_groups[group.property_group.name]["color"]
                peaks = group.get_list_attr("peak")
                query = np.where(np.array(channels) == channel_dict["name"])[0]

                if (
                        len(query) == 0
                        or peaks[query[0]] < lims[0]
                        or peaks[query[0]] > lims[1]
                ):
                    continue

                i = query[0]
                start = group.anomalies[i].start
                end = group.anomalies[i].end
                axs.plot(
                    locs[start:end],
                    values[start:end],
                    color=color,
                )

                if group.azimuth < 180:
                    ori = "right"
                else:
                    ori = "left"
                marker = {"left": "<", "right": ">"}
                if show_markers:
                    if i == 0:
                        axs.scatter(
                            locs[peaks[i]],
                            values[peaks[i]],
                            s=200,
                            c="k",
                            marker=marker[ori],
                            zorder=10,
                        )
                    peak_markers_x += [locs[peaks[i]]]
                    peak_markers_y += [values[peaks[i]]]
                    peak_markers_c += [color]
                    start_markers_x += [locs[group.anomalies[i].start]]
                    start_markers_y += [values[group.anomalies[i].start]]
                    end_markers_x += [locs[group.anomalies[i].end]]
                    end_markers_y += [values[group.anomalies[i].end]]
                    up_markers_x += [locs[group.anomalies[i].inflect_up]]
                    up_markers_y += [values[group.anomalies[i].inflect_up]]
                    dwn_markers_x += [locs[group.anomalies[i].inflect_down]]
                    dwn_markers_y += [values[group.anomalies[i].inflect_down]]

            if show_residual:
                axs.fill_between(
                    locs, values, raw, where=raw > values, color=[1, 0, 0, 0.5]
                )
                axs.fill_between(
                    locs, values, raw, where=raw < values, color=[0, 0, 1, 0.5]
                )

        if np.isinf(y_min):
            return

        if y_scale == "symlog":
            plt.yscale("symlog", linthresh=linear_threshold)

        x_lims = [
            center - width / 2.0,
            center + width / 2.0,
        ]
        y_lims = [np.nanmax([y_min, min_value]), y_max]
        axs.set_xlim(x_lims)
        axs.set_ylim(y_lims)
        axs.set_ylabel("Data")
        axs.plot([center, center], [y_min, y_max], "k--")

        if show_markers:
            axs.scatter(
                peak_markers_x,
                peak_markers_y,
                s=50,
                c=peak_markers_c,
                marker="o",
            )
            axs.scatter(
                start_markers_x,
                start_markers_y,
                s=100,
                color="k",
                marker="4",
            )
            axs.scatter(
                end_markers_x,
                end_markers_y,
                s=100,
                color="k",
                marker="3",
            )
            axs.scatter(
                up_markers_x,
                up_markers_y,
                color="k",
                marker="1",
                s=100,
            )
            axs.scatter(
                dwn_markers_x,
                dwn_markers_y,
                color="k",
                marker="2",
                s=100,
            )

        ticks_loc = axs.get_xticks().tolist()
        axs.set_xticks(ticks_loc)

        if x_label == "Easting":
            axs.text(
                center,
                y_lims[0],
                f"{self.lines.position.interp_x(center):.0f} m E",
                va="top",
                ha="center",
                bbox={"edgecolor": "r"},
            )
            axs.set_xticklabels(
                [f"{self.lines.position.interp_x(label):.0f}" for label in ticks_loc]
            )
            axs.set_xlabel("Easting (m)")

        elif x_label == "Northing":
            axs.text(
                center,
                y_lims[0],
                f"{self.lines.position.interp_y(center):.0f} m N",
                va="top",
                ha="center",
                bbox={"edgecolor": "r"},
            )
            axs.set_xticklabels(
                [f"{self.lines.position.interp_y(label):.0f}" for label in ticks_loc]
            )
            axs.set_xlabel("Northing (m)")

        else:
            axs.text(
                center,
                y_min,
                f"{center:.0f} m",
                va="top",
                ha="center",
                bbox={"edgecolor": "r"},
            )
            axs.set_xlabel("Distance (m)")
        axs.grid(True)
        plt.show()
        return fig

    def trigger_click(
        self,
        n_clicks: int,
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
            param_dict = self.params.to_dict()

            # Get output path.
            if (
                monitoring_directory is not None
                and monitoring_directory != ""
                and Path(monitoring_directory).is_dir()
            ):
                param_dict["monitoring_directory"] = str(
                    Path(monitoring_directory).resolve()
                )
                temp_geoh5 = f"PeakFinder_{time():.0f}.geoh5"

                # Get output workspace.
                ws, _ = BaseApplication.get_output_workspace(
                    False, param_dict["monitoring_directory"], temp_geoh5
                )

                with fetch_active_workspace(ws, mode="r") as new_workspace:
                    # Put entities in output workspace.
                    param_dict["geoh5"] = new_workspace
                    for key, value in param_dict.items():
                        if isinstance(value, ObjectBase):
                            param_dict[key] = value.copy(
                                parent=new_workspace, copy_children=True
                            )

                # Write output uijson.
                new_params = PeakFinderParams(**param_dict)
                new_params.write_input_file(
                    name=temp_geoh5.replace(".geoh5", ".ui.json"),
                    path=param_dict["monitoring_directory"],
                    validate=False,
                )

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
