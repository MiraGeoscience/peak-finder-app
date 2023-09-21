#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

# pylint: disable=W0613, C0302

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, callback_context, ctx, dcc, no_update
from dash.dependencies import Input, Output, State
from flask import Flask
from geoapps_utils.application.application import get_output_workspace
from geoapps_utils.application.dash_application import (
    BaseDashApplication,
    ObjectSelection,
)
from geoapps_utils.plotting import format_axis, symlog
from geoh5py.data import BooleanData, ReferencedData
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

    def __init__(
        self,
        ui_json: InputFile | None = None,
        ui_json_data: dict | None = None,
        params: PeakFinderParams | None = None,
    ):
        """
        Initialize the peak finder layout, callbacks, and server.

        :param ui_json: ui.json file to load.
        :param ui_json_data: Data from ui.json file.
        :param params: Peak finder params.
        """
        super().__init__(ui_json, ui_json_data, params)

        # Start flask server
        external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
        server = Flask(__name__)
        self.app = Dash(
            server=server,
            url_base_pathname=os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "/"),
            external_stylesheets=external_stylesheets,
        )

        # Getting app layout
        self.set_initialized_layout()

        # Set up callbacks
        line_figure_inputs = [
            Input(component_id="line_figure", component_property="figure"),
            Input(component_id="line_figure", component_property="clickData"),
            Input(component_id="full_lines_figure", component_property="clickData"),
            Input(component_id="objects", component_property="data"),
            Input(component_id="property_groups", component_property="data"),
            Input(component_id="smoothing", component_property="value"),
            Input(component_id="max_migration", component_property="value"),
            Input(component_id="min_channels", component_property="value"),
            Input(component_id="min_amplitude", component_property="value"),
            Input(component_id="min_value", component_property="value"),
            Input(component_id="min_width", component_property="value"),
            Input(component_id="n_groups", component_property="value"),
            Input(component_id="max_separation", component_property="value"),
            Input(component_id="line_field", component_property="value"),
            Input(component_id="masking_data", component_property="value"),
            Input(component_id="line_id", component_property="value"),
            Input(component_id="active_channels", component_property="data"),
            Input(component_id="y_scale", component_property="value"),
            Input(component_id="linear_threshold", component_property="value"),
            Input(component_id="x_label", component_property="value"),
            Input(component_id="show_residuals", component_property="value"),
        ]
        self.app.callback(
            Output(component_id="line_loading", component_property="children"),
            *line_figure_inputs,
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
            Output(component_id="line_field", component_property="options"),
            Output(component_id="masking_data", component_property="options"),
            Input(component_id="objects", component_property="data"),
        )(self.init_data_dropdowns)
        self.app.callback(
            Output(component_id="line_id", component_property="options"),
            Output(component_id="line_id", component_property="value"),
            Input(component_id="line_field", component_property="value"),
            Input(component_id="masking_data", component_property="value"),
            State(component_id="line_id", component_property="value"),
        )(self.update_line_id_options)
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
            Input(component_id="property_groups", component_property="data"),
            Input(component_id="flip_sign", component_property="value"),
        )(self.update_active_channels)
        self.app.callback(
            Output(component_id="line_figure", component_property="figure"),
            Output(component_id="linear_threshold", component_property="min"),
            Output(component_id="linear_threshold", component_property="max"),
            Output(component_id="linear_threshold", component_property="marks"),
            *line_figure_inputs,
        )(self.update_line_figure)
        self.app.callback(
            Output(component_id="full_lines_figure", component_property="figure"),
            Input(component_id="full_lines_figure", component_property="figure"),
            Input(component_id="line_figure", component_property="clickData"),
            Input(component_id="full_lines_figure", component_property="clickData"),
            Input(component_id="n_lines", component_property="value"),
            Input(component_id="line_id", component_property="options"),
            Input(component_id="objects", component_property="data"),
            Input(component_id="property_groups", component_property="data"),
            Input(component_id="smoothing", component_property="value"),
            Input(component_id="max_migration", component_property="value"),
            Input(component_id="min_channels", component_property="value"),
            Input(component_id="min_amplitude", component_property="value"),
            Input(component_id="min_value", component_property="value"),
            Input(component_id="min_width", component_property="value"),
            Input(component_id="n_groups", component_property="value"),
            Input(component_id="max_separation", component_property="value"),
            Input(component_id="line_field", component_property="value"),
            Input(component_id="masking_data", component_property="value"),
            Input(component_id="line_id", component_property="value"),
        )(self.update_full_lines_figure)
        self.app.callback(
            Output(component_id="live_link", component_property="value"),
            Output(component_id="output_message", component_property="children"),
            Input(component_id="export", component_property="n_clicks"),
            State(component_id="objects", component_property="data"),
            State(component_id="flip_sign", component_property="value"),
            State(component_id="line_field", component_property="value"),
            State(component_id="masking_data", component_property="value"),
            State(component_id="smoothing", component_property="value"),
            State(component_id="min_amplitude", component_property="value"),
            State(component_id="min_value", component_property="value"),
            State(component_id="min_width", component_property="value"),
            State(component_id="max_migration", component_property="value"),
            State(component_id="min_channels", component_property="value"),
            State(component_id="n_groups", component_property="value"),
            State(component_id="max_separation", component_property="value"),
            State(component_id="line_id", component_property="value"),
            State(component_id="property_groups", component_property="data"),
            State(component_id="structural_markers", component_property="value"),
            State(component_id="ga_group_name", component_property="value"),
            State(component_id="live_link", component_property="value"),
            State(component_id="monitoring_directory", component_property="value"),
            prevent_initial_call=True,
        )(self.trigger_click)

    @property
    def lines_position(self) -> LinePosition | None:
        """
        Line position for the current plot.
        """
        return self._lines_position

    @lines_position.setter
    def lines_position(self, value):
        self._lines_position = value

    @property
    def lines_anomalies(self) -> list[AnomalyGroup] | None:
        """
        Anomalies for the current plot.
        """
        return self._lines_anomalies

    @lines_anomalies.setter
    def lines_anomalies(self, value):
        self._lines_anomalies = value

    def set_initialized_layout(self):
        """
        Initialize the app layout from ui.json data.
        """
        self.app.layout = peak_finder_layout
        BaseDashApplication.init_vals(self.app.layout.children, self._ui_json_data)

        # Assemble property groups
        property_groups = self.params.get_property_groups()
        for value in property_groups.values():
            value["data"] = str(value["data"])
            value["properties"] = [str(p) for p in value["properties"]]
        self.app.layout.children.append(
            dcc.Store(id="property_groups", data=property_groups)
        )

    @staticmethod
    def loading_figure(*args) -> no_update:
        """
        Callback for loading symbol on figure update.

        :param args: Same inputs as update_figure.
        """
        return no_update

    @staticmethod
    def disable_linear_threshold(y_scale: str) -> bool:
        """
        Disable linear threshold input if y_scale is symlog.

        :param y_scale: Whether y-axis ticks are linear or symlog.

        :return: Whether linear threshold input is disabled.
        """
        if y_scale == "symlog":
            return False
        return True

    @staticmethod
    def update_property_groups(
        group_name: str, color_picker: dict, property_groups: dict
    ) -> tuple[dict, dict, list[str]]:
        """
        Update property groups on color change.
        Update color picker on group name dropdown change.

        :param group_name: Name of property group from dropdown.
        :param color_picker: Color picker hex value.
        :param property_groups: Property groups dictionary.

        :return: Updated property groups, color picker, and group name options.
        """
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

    def init_data_dropdowns(self, objects: str) -> tuple[list[dict], list[dict]]:
        """
        Initialize data and line field dropdowns from input object.

        :param objects: Input object.

        :return: Line field dropdown options.
        :return: Masking data dropdown options.
        """
        line_field_options = []
        masking_data_options = []
        obj = self.workspace.get_entity(uuid.UUID(objects))[0]
        for child in obj.children:
            if isinstance(child, ReferencedData):
                line_field_options.append(
                    {"label": child.name, "value": "{" + str(child.uid) + "}"}
                )
            if isinstance(child, BooleanData):
                masking_data_options.append(
                    {"label": child.name, "value": "{" + str(child.uid) + "}"}
                )
        return line_field_options, masking_data_options

    def update_line_id_options(
        self,
        line_field: str | None,
        masking_data: str | None,
        line_id: int | None,
    ) -> tuple[list[dict], int | None]:
        """
        Update line ID dropdown options from line field.

        :param line_field: Line field.
        :param masking_data: Masking data.

        :return: Line ID dropdown options.
        :return: Line ID value.
        """
        if line_field is None:
            return [], None

        line_field = self.workspace.get_entity(uuid.UUID(line_field))[0]
        value_map = line_field.value_map.map  # type: ignore

        if masking_data is not None:
            masking_data = self.workspace.get_entity(uuid.UUID(masking_data))[0]
            line_vals = np.unique(line_field.values[masking_data.values])  # type: ignore
            value_map = {
                key: value for key, value in value_map.items() if key in line_vals
            }

        options = []
        for key, value in value_map.items():  # type: ignore
            options.append({"label": value, "value": key})

        if line_id not in value_map.keys():
            line_id = None

        return options, line_id

    def get_line_indices(
        self, line_field: str, line_id: int, masking_data: str | None
    ) -> np.ndarray | None:
        """
        Find the vertices for a given line ID.

        :param line_field: Line field.
        :param line_id: Line ID.
        :param masking_data: Masking data.

        :return: Line indices.
        """
        line_data = self.workspace.get_entity(uuid.UUID(line_field))[0]

        if masking_data is not None:
            masking_array = self.workspace.get_entity(uuid.UUID(masking_data))[0].values
            indices = np.where(
                (np.asarray(line_data.values) == line_id) & masking_array
            )[0]
        else:
            indices = np.where(np.asarray(line_data.values) == line_id)[0]

        if len(indices) == 0:
            return None
        return indices

    def update_active_channels(
        self,
        property_groups_dict: dict,
        flip_sign: list[bool],
    ) -> tuple[dict, float]:
        """
        Update active channels from property groups.

        :param property_groups_dict: Property groups dictionary.
        :param flip_sign: Whether to flip the sign of the data.

        :return: Active channels.
        :return: Minimum value.
        """
        if flip_sign:
            flip_sign = -1  # type: ignore
        else:
            flip_sign = 1  # type: ignore

        active_channels = {}
        property_groups_dict = dict(property_groups_dict)
        for group in property_groups_dict.values():
            for channel in group["properties"]:
                chan = self.workspace.get_entity(uuid.UUID(channel))[0]
                if getattr(chan, "values", None) is not None:
                    active_channels[channel] = {"name": chan.name}

        d_min, d_max = np.inf, -np.inf

        keys = list(active_channels.keys())
        for uid in keys:
            chan = self.workspace.get_entity(uuid.UUID(uid))[0]
            try:
                active_channels[uid]["values"] = flip_sign * chan.values.copy()
                d_min = np.nanmin([d_min, active_channels[uid]["values"].min()])
                d_max = np.nanmax([d_max, active_channels[uid]["values"].max()])
            except KeyError:
                continue

        min_value = no_update
        if d_max > -np.inf:
            min_value = d_min
        return active_channels, min_value

    def line_update(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        objects: str,
        property_groups_dict: dict,
        smoothing: float,
        max_migration: float,
        min_channels: int,
        min_amplitude: float,
        min_value: float,
        min_width: float,
        n_groups: int,
        max_separation: float,
        line_field: str,
        masking_data: str | None,
        line_id: int,
    ):
        """
        Update the line position and anomalies.

        :param objects: Input object.
        :param property_groups_dict: Property groups dictionary.
        :param smoothing: Smoothing factor.
        :param max_migration: Maximum migration.
        :param min_channels: Minimum number of channels.
        :param min_amplitude: Minimum amplitude.
        :param min_value: Minimum value.
        :param min_width: Minimum width.
        :param n_groups: Number of consecutive peaks to merge.
        :param max_separation: Maximum separation between peaks to merge.
        :param line_field: Line field.
        :param masking_data: Masking data.
        :param line_id: Line ID.

        :return: Line position.
        :return: Anomalies.
        """
        obj = self.workspace.get_entity(uuid.UUID(objects))[0]
        if (
            obj is None
            or len(self.workspace.get_entity(uuid.UUID(line_field))) == 0
            or line_id is None
            or len(property_groups_dict) == 0
        ):
            return None, None

        line_indices = self.get_line_indices(line_field, line_id, masking_data)

        if line_indices is None:
            return None, None

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
            n_groups=n_groups,
            max_separation=max_separation,
        )

        if line_anomaly is None:
            return None, None

        line_groups = line_anomaly.anomalies
        anomalies: list[AnomalyGroup] = []
        if line_groups is not None:
            for line_group in line_groups:
                anomalies += line_group.groups  # type: ignore

        return line_anomaly.position, anomalies

    def update_line_figure(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        figure: dict | None,
        line_click_data: dict | None,
        full_lines_click_data: dict | None,
        objects: str,
        property_groups: dict,
        smoothing: float,
        max_migration: float,
        min_channels: int,
        min_amplitude: float,
        min_value: float,
        min_width: float,
        n_groups: int,
        max_separation: float,
        line_field: str,
        masking_data: str,
        line_id: int,
        active_channels: dict,
        y_scale: str,
        linear_threshold: float,
        x_label: str,
        show_residuals: list[bool],
    ) -> tuple[go.Figure, float | None, float | None, dict | None]:
        """
        Update the figure.

        :param figure: Figure dictionary.
        :param line_click_data: Click data for single line plot.
        :param full_lines_click_data: Click data for full lines plot.
        :param objects: Input object.
        :param property_groups: Property groups dictionary.
        :param smoothing: Smoothing factor.
        :param max_migration: Maximum migration.
        :param min_channels: Minimum number of channels.
        :param min_amplitude: Minimum amplitude.
        :param min_value: Minimum value.
        :param min_width: Minimum width.
        :param n_groups: Number of consecutive peaks to merge.
        :param max_separation: Maximum separation between peaks to merge.
        :param line_field: Line field.
        :param masking_data: Masking data.
        :param line_id: Line ID.
        :param active_channels: Active channels.
        :param y_scale: Whether y-axis ticks are linear or symlog.
        :param linear_threshold: Linear threshold slider value.
        :param x_label: X-axis label.
        :param show_residuals: Whether to plot residuals.

        :return: Updated figure.
        :return: Linear threshold slider min.
        :return: Linear threshold slider max.
        :return: Linear threshold slider marks.
        """
        triggers = [t["prop_id"].split(".")[0] for t in callback_context.triggered]
        if figure is not None:
            if line_click_data is not None and "line_figure" in triggers:
                figure["layout"]["shapes"][0]["x0"] = line_click_data["points"][0]["x"]
                figure["layout"]["shapes"][0]["x1"] = line_click_data["points"][0]["x"]
                return (
                    go.Figure(data=figure["data"], layout=figure["layout"]),
                    no_update,
                    no_update,
                    no_update,
                )
            if (
                full_lines_click_data is not None
                and "full_lines_figure" in triggers
                and self.lines_position is not None
            ):
                x_val = (
                    full_lines_click_data["points"][0]["x"]
                    - self.lines_position.x_locations[0]  # type: ignore
                )
                figure["layout"]["shapes"][0]["x0"] = x_val
                figure["layout"]["shapes"][0]["x1"] = x_val
                return (
                    go.Figure(data=figure["data"], layout=figure["layout"]),
                    no_update,
                    no_update,
                    no_update,
                )

        update_line_triggers = [
            "objects",
            "property_groups",
            "smoothing",
            "max_migration",
            "min_channels",
            "min_amplitude",
            "min_value",
            "min_width",
            "n_groups",
            "max_separation",
            "line_field",
            "masking_data",
            "line_id",
        ]
        update_line = False
        if any(t in triggers for t in update_line_triggers):
            # Update line position and anomalies
            self.lines_position, self.lines_anomalies = self.line_update(
                objects,
                property_groups,
                smoothing,
                max_migration,
                min_channels,
                min_amplitude,
                min_value,
                min_width,
                n_groups,
                max_separation,
                line_field,
                masking_data,
                line_id,
            )
            if self.lines_position is None and self.lines_anomalies is None:
                return no_update, no_update, no_update, no_update
            update_line = True
        figure_data, figure_layout, y_min, y_max, y_label, y_tickvals, y_ticktext = (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        thresh_min, thresh_max, thresh_ticks = no_update, no_update, no_update
        if figure is not None:
            figure_data = figure["data"]
            figure_layout = figure["layout"]

        # Update figure data
        figure_data_triggers = [
            "objects",
            "active_channels",
            "y_scale",
            "linear_threshold",
            "show_residuals",
        ]
        if (
            figure_data is None
            or update_line
            or any(t in triggers for t in figure_data_triggers)
        ):
            (
                figure_data,
                y_label,
                y_tickvals,
                y_ticktext,
                y_min,
                y_max,
                thresh_min,
                thresh_max,
                thresh_ticks,
            ) = self.update_figure_data(
                objects,
                property_groups,
                active_channels,
                masking_data,
                y_scale,
                linear_threshold,
                show_residuals,
            )
        elif "property_groups" in triggers:
            # Update trace colours if property groups are the only change
            figure_data = PeakFinder.update_data_colours(figure_data, property_groups)

        # Update figure layout
        figure_layout = PeakFinder.update_figure_layout(
            figure_layout,
            y_label,
            y_tickvals,
            y_ticktext,
            y_min,
            y_max,
            min_value,
            x_label,
        )
        fig = go.Figure(data=figure_data, layout=figure_layout)
        if len(figure_layout["shapes"]) == 0:
            fig.add_vline(x=0)
        return (
            fig,
            thresh_min,
            thresh_max,
            thresh_ticks,
        )

    @staticmethod
    def update_data_colours(
        figure_data: list[dict],
        property_groups: dict,
    ):
        """
        Update figure data on colour change.

        :param figure_data: Figure data.
        :param property_groups: Property groups dictionary.

        :return: Updated figure data.
        """
        for trace in figure_data:
            if trace["name"] in property_groups:
                trace["line_color"] = property_groups[trace["name"]]["color"]
        return figure_data

    @staticmethod
    def update_figure_layout(  # pylint: disable=too-many-arguments
        figure_layout: dict | None,
        y_label: str | None,
        y_tickvals: np.ndarray | None,
        y_ticktext: list[str] | None,
        y_min: float | None,
        y_max: float | None,
        min_value: float,
        x_label: str,
    ) -> go.Layout:
        """
        Update the figure layout.

        :param figure_layout: Figure layout dictionary.
        :param y_label: Label for y-axis.
        :param y_tickvals: Y-axis tick values.
        :param y_ticktext: Y-axis tick text.
        :param y_min: Minimum y-axis value.
        :param y_max: Maximum y-axis value.
        :param min_value: Minimum value.
        :param x_label: X-axis label.

        :return: Updated figure layout.
        """
        if figure_layout is None:
            layout_dict = {}
        else:
            layout_dict = figure_layout

        if y_min is not None and y_max is not None:
            layout_dict.update({"yaxis_range": [np.nanmax([y_min, min_value]), y_max]})
        if y_label is not None:
            layout_dict.update(
                {
                    "yaxis_title": y_label,
                }
            )
        if y_tickvals is not None and y_ticktext is not None:
            layout_dict.update(
                {
                    "yaxis_tickvals": y_tickvals,
                    "yaxis_ticktext": [f"{y:.2e}" for y in y_ticktext],
                }
            )

        layout_dict.update({"xaxis_title": x_label + " (m)", "yaxis_tickformat": ".2e"})

        fig_layout = go.Layout(layout_dict)

        return fig_layout

    @staticmethod
    def add_markers(  # pylint: disable=too-many-arguments, too-many-locals
        trace_dict: dict,
        peak_markers_x: list[float],
        peak_markers_y: list[float],
        peak_markers_customdata: list[float],
        peak_markers_c: list[str],
        start_markers_x: list[float],
        start_markers_y: list[float],
        start_markers_customdata: list[float],
        end_markers_x: list[float],
        end_markers_y: list[float],
        end_markers_customdata: list[float],
        up_markers_x: list[float],
        up_markers_y: list[float],
        up_markers_customdata: list[float],
        dwn_markers_x: list[float],
        dwn_markers_y: list[float],
        dwn_markers_customdata: list[float],
    ) -> dict:
        """
        Add markers to the figure.

        :param trace_dict: Dictionary of figure traces.
        :param peak_markers_x: Peak marker x-coordinates.
        :param peak_markers_y: Peak marker y-coordinates.
        :param peak_markers_customdata: Peak marker customdata for y values.
        :param peak_markers_c: Peak marker colors.
        :param start_markers_x: Start marker x-coordinates.
        :param start_markers_y: Start marker y-coordinates.
        :param start_markers_customdata: Start marker customdata for y values.
        :param end_markers_x: End marker x-coordinates.
        :param end_markers_y: End marker y-coordinates.
        :param end_markers_customdata: End marker customdata for y values.
        :param up_markers_x: Up marker x-coordinates.
        :param up_markers_y: Up marker y-coordinates.
        :param up_markers_customdata: Up marker customdata for y values.
        :param dwn_markers_x: Down marker x-coordinates.
        :param dwn_markers_y: Down marker y-coordinates.
        :param dwn_markers_customdata: Down marker customdata for y values.

        :return: Updated trace dictionary.
        """
        # Add markers
        if "peaks" not in trace_dict["markers"]:
            trace_dict["markers"]["peaks"] = {
                "x": [None],
                "y": [None],
                "customdata": [None],
                "mode": "markers",
                "marker_color": ["black"],
                "marker_symbol": "circle",
                "marker_size": 8,
                "name": "peaks",
                "legendgroup": "markers",
                "showlegend": False,
                "visible": "legendonly",
                "hovertemplate": (
                    "<b>x</b>: %{x:,.2f} <br>" + "<b>y</b>: %{customdata:,.2e}"
                ),
            }
        trace_dict["markers"]["peaks"]["x"] += peak_markers_x
        trace_dict["markers"]["peaks"]["y"] += peak_markers_y
        trace_dict["markers"]["peaks"]["customdata"] += peak_markers_customdata
        trace_dict["markers"]["peaks"]["marker_color"] += peak_markers_c

        if "start_markers" not in trace_dict["markers"]:
            trace_dict["markers"]["start_markers"] = {
                "x": [None],
                "y": [None],
                "customdata": [None],
                "mode": "markers",
                "marker_color": "black",
                "marker_symbol": "y-right-open",
                "marker_size": 6,
                "name": "start markers",
                "legendgroup": "markers",
                "showlegend": False,
                "visible": "legendonly",
                "hovertemplate": (
                    "<b>x</b>: %{x:,.2f} <br>" + "<b>y</b>: %{customdata:,.2e}"
                ),
            }
        trace_dict["markers"]["start_markers"]["x"] += start_markers_x
        trace_dict["markers"]["start_markers"]["y"] += start_markers_y
        trace_dict["markers"]["start_markers"]["customdata"] += start_markers_customdata

        if "end_markers" not in trace_dict["markers"]:
            trace_dict["markers"]["end_markers"] = {
                "x": [None],
                "y": [None],
                "customdata": [None],
                "mode": "markers",
                "marker_color": "black",
                "marker_symbol": "y-left-open",
                "marker_size": 6,
                "name": "end markers",
                "legendgroup": "markers",
                "showlegend": False,
                "visible": "legendonly",
                "hovertemplate": (
                    "<b>x</b>: %{x:,.2f} <br>" + "<b>y</b>: %{customdata:,.2e}"
                ),
            }
        trace_dict["markers"]["end_markers"]["x"] += end_markers_x
        trace_dict["markers"]["end_markers"]["y"] += end_markers_y
        trace_dict["markers"]["end_markers"]["customdata"] += end_markers_customdata

        if "up_markers" not in trace_dict["markers"]:
            trace_dict["markers"]["up_markers"] = {
                "x": [None],
                "y": [None],
                "customdata": [None],
                "mode": "markers",
                "marker_color": "black",
                "marker_symbol": "y-down-open",
                "marker_size": 6,
                "name": "up markers",
                "legendgroup": "markers",
                "showlegend": False,
                "visible": "legendonly",
                "hovertemplate": (
                    "<b>x</b>: %{x:,.2f} <br>" + "<b>y</b>: %{customdata:,.2e}"
                ),
            }
        trace_dict["markers"]["up_markers"]["x"] += up_markers_x
        trace_dict["markers"]["up_markers"]["y"] += up_markers_y
        trace_dict["markers"]["up_markers"]["customdata"] += up_markers_customdata

        if "down_markers" not in trace_dict["markers"]:
            trace_dict["markers"]["down_markers"] = {
                "x": [None],
                "y": [None],
                "customdata": [None],
                "mode": "markers",
                "marker_color": "black",
                "marker_symbol": "y-up-open",
                "marker_size": 6,
                "name": "down markers",
                "legendgroup": "markers",
                "showlegend": False,
                "visible": "legendonly",
                "hovertemplate": (
                    "<b>x</b>: %{x:,.2f} <br>" + "<b>y</b>: %{customdata:,.2e}"
                ),
            }
        trace_dict["markers"]["down_markers"]["x"] += dwn_markers_x
        trace_dict["markers"]["down_markers"]["y"] += dwn_markers_y
        trace_dict["markers"]["down_markers"]["customdata"] += dwn_markers_customdata

        return trace_dict

    @staticmethod
    def add_residuals(
        fig_data: list[go.Scatter],
        values: np.ndarray,
        raw: np.ndarray,
        locs: np.ndarray,
    ) -> list[go.Scatter]:
        """
        Add residuals to the figure.

        :param fig_data: Figure data.
        :param values: Resampled values.
        :param raw: Raw values.
        :param locs: Locations.

        :return: Updated figure data.
        """
        pos_inds = np.where(raw > values)[0]
        neg_inds = np.where(raw < values)[0]

        pos_residuals = raw.copy()
        pos_residuals[pos_inds] = values[pos_inds]
        neg_residuals = raw.copy()
        neg_residuals[neg_inds] = values[neg_inds]

        fig_data.append(
            go.Scatter(
                x=locs,
                y=values,
                line={"color": "rgba(0, 0, 0, 0)"},
                showlegend=False,
                hoverinfo="skip",
            ),
        )
        fig_data.append(
            go.Scatter(
                x=locs,
                y=pos_residuals,
                line={"color": "rgba(0, 0, 0, 0)"},
                fill="tonexty",
                fillcolor="rgba(255, 0, 0, 0.5)",
                name="positive residuals",
                legendgroup="positive residuals",
                showlegend=False,
                visible="legendonly",
                hoverinfo="skip",
            )
        )

        fig_data.append(
            go.Scatter(
                x=locs,
                y=values,
                line={"color": "rgba(0, 0, 0, 0)"},
                showlegend=False,
                hoverinfo="skip",
            ),
        )
        fig_data.append(
            go.Scatter(
                x=locs,
                y=neg_residuals,
                line={"color": "rgba(0, 0, 0, 0)"},
                fill="tonexty",
                fillcolor="rgba(0, 0, 255, 0.5)",
                name="negative residuals",
                legendgroup="negative residuals",
                showlegend=False,
                visible="legendonly",
                hoverinfo="skip",
            )
        )
        return fig_data

    def update_figure_data(  # pylint: disable=too-many-locals, too-many-branches, too-many-statements, too-many-arguments
        self,
        objects: str,
        property_groups: dict,
        active_channels: dict,
        masking_data: str,
        y_scale: str,
        linear_threshold: float,
        show_residuals: list[bool],
    ) -> tuple[
        list[go.Scatter],
        str | None,
        np.ndarray | None,
        np.ndarray | None,
        float | None,
        float | None,
        float | None,
        float | None,
        dict | None,
    ]:
        """
        Update the figure data.

        :param objects: Input object.
        :param property_groups: Property groups dictionary.
        :param active_channels: Active channels.
        :param masking_data: Masking data.
        :param y_scale: Whether y-axis ticks are linear or symlog.
        :param linear_threshold: Linear threshold.
        :param show_residuals: Whether to plot residuals.

        :return: Updated figure data.
        :return: Label for y-axis.
        :return: Y-axis tick values.
        :return: Y-axis tick text.
        :return: Minimum y-axis value.
        :return: Maximum y-axis value.
        :return: Linear threshold slider min.
        :return: Linear threshold slider max.
        :return: Linear threshold slider marks.
        """
        obj = self.workspace.get_entity(uuid.UUID(objects))[0]
        fig_data: list[go.Scatter] = []

        if (
            obj is None
            or getattr(obj, "line_indices", None) is None
            or len(obj.line_indices) < 2
            or len(active_channels) == 0
            or self.lines_position is None
        ):
            return fig_data, None, None, None, None, None, None, None, None

        y_min, y_max = np.inf, -np.inf
        locs = self.lines_position.locations_resampled
        peak_markers_x, peak_markers_y, peak_markers_customdata, peak_markers_c = (
            [],
            [],
            [],
            [],
        )
        end_markers_x, end_markers_y, end_markers_customdata = [], [], []
        start_markers_x, start_markers_y, start_markers_customdata = [], [], []
        up_markers_x, up_markers_y, up_markers_customdata = [], [], []
        dwn_markers_x, dwn_markers_y, dwn_markers_customdata = [], [], []

        trace_dict = {
            "lines": {
                "lines": {
                    "x": [None],
                    "y": [None],
                    "mode": "lines",
                    "name": "full lines",
                    "line_color": "lightgrey",
                    "showlegend": False,
                    "hoverinfo": "skip",
                }
            },
            "property_groups": {},
            "markers": {},
        }

        log = y_scale == "symlog"
        threshold = np.float_power(10, linear_threshold)

        all_values = []
        for channel_dict in list(active_channels.values()):
            if "values" not in channel_dict:
                continue
            if masking_data is not None:
                mask = self.workspace.get_entity(uuid.UUID(masking_data))[0].values
                values = np.array(channel_dict["values"])
                values[mask is False] = np.nan
                values = values[obj.line_indices]
                values = values[~np.isnan(values)]
            else:
                values = np.array(channel_dict["values"])[obj.line_indices]

            values, raw = self.lines_position.resample_values(values)
            all_values += list(values.flatten())

            if log:
                sym_values = symlog(values, threshold)
                sym_raw = symlog(raw, threshold)
            else:
                sym_values = values
                sym_raw = raw

            y_min = np.nanmin([sym_values.min(), y_min])
            y_max = np.nanmax([sym_values.max(), y_max])

            trace_dict["lines"]["lines"]["x"] += list(locs) + [None]  # type: ignore
            trace_dict["lines"]["lines"]["y"] += list(sym_values) + [None]  # type: ignore

            for anomaly_group in self.lines_anomalies:  # type: ignore
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
                start = anomaly_group.start
                end = anomaly_group.end

                if group_name not in trace_dict["property_groups"]:  # type: ignore
                    trace_dict["property_groups"][group_name] = {  # type: ignore
                        "x": [None],
                        "y": [None],
                        "customdata": [None],
                        "mode": "lines",
                        "line_color": color,
                        "name": group_name,
                        "hovertemplate": (
                            "<b>x</b>: %{x:,.2f} <br>" + "<b>y</b>: %{customdata:,.2e}"
                        ),
                    }
                trace_dict["property_groups"][group_name]["x"] += list(  # type: ignore
                    locs[start:end]
                ) + [None]
                trace_dict["property_groups"][group_name]["y"] += list(  # type: ignore
                    sym_values[start:end]
                ) + [None]
                trace_dict["property_groups"][group_name]["customdata"] += list(  # type: ignore
                    values[start:end]
                ) + [
                    None
                ]

                if anomaly_group.azimuth < 180:  # type: ignore
                    ori = "right"
                else:
                    ori = "left"

                # Add markers
                if i == 0:
                    if ori + "_azimuth" not in trace_dict["markers"]:  # type: ignore
                        trace_dict["markers"][ori + "_azimuth"] = {  # type: ignore
                            "x": [None],
                            "y": [None],
                            "customdata": [None],
                            "mode": "markers",
                            "marker_color": "black",
                            "marker_symbol": "arrow-" + ori,
                            "marker_size": 8,
                            "name": "peaks start",
                            "legendgroup": "markers",
                            "showlegend": False,
                            "visible": "legendonly",
                            "hovertemplate": (
                                "<b>x</b>: %{x:,.2f} <br>"
                                + "<b>y</b>: %{customdata:,.2e}"
                            ),
                        }
                    trace_dict["markers"][ori + "_azimuth"]["x"] += [  # type: ignore
                        locs[peaks[i]]
                    ]
                    trace_dict["markers"][ori + "_azimuth"]["y"] += [  # type: ignore
                        sym_values[peaks[i]]
                    ]
                    trace_dict["markers"][ori + "_azimuth"]["customdata"] += [  # type: ignore
                        values[peaks[i]]
                    ]

                peak_markers_x += [locs[peaks[i]]]
                peak_markers_y += [sym_values[peaks[i]]]
                peak_markers_customdata += [values[peaks[i]]]
                peak_markers_c += [color]
                start_markers_x += [locs[anomaly_group.anomalies[i].start]]
                start_markers_y += [sym_values[anomaly_group.anomalies[i].start]]
                start_markers_customdata += [values[anomaly_group.anomalies[i].start]]
                end_markers_x += [locs[anomaly_group.anomalies[i].end]]
                end_markers_y += [sym_values[anomaly_group.anomalies[i].end]]
                end_markers_customdata += [values[anomaly_group.anomalies[i].end]]
                up_markers_x += [locs[anomaly_group.anomalies[i].inflect_up]]
                up_markers_y += [sym_values[anomaly_group.anomalies[i].inflect_up]]
                up_markers_customdata += [values[anomaly_group.anomalies[i].inflect_up]]
                dwn_markers_x += [locs[anomaly_group.anomalies[i].inflect_down]]
                dwn_markers_y += [sym_values[anomaly_group.anomalies[i].inflect_down]]
                dwn_markers_customdata += [
                    values[anomaly_group.anomalies[i].inflect_down]
                ]

            if show_residuals:
                fig_data = PeakFinder.add_residuals(
                    fig_data,
                    sym_values,
                    sym_raw,
                    locs,
                )

        if np.isinf(y_min):
            return fig_data, None, None, None, None, None, None, None, None

        all_values = np.array(all_values)
        _, y_label, y_tickvals, y_ticktext = format_axis(
            channel="Data",
            axis=all_values,
            log=log,
            threshold=threshold,
        )

        trace_dict = PeakFinder.add_markers(
            trace_dict,
            peak_markers_x,
            peak_markers_y,
            peak_markers_customdata,
            peak_markers_c,
            start_markers_x,
            start_markers_y,
            start_markers_customdata,
            end_markers_x,
            end_markers_y,
            end_markers_customdata,
            up_markers_x,
            up_markers_y,
            up_markers_customdata,
            dwn_markers_x,
            dwn_markers_y,
            dwn_markers_customdata,
        )

        for trace_name in ["lines", "property_groups", "markers"]:
            for trace in list(trace_dict[trace_name].values()):  # type: ignore
                fig_data.append(go.Scatter(**trace))

        fig_data.append(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker_color="black",
                marker_symbol="circle",
                legendgroup="markers",
                name="markers",
                visible="legendonly",
            ),
        )
        if show_residuals:
            fig_data.append(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line_color="rgba(255, 0, 0, 0.5)",
                    line_width=8,
                    legendgroup="positive residuals",
                    name="positive residuals",
                    visible="legendonly",
                ),
            )
            fig_data.append(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line_color="rgba(0, 0, 255, 0.5)",
                    line_width=8,
                    legendgroup="negative residuals",
                    name="negative residuals",
                    visible="legendonly",
                ),
            )

        # Update linear threshold
        pos_vals = all_values[all_values > 0]  # type: ignore

        thresh_min = np.log10(np.min(pos_vals))
        thresh_max = np.log10(np.max(pos_vals))
        thresh_ticks = {
            t: "10E" + f"{t:.2g}" for t in np.linspace(thresh_min, thresh_max, 5)
        }

        return (
            fig_data,
            y_label,
            y_tickvals,
            y_ticktext,
            y_min,
            y_max,
            thresh_min,
            thresh_max,
            thresh_ticks,
        )

    def update_full_lines_figure(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        figure: dict | None,
        line_click_data: dict | None,
        full_lines_click_data: dict | None,
        n_lines: int,
        line_id_options: list[dict[str, str]],
        objects: str,
        property_groups: dict,
        smoothing: float,
        max_migration: float,
        min_channels: int,
        min_amplitude: int,
        min_value: float,
        min_width: float,
        n_groups: int,
        max_separation: float,
        line_field: str,
        masking_data: str,
        line_id: int,
    ):
        """
        Update the full lines figure.

        :param figure: Figure dictionary.
        :param line_click_data: Line figure click data.
        :param full_lines_click_data: Full lines figure click data.
        :param n_lines: Number of lines to display.
        :param line_id_options: Line id options.
        :param objects: Input object.
        :param property_groups: Property groups dictionary.
        :param smoothing: Smoothing.
        :param max_migration: Maximum migration.
        :param min_channels: Minimum number of channels.
        :param min_amplitude: Minimum amplitude.
        :param min_value: Minimum value.
        :param min_width: Minimum width.
        :param n_groups: Number of consecutive peaks to merge.
        :param max_separation: Maximum separation between peaks to merge.
        :param line_field: Line field.
        :param masking_data: Masking data.
        :param line_id: Line id.
        """
        triggers = [t["prop_id"].split(".")[0] for t in callback_context.triggered]
        if figure is not None:
            if (
                line_click_data is not None
                and "line_figure" in triggers
                and self.lines_position is not None
            ):
                x_locs = self.lines_position.x_locations
                x_val = x_locs[0] + line_click_data["points"][0]["x"]  # type: ignore
                ind = (np.abs(x_locs - x_val)).argmin()
                y_val = self.lines_position.y_locations[ind]  # type: ignore
                figure["data"][-1]["x"] = [x_val]
                figure["data"][-1]["y"] = [y_val]
                return figure
            if full_lines_click_data is not None and "full_lines_figure" in triggers:
                x_val = full_lines_click_data["points"][0]["x"]
                y_val = full_lines_click_data["points"][0]["y"]
                figure["data"][-1]["x"] = [x_val]
                figure["data"][-1]["y"] = [y_val]
                return figure

        figure = go.Figure()
        if n_lines is None or line_id is None:
            return figure

        min_ind = max(0, line_id - n_lines)
        max_ind = min(len(line_id_options), line_id + n_lines + 1)

        anomaly_traces = {}
        for key, value in property_groups.items():
            anomaly_traces[key] = {
                "x": [None],
                "y": [None],
                "marker_color": value["color"],
                "mode": "markers",
                "name": key,
            }

        marker_x = None
        marker_y = None
        for line in line_id_options[min_ind:max_ind]:
            position, anomalies = self.line_update(
                objects,
                property_groups,
                smoothing,
                max_migration,
                min_channels,
                min_amplitude,
                min_value,
                min_width,
                n_groups,
                max_separation,
                line_field,
                masking_data,
                int(line["value"]),
            )
            if position is not None:
                if line["value"] == line_id:
                    marker_x = position.x_locations[0]
                    marker_y = position.y_locations[0]
                    figure.add_trace(  # type: ignore
                        go.Scatter(
                            x=position.x_locations,
                            y=position.y_locations,
                            name=line["label"],
                            line_color="black",
                        )
                    )
                else:
                    figure.add_trace(  # type: ignore
                        go.Scatter(
                            x=position.x_locations,
                            y=position.y_locations,
                            name=line["label"],
                        )
                    )

            if anomalies is not None:
                for anom in anomalies:
                    anomaly_traces[anom.property_group.name]["x"].append(
                        anom.group_center[0]
                    )
                    anomaly_traces[anom.property_group.name]["y"].append(
                        anom.group_center[1]
                    )

        for trace in list(anomaly_traces.values()):
            figure.add_trace(  # type: ignore
                go.Scatter(
                    **trace,
                )
            )

        figure.add_trace(  # type: ignore
            go.Scatter(
                x=[marker_x],
                y=[marker_y],
                marker_color="black",
                marker_symbol="star",
                marker_size=10,
                mode="markers",
                showlegend=False,
            )
        )

        figure.update_layout(  # type: ignore
            xaxis_title="Easting (m)",
            yaxis_title="Northing (m)",
            yaxis_scaleanchor="x",
            yaxis_scaleratio=1,
        )

        return figure

    def trigger_click(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        n_clicks: int,
        objects: str,
        flip_sign: list[bool],
        line_field: str,
        masking_data: str | None,
        smoothing: float,
        min_amplitude: float,
        min_value: float,
        min_width: float,
        max_migration: float,
        min_channels: int,
        n_groups: int,
        max_separation: float,
        line_id: int,
        property_groups: dict,
        structural_markers: list[bool],
        ga_group_name: str,
        live_link: list[bool],
        monitoring_directory: str,
    ) -> tuple[list[bool], list[str]]:
        """
        Write output ui.json file and workspace, run driver.

        :param n_clicks: Trigger for callback.
        :param objects: Input object.
        :param flip_sign: Whether to flip the sign of the data.
        :param line_field: Line field.
        :param masking_data: Masking data.
        :param smoothing: Smoothing factor.
        :param min_amplitude: Minimum amplitude.
        :param min_value: Minimum value.
        :param min_width: Minimum width.
        :param max_migration: Maximum migration.
        :param min_channels: Minimum number of channels.
        :param n_groups: Number of consecutive peaks to merge.
        :param max_separation: Maximum separation between peaks to merge.
        :param line_id: Line ID.
        :param property_groups: Property groups dictionary.
        :param structural_markers: Whether to save structural markers.
        :param ga_group_name: Group name.
        :param live_link: Whether to use live link.
        :param monitoring_directory: Monitoring directory.

        :return: Live link status.
        :return: Output save message.
        """
        if (
            (monitoring_directory is None)
            or (monitoring_directory == "")
            or not Path(monitoring_directory).is_dir()
        ):
            return no_update, ["Invalid output path."]

        if not live_link:
            live_link = False  # type: ignore
        else:
            live_link = True  # type: ignore

        # Update self.params from dash component values
        param_dict = self.get_params_dict(locals())

        # Get output path.
        param_dict["monitoring_directory"] = str(Path(monitoring_directory).resolve())
        temp_geoh5 = f"{ga_group_name}_{time.time():.0f}.geoh5"

        # Get output workspace.
        workspace, live_link = get_output_workspace(
            live_link, monitoring_directory, temp_geoh5
        )

        if not live_link:
            param_dict["monitoring_directory"] = ""

        with workspace as new_workspace:
            # Put entities in output workspace.
            param_dict["geoh5"] = new_workspace
            param_dict["objects"] = param_dict["objects"].copy(
                parent=new_workspace, copy_children=True
            )
            p_g_new = {p_g.name: p_g for p_g in param_dict["objects"].property_groups}
            # Add property groups
            for key, value in property_groups.items():
                param_dict[f"group_{value['param']}_data"] = p_g_new[key]
                param_dict[f"group_{value['param']}_color"] = value["color"]

            # Write output uijson.
            new_params = PeakFinderParams(**param_dict)
            new_params.write_input_file(
                name=str(new_workspace.h5file).replace(".geoh5", ".ui.json"),
                path=param_dict["monitoring_directory"],
                validate=False,
            )
            driver = PeakFinderDriver(new_params)
            driver.run()

        if live_link:
            return [True], [
                "Live link active. Check your ANALYST session for new mesh."
            ]
        return [], ["Saved to " + monitoring_directory]


if __name__ == "__main__":
    print("Loading geoh5 file . . .")
    FILE = sys.argv[1]
    ifile = InputFile.read_ui_json(FILE)
    if ifile.data["launch_dash"]:
        ifile.workspace.open("r")
        print("Loaded. Launching peak finder app . . .")
        ObjectSelection.run("Peak Finder", PeakFinder, ifile)
    else:
        print("Loaded. Running peak finder driver . . .")
        PeakFinderDriver.start(FILE)
    print("Done")
