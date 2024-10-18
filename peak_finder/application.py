#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app.
#
#  All rights reserved.

# pylint: disable=W0613, C0302, duplicate-code

from __future__ import annotations

import os
import sys
import uuid

import numpy as np
import plotly.graph_objects as go
from dash import Dash, callback_context, ctx, dcc, no_update
from dash.dependencies import Input, Output, State
from dash.exceptions import MissingCallbackContextException
from dask import compute
from dask.diagnostics import ProgressBar
from flask import Flask
from geoapps_utils.utils.plotting import format_axis, symlog
from geoh5py import Workspace
from geoh5py.data import BooleanData, Data, ReferencedData
from geoh5py.objects import Curve
from geoh5py.shared.utils import fetch_active_workspace
from geoh5py.ui_json import InputFile
from tqdm import tqdm

from peak_finder.anomaly_group import AnomalyGroup
from peak_finder.dash_application import BaseDashApplication, ObjectSelection
from peak_finder.driver import PeakFinderDriver
from peak_finder.layout import peak_finder_layout
from peak_finder.line_position import LinePosition
from peak_finder.params import PeakFinderParams
from peak_finder.utils import get_ordered_survey_lines


class PeakFinder(BaseDashApplication):  # pylint: disable=too-many-public-methods, too-many-instance-attributes
    """
    Dash app to fine tune Peak Finder parameters.
    """

    _param_class = PeakFinderParams
    _driver_class = PeakFinderDriver

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
        self._active_channels: dict | None = None
        self._figure = None
        self._line_field: ReferencedData | None = None
        self._line_indices = None
        self._computed_lines = None
        self._survey: Curve | None = None
        self._property_groups = None
        self._ordered_survey_lines: dict | None = None

        super().__init__(ui_json, ui_json_data, params)

        self._app = None

        # Start flask server
        self.external_stylesheets = None
        self.server = Flask(__name__)

        # Getting app layout
        with fetch_active_workspace(self.params.geoh5):
            self.set_initialized_layout()

        # Callbacks for layout
        # Update visibility of plots based on dropdown selection.
        self.app.callback(
            Output(component_id="line_figure", component_property="style"),
            Output(component_id="survey_figure", component_property="style"),
            Input(component_id="figure_selection", component_property="value"),
        )(PeakFinder.update_plot_visibility)
        # Update visibility of widgets based on dropdown selection.
        self.app.callback(
            Output(component_id="visual_params", component_property="style"),
            Output(component_id="detection_params", component_property="style"),
            Output(component_id="color_picker_visibility", component_property="value"),
            Input(component_id="widget_selection", component_property="value"),
        )(PeakFinder.update_widget_visibility)
        # Disable linear threshold input if y-axis is symlog.
        self.app.callback(
            Output(component_id="linear_threshold", component_property="disabled"),
            Input(component_id="y_scale", component_property="value"),
        )(PeakFinder.disable_linear_threshold)
        # Update colour picker visibility from checkbox.
        self.app.callback(
            Output(component_id="color_picker_div", component_property="style"),
            Input(component_id="color_picker_visibility", component_property="value"),
        )(BaseDashApplication.update_visibility_from_checklist)

        # Callbacks for data selection
        # Update masking data dropdown from a change in survey object.
        self.app.callback(
            Output(component_id="masking_data", component_property="options"),
            Input(component_id="survey_trigger", component_property="data"),
        )(self.update_masking_dropdowns)

        # Apply masking to survey object.
        self.app.callback(
            Output(component_id="survey_trigger", component_property="data"),
            Output(component_id="min_value", component_property="value"),
            Input(component_id="survey_trigger", component_property="data"),
            Input(component_id="masking_data", component_property="value"),
            Input(component_id="flip_sign", component_property="value"),
        )(self.update_survey_mask)
        # Update line indices for plotting.
        self.app.callback(
            Output(component_id="line_indices_trigger", component_property="data"),
            Input(component_id="line_indices_trigger", component_property="data"),
            Input(component_id="survey_trigger", component_property="data"),
            Input(component_id="selected_line", component_property="value"),
            Input(component_id="n_lines", component_property="value"),
        )(self.update_line_indices)
        # Compute active lines for plotting.
        self.app.callback(
            Output(component_id="lines_computation_trigger", component_property="data"),
            State(component_id="lines_computation_trigger", component_property="data"),
            Input(component_id="line_indices_trigger", component_property="data"),
            Input(component_id="survey_trigger", component_property="data"),
            Input(component_id="selected_line", component_property="value"),
            Input(component_id="n_lines", component_property="value"),
            Input(component_id="smoothing", component_property="value"),
            Input(component_id="max_migration", component_property="value"),
            Input(component_id="min_channels", component_property="value"),
            Input(component_id="min_amplitude", component_property="value"),
            Input(component_id="min_value", component_property="value"),
            Input(component_id="min_width", component_property="value"),
            Input(component_id="n_groups", component_property="value"),
            Input(component_id="max_separation", component_property="value"),
        )(self.compute_lines)

        # Callbacks to update components of selected line figure
        self.app.callback(
            Output(component_id="figure_lines_trigger", component_property="data"),
            Output(component_id="linear_threshold", component_property="min"),
            Output(component_id="linear_threshold", component_property="max"),
            Output(component_id="linear_threshold", component_property="marks"),
            State(component_id="figure_lines_trigger", component_property="data"),
            Input(component_id="lines_computation_trigger", component_property="data"),
            Input(component_id="line_indices_trigger", component_property="data"),
            Input(component_id="selected_line", component_property="value"),
            Input(component_id="y_scale", component_property="value"),
            Input(component_id="linear_threshold", component_property="value"),
            Input(component_id="trace_map", component_property="data"),
            Input(component_id="min_value", component_property="value"),
            Input(component_id="x_label", component_property="value"),
            Input(component_id="flip_sign", component_property="value"),
        )(self.update_figure_lines)
        self.app.callback(
            Output(component_id="figure_markers_trigger", component_property="data"),
            State(component_id="figure_markers_trigger", component_property="data"),
            Input(component_id="lines_computation_trigger", component_property="data"),
            Input(component_id="line_indices_trigger", component_property="data"),
            Input(component_id="structural_markers", component_property="value"),
            Input(component_id="selected_line", component_property="value"),
            Input(component_id="y_scale", component_property="value"),
            Input(component_id="linear_threshold", component_property="value"),
            Input(component_id="trace_map", component_property="data"),
            Input(component_id="flip_sign", component_property="value"),
        )(self.update_figure_markers)
        self.app.callback(
            Output(component_id="figure_residuals_trigger", component_property="data"),
            State(component_id="figure_residuals_trigger", component_property="data"),
            Input(component_id="lines_computation_trigger", component_property="data"),
            Input(component_id="line_indices_trigger", component_property="data"),
            Input(component_id="show_residuals", component_property="value"),
            Input(component_id="selected_line", component_property="value"),
            Input(component_id="y_scale", component_property="value"),
            Input(component_id="linear_threshold", component_property="value"),
            Input(component_id="trace_map", component_property="data"),
            Input(component_id="flip_sign", component_property="value"),
        )(self.update_figure_residuals)
        # Update property groups colour from color picker, and color picker from property groups.
        self.app.callback(
            Output(component_id="color_picker", component_property="value"),
            Output(component_id="figure_colours_trigger", component_property="data"),
            State(component_id="figure_colours_trigger", component_property="data"),
            Input(component_id="color_picker", component_property="value"),
            Input(component_id="group_name", component_property="value"),
            State(component_id="trace_map", component_property="data"),
        )(self.update_figure_colours)
        self.app.callback(
            Output(component_id="figure_click_data_trigger", component_property="data"),
            State(component_id="figure_click_data_trigger", component_property="data"),
            Input(component_id="lines_computation_trigger", component_property="data"),
            Input(component_id="line_figure", component_property="clickData"),
            Input(component_id="survey_figure", component_property="clickData"),
            Input(component_id="selected_line", component_property="value"),
        )(self.update_figure_click_data)
        # Update the lines, markers, residuals, colours, click data for the selected line figure.
        self.app.callback(
            Output(component_id="line_figure", component_property="figure"),
            Input(component_id="figure_lines_trigger", component_property="data"),
            Input(component_id="figure_markers_trigger", component_property="data"),
            Input(component_id="figure_residuals_trigger", component_property="data"),
            Input(component_id="figure_colours_trigger", component_property="data"),
            Input(component_id="figure_click_data_trigger", component_property="data"),
        )(self.update_selected_line_figure)

        # Update the survey figure.
        self.app.callback(
            Output(component_id="survey_figure", component_property="figure"),
            Input(component_id="lines_computation_trigger", component_property="data"),
            Input(component_id="survey_figure", component_property="figure"),
            Input(component_id="line_figure", component_property="clickData"),
            Input(component_id="survey_figure", component_property="clickData"),
            Input(component_id="selected_line", component_property="value"),
            Input(component_id="n_lines", component_property="value"),
        )(self.update_survey_figure)

        # Save current parameters and run the driver.
        self.app.callback(
            Output(component_id="output_message", component_property="children"),
            Input(component_id="export", component_property="n_clicks"),
            State(component_id="flip_sign", component_property="value"),
            State(component_id="trend_lines", component_property="value"),
            State(component_id="masking_data", component_property="value"),
            State(component_id="smoothing", component_property="value"),
            State(component_id="min_amplitude", component_property="value"),
            State(component_id="min_value", component_property="value"),
            State(component_id="min_width", component_property="value"),
            State(component_id="max_migration", component_property="value"),
            State(component_id="min_channels", component_property="value"),
            State(component_id="n_groups", component_property="value"),
            State(component_id="max_separation", component_property="value"),
            State(component_id="selected_line", component_property="value"),
            State(component_id="ga_group_name", component_property="value"),
            prevent_initial_call=True,
        )(self.trigger_click)

    @property
    def app(self) -> Dash:
        """Dash app"""
        if self._app is None:
            self._app = Dash(
                server=self.server,
                url_base_pathname=os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "/"),
                external_stylesheets=self.external_stylesheets,
            )

        return self._app

    @property
    def active_channels(self) -> dict | None:
        """
        Data for the channels to be plotted.
        """
        if self.property_groups is None:
            return None

        if self._active_channels is None:
            self._active_channels = {}
            for group in self.property_groups.values():
                for channel in group["properties"]:
                    chan = self.workspace.get_entity(uuid.UUID(channel))[0]
                    if isinstance(chan, Data):
                        self._active_channels[channel] = {
                            "name": chan.name,
                            "values": chan.values.copy(),
                        }
        return self._active_channels

    @property
    def figure(self) -> go.Figure | None:
        """
        Selected line figure.
        """
        return self._figure

    @figure.setter
    def figure(self, value):
        self._figure = value

    @property
    def line_indices(self) -> dict | None:
        """
        Line indices for each active line.
        """
        return self._line_indices

    @line_indices.setter
    def line_indices(self, value):
        self._line_indices = value

    @property
    def line_field(self) -> ReferencedData | None:
        """
        Line labels for survey.
        """
        return self._line_field

    @line_field.setter
    def line_field(self, value: str | uuid.UUID | ReferencedData | None):
        if isinstance(value, str | uuid.UUID) and self.survey is not None:
            data = self.survey.get_entity(uuid.UUID(str(value)))[0]

            if not isinstance(data, ReferencedData):
                raise TypeError("Line field must be of type ReferencedData.")

            self._line_field = data

        elif isinstance(value, ReferencedData):
            self._line_field = value
        else:
            self._line_field = None

    @property
    def computed_lines(self) -> dict | None:
        """
        Line anomalies and positions for the current line ids.
        """
        return self._computed_lines

    @computed_lines.setter
    def computed_lines(self, value):
        self._computed_lines = value

    @property
    def survey(self) -> Curve | None:
        """
        Current survey object.
        """
        if self._survey is None and self.params.objects is not None:
            self.workspace: Workspace = Workspace()
            with fetch_active_workspace(self.params.geoh5):
                self._survey = self.params.objects.copy(parent=self.workspace)
                self._line_field = self.params.get_line_field(self._survey)

            self._active_channels = None
            self._ordered_survey_lines = None

        return self._survey

    @property
    def property_groups(self) -> dict | None:
        """
        Property groups data and colours.
        """
        return self._property_groups

    @property_groups.setter
    def property_groups(self, value):
        self._property_groups = value

    @property
    def ordered_survey_lines(self) -> dict | None:
        """
        Order of survey lines.
        """
        if (
            self._ordered_survey_lines is None
            and self.survey is not None
            and self.survey.vertices is not None
            and self.line_field is not None
        ):
            self._ordered_survey_lines = get_ordered_survey_lines(
                self.survey, self.line_field
            )

        return self._ordered_survey_lines

    def set_initialized_layout(self):
        """
        Initialize the app layout from ui.json data.
        """
        self.app.layout = peak_finder_layout

        # Assemble property groups.
        property_groups = self.params.get_property_groups()
        for value in property_groups.values():
            value["data"] = str(value["data"])
            value["properties"] = [str(p) for p in value["properties"]]

        trace_map = self.initialize_line_figure(property_groups)

        self.property_groups = property_groups

        self.app.layout.children += [
            dcc.Store(id="trace_map", data=trace_map),
        ]

        specify_values = {}
        if self.ordered_survey_lines is not None and self.property_groups is not None:
            # Line dropdown options
            selected_line_options = [
                {"label": label, "value": id}
                for id, label in self.ordered_survey_lines.items()
            ]
            # Initial line
            selected_line = None
            if len(selected_line_options) > 0:
                selected_line = selected_line_options[0]["value"]

            specify_values = {
                "selected_line": [
                    {"property": "options", "value": selected_line_options},
                    {"property": "value", "value": selected_line},
                ],
                "group_name": [
                    {
                        "property": "options",
                        "value": list(self.property_groups.keys()),
                    }
                ],
            }

        BaseDashApplication.init_vals(
            self.app.layout.children, self._ui_json_data, kwargs=specify_values
        )

    @staticmethod
    def update_plot_visibility(plot_selection: list[str]) -> tuple:
        """
        Update which plots are visible based on dropdown selection.

        :param plot_selection: Dropdown selection.

        :return: Visibility of line plot and survey plot.
        """

        if plot_selection is None:
            return no_update, no_update

        output = []
        if "Line figure" in plot_selection:
            output.append({"display": "block"})
        else:
            output.append({"display": "none"})
        if "Survey figure" in plot_selection:
            output.append({"display": "block"})
        else:
            output.append({"display": "none"})
        return tuple(output)

    @staticmethod
    def update_widget_visibility(widget_selection: str) -> tuple:
        """
        Update which widgets are visible based on dropdown selection.

        :param widget_selection: Dropdown selection.

        :return: Visibility of data selection widgets.
        """
        if widget_selection is None:
            return no_update, no_update, no_update

        if widget_selection == "Visual parameters":
            return (
                {"display": "block"},
                {"display": "none"},
                no_update,
            )

        if widget_selection == "Detection parameters":
            return {"display": "none"}, {"display": "block"}, []

        return no_update, no_update, no_update

    @staticmethod
    def update_group_selection(widget_selection: str) -> list[bool]:
        if widget_selection == "Visual parameters":
            return no_update
        return []

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

    def update_masking_dropdowns(self, survey_trigger: int) -> list[dict]:
        """
        Initialize data and line field dropdowns from input object.

        :param survey_trigger: Trigger indicating survey object has been updated.

        :return: Masking data dropdown options.
        """
        masking_data_options = [{"label": "None", "value": "None"}]
        if self.survey is None or not hasattr(self.survey, "children"):
            return no_update
        for child in self.survey.children:
            if isinstance(child, BooleanData):
                masking_data_options.append(
                    {"label": child.name, "value": "{" + str(child.uid) + "}"}
                )
        return masking_data_options

    def update_survey_mask(
        self,
        survey_trigger: int,
        masking_data: str,
        flip_sign: bool,
    ) -> tuple[int, float]:
        """
        Apply masking to survey object.

        :param survey_trigger: Survey trigger to update.
        :param masking_data: Masking data.
        :param flip_sign: Whether to flip the sign of the data.

        :return: Trigger to indicate survey object has been updated.
        :return: Minimum value for figure, which changes on object change.
        """
        if self.survey is None:
            return no_update, no_update

        self._survey = None
        if masking_data is not None and masking_data != "None":
            if self.survey is not None and hasattr(self.survey, "remove_vertices"):
                masking_data_obj = self.survey.get_data(uuid.UUID(masking_data))[0]
                masking_array = masking_data_obj.values
                if masking_array is not None:
                    self.survey.remove_vertices(~masking_array)

        self.computed_lines = None

        if isinstance(self.line_field, Data):
            self.line_field = self.line_field.uid  # type: ignore

        min_value = no_update
        if self.active_channels is not None:
            sign = 1
            if flip_sign:
                sign *= -1

            d_min, d_max = np.inf, -np.inf
            for channel in self.active_channels:
                d_min = np.nanmin(
                    [
                        d_min,
                        sign * self.active_channels[channel]["values"].min(),  # type: ignore
                    ]
                )
                d_max = np.nanmax(
                    [
                        d_max,
                        sign * self.active_channels[channel]["values"].max(),  # type: ignore
                    ]
                )
            if d_max > -np.inf:
                min_value = d_min

        return survey_trigger + 1, min_value

    def get_active_line_ids(self, selected_line: int, n_lines: int) -> list[int]:
        """
        Get active line IDs for plotting.

        :param selected_line: Selected line ID.
        :param n_lines: Number of outward survey lines.

        :return: Active line IDs.
        """
        if self.ordered_survey_lines is None:
            return []

        survey_line_ids = list(self.ordered_survey_lines.keys())
        selected_line_ind = survey_line_ids.index(selected_line)

        survey_lines = survey_line_ids[
            max(0, selected_line_ind - n_lines) : min(
                len(survey_line_ids), selected_line_ind + n_lines + 1
            )
        ]

        return survey_lines

    def update_line_indices(  # pylint: disable=too-many-locals
        self,
        line_indices_trigger: int,
        survey_trigger: int,
        selected_line: int,
        n_lines: int,
    ) -> int:
        """
        Get line indices for plotting.

        :param line_indices_trigger: Trigger for updating line indices.
        :param survey_trigger: Trigger for updating survey object.
        :param selected_line: Selected line ID.
        :param n_lines: Number of outward survey lines.

        :return: Trigger indicating line indices have been updated.
        """
        if (
            isinstance(self.survey, Curve)
            and isinstance(self.line_field, Data)
            and hasattr(self.line_field, "values")
            and selected_line is not None
            and n_lines is not None
        ):
            survey_lines = self.get_active_line_ids(selected_line, n_lines)
            self.line_indices = PeakFinderDriver.get_line_indices(
                self.survey, self.line_field, survey_lines
            )
            line_indices_trigger += 1
        return line_indices_trigger

    def compute_lines(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        *,
        lines_computation_trigger: int,
        line_indices_trigger: int,
        survey_trigger: int,
        selected_line: int,
        n_lines: int,
        smoothing: float,
        max_migration: float,
        min_channels: int,
        min_amplitude: float,
        min_value: float,
        min_width: float,
        n_groups: int,
        max_separation: float,
    ) -> int | None:
        """
        Compute line anomalies.

        :param lines_computation_trigger: Trigger for updating the line computation.
        :param line_indices_trigger: Trigger for updating line indices.
        :param survey_trigger: Trigger indicating survey has been updated.
        :param selected_line: Selected line ID.
        :param n_lines: Number of outward survey lines.
        :param smoothing: Smoothing factor.
        :param max_migration: Maximum peak migration.
        :param min_channels: Minimum number of channels in anomaly.
        :param min_amplitude: Minimum amplitude of anomaly as percent.
        :param min_value: Minimum data value of anomaly.
        :param min_width: Minimum width of anomaly in meters.
        :param n_groups: Number of groups to use for grouping anomalies.
        :param max_separation: Maximum separation between anomalies in meters.

        :return: Trigger indicating line computation has been updated.
        """
        try:
            triggers = [t["prop_id"].split(".")[0] for t in callback_context.triggered]
        except MissingCallbackContextException:
            triggers = []

        if (
            self.survey is None
            or self.line_indices is None
            or self.property_groups is None
            or selected_line is None
        ):
            return no_update

        property_groups = [
            self.survey.find_or_create_property_group(name=name)  # type: ignore
            for name in self.property_groups
        ]

        survey_lines = self.get_active_line_ids(selected_line, n_lines)
        survey_lines_subset = survey_lines
        if (
            self.computed_lines is not None
            and "survey_trigger" not in triggers
            and "n_lines" in triggers
        ):
            survey_lines_subset = [
                line for line in survey_lines if line not in self.computed_lines
            ]

        line_computation = PeakFinderDriver.compute_lines(
            survey=self.survey,  # type: ignore
            line_indices_dict=self.line_indices,
            line_ids=survey_lines_subset,
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

        with ProgressBar():
            results = compute(line_computation)[0]

        # Remove un-needed lines
        if self.computed_lines is not None and "n_lines" in triggers:
            entries_to_remove = [
                line for line in self.computed_lines if line not in survey_lines
            ]
            for key in entries_to_remove:
                self.computed_lines.pop(key, None)
        else:
            self.computed_lines = {}

        # Add new lines
        for line_anomaly in tqdm(results):
            if "n_lines" in triggers and line_anomaly.line_id in self.computed_lines:
                continue

            line_groups = line_anomaly.anomalies
            line_anomalies: list[AnomalyGroup] = []
            if line_groups is not None:
                for line_group in line_groups:
                    line_anomalies += line_group.groups

            if line_anomaly.line_id not in self.computed_lines:
                self.computed_lines[line_anomaly.line_id] = {
                    "position": [],
                    "anomalies": [],
                    "plot_line_start": np.inf,
                }

            # Add position to self.lines
            self.computed_lines[line_anomaly.line_id]["position"].append(
                line_anomaly.position
            )

            self.computed_lines[line_anomaly.line_id]["plot_line_start"] = min(
                np.min(line_anomaly.position.locations_resampled),
                self.computed_lines[line_anomaly.line_id]["plot_line_start"],
            )

            self.computed_lines[line_anomaly.line_id]["anomalies"].append(
                line_anomalies
            )

        return lines_computation_trigger + 1

    def update_selected_line_figure(self, *args) -> go.Figure | None:
        """
        :param args: Triggers for updating the figure.

        :return: Updated figure.
        """
        return self.figure

    def update_figure_lines(  # pylint: disable=too-many-arguments, too-many-locals, too-many-branches  # noqa: C901
        self,
        *,
        figure_lines_trigger: int,
        lines_computation_trigger: int,
        line_indices_trigger: dict,
        selected_line: int,
        y_scale: str,
        linear_threshold: float,
        trace_map: dict,
        min_value: float,
        x_label: str,
        flip_sign: bool,
    ) -> tuple:
        """
        Update the figure lines data.

        :param figure_lines_trigger: Trigger for updating the figure lines data.
        :param lines_computation_trigger: Trigger indicating line computation has been updated.
        :param line_indices_trigger: Trigger indicating line indices have been updated.
        :param selected_line: Selected line ID.
        :param y_scale: Whether y-axis ticks are linear or symlog.
        :param linear_threshold: Linear threshold.
        :param trace_map: Dict mapping trace names to indices.
        :param min_value: Minimum value for figure.
        :param x_label: Label for x-axis.
        :param flip_sign: Whether to flip the sign of the data.

        :return: Trigger for updating the figure lines data.
        :return: Linear threshold slider min.
        :return: Linear threshold slider max.
        :return: Linear threshold slider marks.
        """
        if (
            self.active_channels is None
            or self.computed_lines is None
            or selected_line is None
            or self.figure is None
            or self.line_indices is None
        ):
            return no_update, no_update, no_update, no_update

        sign = 1
        if flip_sign:
            sign *= -1

        y_min, y_max = np.inf, -np.inf
        log = y_scale == "symlog"
        threshold = np.float_power(10, linear_threshold)
        values_list = []

        trace_dict: dict[str, dict[str, dict]] = {
            "lines": {
                "lines": {
                    "x": [None],
                    "y": [None],
                }
            },
            "property_groups": {},
            "markers": {},
        }

        for channel_dict in list(  # pylint: disable=too-many-nested-blocks
            self.active_channels.values()
        ):
            if "values" not in channel_dict or selected_line not in self.computed_lines:
                continue
            full_values = sign * np.array(channel_dict["values"])

            for position, anomalies in zip(
                self.computed_lines[selected_line]["position"],
                self.computed_lines[selected_line]["anomalies"],
                strict=True,
            ):
                locs = position.locations_resampled

                if position.line_indices.sum() < 2 or locs is None:
                    continue

                values = full_values[position.line_indices]
                values, _ = position.resample_values(values)
                values_list += list(values.flatten())

                if log:
                    sym_values = symlog(values, threshold)
                else:
                    sym_values = values

                y_min = np.nanmin([sym_values.min(), y_min])
                y_max = np.nanmax([sym_values.max(), y_max])

                trace_dict["lines"]["lines"]["x"] += list(locs) + [None]  # type: ignore
                trace_dict["lines"]["lines"]["y"] += list(sym_values) + [None]  # type: ignore

                for anomaly_group in anomalies:
                    for subgroup in anomaly_group.subgroups:
                        channels = np.array(
                            [a.parent.data_entity.name for a in subgroup.anomalies]
                        )
                        group_name = subgroup.property_group.name
                        query = np.where(np.array(channels) == channel_dict["name"])[0]
                        if len(query) == 0:
                            continue

                        for i in query:
                            start = subgroup.anomalies[i].start
                            end = subgroup.anomalies[i].end

                            if group_name not in trace_dict["property_groups"]:  # type: ignore
                                trace_dict["property_groups"][group_name] = {  # type: ignore
                                    "x": [None],
                                    "y": [None],
                                    "customdata": [None],
                                }
                            trace_dict["property_groups"][group_name]["x"] += list(
                                locs[start:end]
                            ) + [None]
                            trace_dict["property_groups"][group_name]["y"] += list(
                                sym_values[start:end]
                            ) + [None]
                            trace_dict["property_groups"][group_name]["customdata"] += (
                                list(values[start:end]) + [None]
                            )

        if np.isinf(y_min) or self.property_groups is None:
            return no_update, None, None, None

        all_values = np.array(values_list)
        _, y_label, y_tickvals, y_ticktext = format_axis(
            channel="Data",
            axis=all_values,
            log=log,
            threshold=threshold,
        )

        # Remove traces in trace map but not trace dict from plot
        remaining_traces = set(self.property_groups.keys()) - set(
            trace_dict["property_groups"].keys()
        )

        for trace in remaining_traces:
            self.figure.data[trace_map[trace]]["x"] = [None]
            self.figure.data[trace_map[trace]]["y"] = [None]
            if "customdata" in self.figure.data[trace_map[trace]]:
                self.figure.data[trace_map[trace]]["customdata"] = [None]

        # Update data on traces
        for trace_name in ["lines", "property_groups"]:
            if trace_name in trace_dict:
                for key, value in trace_dict[trace_name].items():  # type: ignore
                    self.figure.data[trace_map[key]]["x"] = value["x"]
                    self.figure.data[trace_map[key]]["y"] = value["y"]
                    if "customdata" in value:
                        self.figure.data[trace_map[key]]["customdata"] = value[
                            "customdata"
                        ]

        # Update linear threshold
        pos_vals = all_values[all_values > 0]  # type: ignore

        thresh_min = np.log10(np.min(pos_vals))
        thresh_max = np.log10(np.max(pos_vals))
        thresh_ticks = {
            t: "10E" + f"{t:.2g}" for t in np.linspace(thresh_min, thresh_max, 5)
        }

        # Update figure layout
        self.update_figure_layout(
            y_label=y_label,
            y_tickvals=y_tickvals,
            y_ticktext=y_ticktext,
            y_min=y_min,
            y_max=y_max,
            min_value=symlog(min_value, threshold),
            x_label=x_label,
            line_position=self.computed_lines[selected_line]["position"],
        )
        return (
            figure_lines_trigger + 1,
            thresh_min,
            thresh_max,
            thresh_ticks,
        )

    @staticmethod
    def add_markers(  # pylint: disable=too-many-arguments, too-many-locals
        *,
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
        Format marker arrays as traces and add to trace_dict.

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
                "marker_color": ["black"],
            }
        trace_dict["markers"]["peaks"]["x"] = peak_markers_x
        trace_dict["markers"]["peaks"]["y"] = peak_markers_y
        trace_dict["markers"]["peaks"]["customdata"] = peak_markers_customdata
        trace_dict["markers"]["peaks"]["marker_color"] = peak_markers_c

        if "start_markers" not in trace_dict["markers"]:
            trace_dict["markers"]["start_markers"] = {
                "x": [None],
                "y": [None],
                "customdata": [None],
            }
        trace_dict["markers"]["start_markers"]["x"] = start_markers_x
        trace_dict["markers"]["start_markers"]["y"] = start_markers_y
        trace_dict["markers"]["start_markers"]["customdata"] = start_markers_customdata

        if "end_markers" not in trace_dict["markers"]:
            trace_dict["markers"]["end_markers"] = {
                "x": [None],
                "y": [None],
                "customdata": [None],
            }
        trace_dict["markers"]["end_markers"]["x"] = end_markers_x
        trace_dict["markers"]["end_markers"]["y"] = end_markers_y
        trace_dict["markers"]["end_markers"]["customdata"] = end_markers_customdata

        if "up_markers" not in trace_dict["markers"]:
            trace_dict["markers"]["up_markers"] = {
                "x": [None],
                "y": [None],
                "customdata": [None],
            }
        trace_dict["markers"]["up_markers"]["x"] = up_markers_x
        trace_dict["markers"]["up_markers"]["y"] = up_markers_y
        trace_dict["markers"]["up_markers"]["customdata"] = up_markers_customdata

        if "down_markers" not in trace_dict["markers"]:
            trace_dict["markers"]["down_markers"] = {
                "x": [None],
                "y": [None],
                "customdata": [None],
            }
        trace_dict["markers"]["down_markers"]["x"] = dwn_markers_x
        trace_dict["markers"]["down_markers"]["y"] = dwn_markers_y
        trace_dict["markers"]["down_markers"]["customdata"] = dwn_markers_customdata

        return trace_dict

    def update_figure_markers(  # noqa: C901  # pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
        self,
        *,
        figure_markers_trigger: int,
        lines_computation_trigger: int,
        line_indices_trigger: int,
        show_markers: list[bool],
        selected_line: int,
        y_scale: str,
        linear_threshold: float,
        trace_map: dict,
        flip_sign: bool,
    ) -> int:
        """
        Update the figure markers data.

        :param figure_markers_trigger: Trigger for updating the figure markers data.
        :param lines_computation_trigger: Trigger indicating line computation has been updated.
        :param line_indices_trigger: Trigger indicating line indices have been updated.
        :param show_markers: Whether to show markers.
        :param selected_line: Selected line ID.
        :param y_scale: Whether y-axis ticks are linear or symlog.
        :param linear_threshold: Linear threshold.
        :param trace_map: Dict mapping trace names to indices.
        :param flip_sign: Whether to flip the sign of the data.

        :return: Trigger for updating the figure markers data.
        """
        if (
            self.active_channels is None
            or self.computed_lines is None
            or self.figure is None
            or self.line_indices is None
            or self.property_groups is None
        ):
            return no_update

        sign = 1
        if flip_sign:
            sign *= -1

        if not show_markers:
            self.figure.data[trace_map["markers_legend"]]["visible"] = False
            for trace_name in [
                "peaks",
                "start_markers",
                "end_markers",
                "up_markers",
                "down_markers",
                "left_azimuth",
                "right_azimuth",
            ]:
                if trace_name in trace_map:
                    self.figure.data[trace_map[trace_name]]["x"] = []
                    self.figure.data[trace_map[trace_name]]["y"] = []
            return figure_markers_trigger + 1
        self.figure.data[trace_map["markers_legend"]]["visible"] = True

        log = y_scale == "symlog"
        threshold = np.float_power(10, linear_threshold)
        all_values = []
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

        trace_dict: dict[str, dict] = {
            "markers": {
                "left_azimuth": {
                    "x": [None],
                    "y": [None],
                    "customdata": [None],
                },
                "right_azimuth": {
                    "x": [None],
                    "y": [None],
                    "customdata": [None],
                },
            },
        }

        for position, anomalies in zip(
            self.computed_lines[selected_line]["position"],
            self.computed_lines[selected_line]["anomalies"],
            strict=True,
        ):
            indices = position.line_indices

            if indices.sum() < 2:
                continue
            locs = position.locations_resampled

            for channel_dict in list(self.active_channels.values()):
                if "values" not in channel_dict:
                    continue

                values = sign * np.array(channel_dict["values"])[indices]
                values, _ = position.resample_values(values)
                all_values += list(values.flatten())

                if log:
                    sym_values = symlog(values, threshold)
                else:
                    sym_values = values

                for anomaly_group in anomalies:
                    for subgroup in anomaly_group.subgroups:
                        channels = np.array(
                            [a.parent.data_entity.name for a in subgroup.anomalies]
                        )
                        group_name = subgroup.property_group.name
                        color = self.property_groups[group_name]["color"]
                        query = np.where(np.array(channels) == channel_dict["name"])[0]
                        if len(query) == 0:
                            continue

                        i = query[0]
                        peak = subgroup.peaks[i]
                        peak_markers_x += [locs[peak]]
                        peak_markers_y += [sym_values[peak]]
                        peak_markers_customdata += [values[peak]]
                        peak_markers_c += [color]
                        start_markers_x += [locs[subgroup.anomalies[i].start]]
                        start_markers_y += [sym_values[subgroup.anomalies[i].start]]
                        start_markers_customdata += [
                            values[subgroup.anomalies[i].start]
                        ]
                        end_markers_x += [locs[subgroup.anomalies[i].end]]
                        end_markers_y += [sym_values[subgroup.anomalies[i].end]]
                        end_markers_customdata += [values[subgroup.anomalies[i].end]]
                        up_markers_x += [locs[subgroup.anomalies[i].inflect_up]]
                        up_markers_y += [sym_values[subgroup.anomalies[i].inflect_up]]
                        up_markers_customdata += [
                            values[subgroup.anomalies[i].inflect_up]
                        ]
                        dwn_markers_x += [locs[subgroup.anomalies[i].inflect_down]]
                        dwn_markers_y += [
                            sym_values[subgroup.anomalies[i].inflect_down]
                        ]
                        dwn_markers_customdata += [
                            values[subgroup.anomalies[i].inflect_down]
                        ]

        # Add markers to trace_dict
        trace_dict = PeakFinder.add_markers(
            trace_dict=trace_dict,
            peak_markers_x=peak_markers_x,
            peak_markers_y=peak_markers_y,
            peak_markers_customdata=peak_markers_customdata,
            peak_markers_c=peak_markers_c,
            start_markers_x=start_markers_x,
            start_markers_y=start_markers_y,
            start_markers_customdata=start_markers_customdata,
            end_markers_x=end_markers_x,
            end_markers_y=end_markers_y,
            end_markers_customdata=end_markers_customdata,
            up_markers_x=up_markers_x,
            up_markers_y=up_markers_y,
            up_markers_customdata=up_markers_customdata,
            dwn_markers_x=dwn_markers_x,
            dwn_markers_y=dwn_markers_y,
            dwn_markers_customdata=dwn_markers_customdata,
        )

        # Update figure markers from trace_dict
        if "markers" in trace_dict:
            for key, value in trace_dict["markers"].items():  # type: ignore
                self.figure.data[trace_map[key]]["x"] = value["x"]
                self.figure.data[trace_map[key]]["y"] = value["y"]
                if "customdata" in value:
                    self.figure.data[trace_map[key]]["customdata"] = value["customdata"]
                if "marker_color" in value:
                    self.figure.data[trace_map[key]]["marker_color"] = value[
                        "marker_color"
                    ]

        return figure_markers_trigger + 1

    def add_residuals(
        self,
        values: np.ndarray,
        raw: np.ndarray,
        locs: np.ndarray,
    ):
        """
        Add residuals to the figure.

        :param values: Resampled values.
        :param raw: Raw values.
        :param locs: Locations.
        """
        if self.figure is None:
            return

        pos_inds = np.where(raw > values)[0]
        neg_inds = np.where(raw < values)[0]

        pos_residuals = raw.copy()
        pos_residuals[pos_inds] = values[pos_inds]
        neg_residuals = raw.copy()
        neg_residuals[neg_inds] = values[neg_inds]

        self.figure.add_trace(
            go.Scatter(
                x=locs,
                y=values,
                line={"color": "rgba(0, 0, 0, 0)"},
                showlegend=False,
                hoverinfo="skip",
            ),
        )
        self.figure.add_trace(
            go.Scatter(
                x=locs,
                y=pos_residuals,
                line={"color": "rgba(0, 0, 0, 0)"},
                fill="tonexty",
                fillcolor="rgba(255, 0, 0, 0.5)",
                name="positive residuals",
                legendgroup="positive residuals",
                showlegend=False,
                visible=True,
                hoverinfo="skip",
            )
        )

        self.figure.add_trace(
            go.Scatter(
                x=locs,
                y=values,
                line={"color": "rgba(0, 0, 0, 0)"},
                showlegend=False,
                hoverinfo="skip",
            ),
        )
        self.figure.add_trace(
            go.Scatter(
                x=locs,
                y=neg_residuals,
                line={"color": "rgba(0, 0, 0, 0)"},
                fill="tonexty",
                fillcolor="rgba(0, 0, 255, 0.5)",
                name="negative residuals",
                legendgroup="negative residuals",
                showlegend=False,
                visible=True,
                hoverinfo="skip",
            )
        )

    def update_figure_residuals(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        *,
        figure_residuals_trigger: int,
        lines_computation_trigger: int,
        line_indices_trigger: int,
        show_residuals: list[bool],
        selected_line: int,
        y_scale: str,
        linear_threshold: float,
        trace_map: dict,
        flip_sign: bool,
    ) -> int:
        """
        Add residuals to figure.

        :param figure_residuals_trigger: Trigger for updating the figure residuals data.
        :param lines_computation_trigger: Trigger indicating line computation has been updated.
        :param line_indices_trigger: Trigger indicating line indices have been updated.
        :param show_residuals: Whether to show residuals.
        :param selected_line: Selected line ID.
        :param y_scale: Whether y-axis ticks are linear or symlog.
        :param linear_threshold: Linear threshold.
        :param trace_map: Dict mapping trace names to indices.
        :param flip_sign: Whether to flip the sign of the data.

        :return: Trigger for updating the figure residuals data.
        """
        if (
            self.active_channels is None
            or self.computed_lines is None
            or not self.computed_lines
            or selected_line is None
            or self.figure is None
            or self.line_indices is None
        ):
            return no_update

        sign = 1
        if flip_sign:
            sign *= -1

        triggers = [t["prop_id"].split(".")[0] for t in callback_context.triggered]
        if "lines_computation_trigger" in triggers or (
            "show_residuals" in triggers and not show_residuals
        ):
            for ind in range(len(trace_map), len(self.figure.data)):
                self.figure.data[ind]["x"] = []
                self.figure.data[ind]["y"] = []

        if not show_residuals:
            self.figure.data[trace_map["pos_residuals_legend"]]["visible"] = False
            self.figure.data[trace_map["neg_residuals_legend"]]["visible"] = False
            return figure_residuals_trigger + 1

        self.figure.data[trace_map["pos_residuals_legend"]]["visible"] = True
        self.figure.data[trace_map["neg_residuals_legend"]]["visible"] = True

        log = y_scale == "symlog"
        threshold = np.float_power(10, linear_threshold)

        for position, anomalies in zip(
            self.computed_lines[selected_line]["position"],
            self.computed_lines[selected_line]["anomalies"],
            strict=True,
        ):
            indices = position.line_indices

            if indices.sum() < 2:
                continue
            locs = position.locations_resampled

            for channel_dict in list(self.active_channels.values()):
                if "values" not in channel_dict:
                    continue

                values = sign * np.array(channel_dict["values"])[indices]
                values, raw = position.resample_values(values)

                if log:
                    sym_values = symlog(values, threshold)
                    sym_raw = symlog(raw, threshold)
                else:
                    sym_values = values
                    sym_raw = raw

                for anomaly_group in anomalies:
                    channels = np.array(
                        [a.parent.data_entity.name for a in anomaly_group.anomalies]
                    )
                    query = np.where(np.array(channels) == channel_dict["name"])[0]
                    if len(query) == 0:
                        continue

                self.add_residuals(
                    sym_values,
                    sym_raw,
                    locs,
                )
        return figure_residuals_trigger + 1

    def update_figure_colours(
        self,
        figure_colours_trigger: int,
        color_picker: dict,
        group_name: str,
        trace_map: dict,
    ) -> tuple[dict, int]:
        """
        Update property groups color on color_picker change.
        Update color picker on group name dropdown change.

        :param figure_colours_trigger: Trigger for updating figure colours.
        :param color_picker: Color picker hex value.
        :param group_name: Name of property group from dropdown.
        :param trace_map: Dict mapping figure trace indices to trace name.

        :return: Updated color picker.
        :return: Trigger to update figure colours.
        """
        if self.property_groups is None:
            return no_update, no_update

        color_picker_out = no_update
        trigger = ctx.triggered_id

        if trigger == "group_name":
            color_picker_out = {"hex": self.property_groups[group_name]["color"]}
        elif trigger == "color_picker":
            original_colour = self.property_groups[group_name]["color"]
            new_colour = str(color_picker["hex"])
            self.property_groups[group_name]["color"] = new_colour
            # Update line colours
            if self.figure is not None:
                if group_name in trace_map:
                    self.figure.data[trace_map[group_name]]["line_color"] = new_colour
                    # Update marker colours
                    colour_array = np.array(
                        self.figure.data[trace_map["peaks"]]["marker_color"]
                    )
                    colour_array[colour_array == str(original_colour)] = new_colour
                    self.figure.data[trace_map["peaks"]]["marker_color"] = colour_array

        return color_picker_out, figure_colours_trigger + 1

    def update_figure_click_data(
        self,
        *,
        figure_click_data_trigger: int,
        lines_computation_trigger: int,
        line_click_data: dict | None,
        full_lines_click_data: dict | None,
        selected_line: int,
    ) -> int:
        """
        Update the markers on the single line figure from clicking on either figure.

        :param figure_click_data_trigger: Trigger for updating the figure click data.
        :param lines_computation_trigger: Trigger indicating line computation has been updated.
        :param line_click_data: Click data from the single line figure.
        :param full_lines_click_data: Click data from the full lines figure.
        :param selected_line: Selected line ID.

        :return: Trigger for updating the click data.
        """
        if (
            self.figure is None
            or self.figure.layout.shapes is None
            or self.computed_lines is None
        ):
            return no_update

        if len(self.figure.layout.shapes) == 0:
            self.figure.add_vline(x=0)

        triggers = [t["prop_id"].split(".")[0] for t in callback_context.triggered]

        if (
            "lines_computation_trigger" in triggers
            and selected_line in self.computed_lines
        ):
            self.figure.update_shapes({"x0": 0, "x1": 0})
        elif line_click_data is not None and "line_figure" in triggers:
            x_val = line_click_data["points"][0]["x"]
            self.figure.update_shapes({"x0": x_val, "x1": x_val})
        elif full_lines_click_data is not None and "survey_figure" in triggers:
            x_locs = np.concatenate(
                tuple(
                    pos.x_locations
                    for pos in self.computed_lines[selected_line]["position"]
                )
            )
            y_locs = np.concatenate(
                tuple(
                    pos.y_locations
                    for pos in self.computed_lines[selected_line]["position"]
                )
            )
            min_index = np.argmin(x_locs)
            x_min = x_locs[min_index]
            y_min = y_locs[min_index]

            x_val = np.linalg.norm(
                [
                    full_lines_click_data["points"][0]["x"] - x_min,
                    full_lines_click_data["points"][0]["y"] - y_min,
                ]
            )

            self.figure.update_shapes({"x0": x_val, "x1": x_val})

        return figure_click_data_trigger + 1

    def update_figure_layout(  # pylint: disable=too-many-arguments
        self,
        *,
        y_label: str | None,
        y_tickvals: np.ndarray | None,
        y_ticktext: list[str] | None,
        y_min: float | None,
        y_max: float | None,
        min_value: float,
        x_label: str,
        line_position: LinePosition,
    ):
        """
        Update the figure layout.

        :param y_label: Label for y-axis.
        :param y_tickvals: Y-axis tick values.
        :param y_ticktext: Y-axis tick text.
        :param y_min: Minimum y-axis value.
        :param y_max: Maximum y-axis value.
        :param min_value: Minimum value.
        :param x_label: X-axis label.
        """
        if self.figure is None:
            return

        if y_min is not None and y_max is not None:
            self.figure.update_layout(
                {"yaxis_range": [np.nanmax([y_min, min_value]), y_max]}
            )
        if y_label is not None:
            self.figure.update_layout(
                {
                    "yaxis_title": y_label,
                }
            )
        if y_tickvals is not None and y_ticktext is not None:
            self.figure.update_layout(
                {
                    "yaxis_tickvals": y_tickvals,
                    "yaxis_ticktext": [f"{y:.2e}" for y in y_ticktext],
                }
            )

        self.figure.update_layout(
            {"xaxis_title": x_label + " (m)", "yaxis_tickformat": ".2e"}
        )

    def initialize_line_figure(
        self,
        property_groups: dict,
    ) -> dict:
        """
        Add initial, empty traces to figure and return dict mapping trace names to indices.

        :param property_groups: Property groups dictionary.

        :return: Dict mapping trace names to indices.
        """
        self.figure = go.Figure()

        # Add full lines
        all_traces = {
            "lines": {
                "x": [None],
                "y": [None],
                "mode": "lines",
                "name": "full lines",
                "line_color": "lightgrey",
                "showlegend": False,
                "hoverinfo": "skip",
            },
        }

        # Add property groups
        for ind, (key, val) in enumerate(property_groups.items()):
            all_traces[key] = {
                "x": [None],
                "y": [None],
                "mode": "lines",
                "name": key,
                "line_color": val["color"],
                "hovertemplate": (
                    "<b>x</b>: %{x:,.2f} <br>" + "<b>y</b>: %{customdata:,.2e}"
                ),
            }

        # Add markers
        all_traces.update(
            {
                "markers_legend": {
                    "x": [None],
                    "y": [None],
                    "mode": "markers",
                    "marker_color": "black",
                    "marker_symbol": "circle",
                    "legendgroup": "markers",
                    "name": "markers",
                    "visible": True,
                    "showlegend": True,
                },
                "pos_residuals_legend": {
                    "x": [None],
                    "y": [None],
                    "mode": "lines",
                    "line_color": "rgba(255, 0, 0, 0.5)",
                    "line_width": 8,
                    "legendgroup": "positive residuals",
                    "name": "positive residuals",
                    "visible": True,
                    "showlegend": True,
                },
                "neg_residuals_legend": {
                    "x": [None],
                    "y": [None],
                    "mode": "lines",
                    "line_color": "rgba(0, 0, 255, 0.5)",
                    "line_width": 8,
                    "legendgroup": "negative residuals",
                    "name": "negative residuals",
                    "visible": True,
                    "showlegend": True,
                },
            }
        )
        for ori in ["left", "right"]:
            all_traces[ori + "_azimuth"] = {
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
                "visible": True,
                "hovertemplate": (
                    "<b>x</b>: %{x:,.2f} <br>" + "<b>y</b>: %{customdata:,.2e}"
                ),
            }

        all_traces.update(
            {
                "peaks": {
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
                    "visible": True,
                    "hovertemplate": (
                        "<b>x</b>: %{x:,.2f} <br>" + "<b>y</b>: %{customdata:,.2e}"
                    ),
                },
                "start_markers": {
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
                    "visible": True,
                    "hovertemplate": (
                        "<b>x</b>: %{x:,.2f} <br>" + "<b>y</b>: %{customdata:,.2e}"
                    ),
                },
                "end_markers": {
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
                    "visible": True,
                    "hovertemplate": (
                        "<b>x</b>: %{x:,.2f} <br>" + "<b>y</b>: %{customdata:,.2e}"
                    ),
                },
                "up_markers": {
                    "x": [None],
                    "y": [None],
                    "customdata": [None],
                    "mode": "markers",
                    "marker_color": "black",
                    "marker_symbol": "y-up-open",
                    "marker_size": 6,
                    "name": "up markers",
                    "legendgroup": "markers",
                    "showlegend": False,
                    "visible": True,
                    "hovertemplate": (
                        "<b>x</b>: %{x:,.2f} <br>" + "<b>y</b>: %{customdata:,.2e}"
                    ),
                },
                "down_markers": {
                    "x": [None],
                    "y": [None],
                    "customdata": [None],
                    "mode": "markers",
                    "marker_color": "black",
                    "marker_symbol": "y-down-open",
                    "marker_size": 6,
                    "name": "down markers",
                    "legendgroup": "markers",
                    "showlegend": False,
                    "visible": True,
                    "hovertemplate": (
                        "<b>x</b>: %{x:,.2f} <br>" + "<b>y</b>: %{customdata:,.2e}"
                    ),
                },
            }
        )

        trace_map = {}
        for ind, (key, trace) in enumerate(all_traces.items()):
            self.figure.add_trace(go.Scatter(**trace))
            trace_map[key] = ind

        self.figure.add_vline(x=None)
        self.figure.update_layout(margin={"t": 20, "l": 20, "b": 20, "r": 20})
        return trace_map

    def update_survey_figure(  # pylint: disable=too-many-arguments, too-many-locals, too-many-branches
        self,
        *,
        lines_computation_trigger: int,
        figure: dict | None,
        line_click_data: dict | None,
        survey_figure_click_data: dict | None,
        selected_line: int,
        n_lines: int,
    ) -> go.Figure:
        """
        Update the full lines figure.

        :param lines_computation_trigger: Trigger indicating line computation has been updated.
        :param figure: Figure dictionary.
        :param line_click_data: Line figure click data.
        :param survey_figure_click_data: Survey figure click data.
        :param selected_line: Selected line ID.
        :param n_lines: Number of lines outward from the selected line to plot.

        :return: Full lines figure.
        """
        triggers = [t["prop_id"].split(".")[0] for t in callback_context.triggered]
        if figure is not None:
            if (
                line_click_data is not None
                and "line_figure" in triggers
                and self.computed_lines is not None
            ):
                x_locs = np.concatenate(
                    tuple(
                        pos.x_locations
                        for pos in self.computed_lines[selected_line]["position"]
                    )
                )
                y_locs = np.concatenate(
                    tuple(
                        pos.y_locations
                        for pos in self.computed_lines[selected_line]["position"]
                    )
                )
                # Get start of line
                start_ind = np.argmin(x_locs)
                x_min = x_locs[start_ind]
                y_min = y_locs[start_ind]

                # Get distances along line
                dists = np.linalg.norm(np.c_[x_locs - x_min, y_locs - y_min], axis=1)

                # Get point closest to this distance along the line
                click_dist = line_click_data["points"][0]["x"]
                closest_ind = (np.abs(dists - click_dist)).argmin()

                x_val = x_locs[closest_ind]
                y_val = y_locs[closest_ind]

                figure["data"][-1]["x"] = [x_val]
                figure["data"][-1]["y"] = [y_val]
                return figure
            if survey_figure_click_data is not None and "survey_figure" in triggers:
                x_val = survey_figure_click_data["points"][0]["x"]
                y_val = survey_figure_click_data["points"][0]["y"]
                figure["data"][-1]["x"] = [x_val]
                figure["data"][-1]["y"] = [y_val]
                return figure

        figure = go.Figure()
        if (
            self.computed_lines is None
            or selected_line is None
            or self.ordered_survey_lines is None
            or self.property_groups is None
        ):
            return figure

        survey_lines = self.get_active_line_ids(selected_line, n_lines)

        line_ids_labels = {
            line: self.ordered_survey_lines[line] for line in survey_lines
        }

        anomaly_traces = {}
        for key, value in self.property_groups.items():
            anomaly_traces[key] = {
                "x": [None],
                "y": [None],
                "marker_color": value["color"],
                "mode": "markers",
                "name": key,
            }

        marker_x = np.inf
        marker_y = np.inf
        line_dict = {}
        for line in self.computed_lines:  # type: ignore  # pylint: disable=C0206
            line_position = self.computed_lines[line]["position"]
            line_anomalies = self.computed_lines[line]["anomalies"]

            label = line_ids_labels[line]  # type: ignore
            n_parts = len(line_position)

            line_dict[line] = {
                "x": [None],
                "y": [None],
                "name": label,
            }
            if line == selected_line:
                line_dict[line]["line_color"] = "black"

            for ind in range(n_parts):
                position = line_position[ind]
                anomalies = line_anomalies[ind]

                if position is not None and position.locations_resampled is not None:
                    x_locs = position.x_locations
                    y_locs = position.y_locations
                    if line == selected_line and x_locs[0] < marker_x:
                        marker_x = x_locs[0]
                        marker_y = y_locs[0]
                    line_dict[line]["x"] += [None] + list(x_locs)  # type: ignore
                    line_dict[line]["y"] += [None] + list(y_locs)  # type: ignore

                if anomalies is not None:
                    x_min = np.min(position.x_locations)

                    for anom in anomalies:
                        peak = position.locations[anom.peaks[0]]
                        x_val = x_min + peak
                        ind = (np.abs(x_locs - x_val)).argmin()
                        anomaly_traces[anom.property_group.name]["x"].append(
                            x_locs[ind]
                        )
                        anomaly_traces[anom.property_group.name]["y"].append(
                            y_locs[ind]
                        )

        for trace in list(line_dict.values()):
            figure.add_trace(  # type: ignore
                go.Scatter(
                    **trace,
                )
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
            margin={"t": 20, "l": 20, "b": 20, "r": 20},
        )

        return figure

    def trigger_click(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        *,
        n_clicks: int,
        flip_sign: list[bool],
        trend_lines: list[bool],
        masking_data: str | None,
        smoothing: float,
        min_amplitude: float,
        min_value: float,
        min_width: float,
        max_migration: float,
        min_channels: int,
        n_groups: int,
        max_separation: float,
        selected_line: int,
        ga_group_name: str,
    ) -> list[str]:
        """
        Write output ui.json file and workspace, run driver.

        :param n_clicks: Trigger for callback.
        :param flip_sign: Whether to flip the sign of the data.
        :param trend_lines: Whether to export trend lines.
        :param masking_data: Masking data.
        :param smoothing: Smoothing factor.
        :param min_amplitude: Minimum amplitude.
        :param min_value: Minimum value.
        :param min_width: Minimum width.
        :param max_migration: Maximum migration.
        :param min_channels: Minimum number of channels.
        :param n_groups: Number of consecutive peaks to merge.
        :param max_separation: Maximum separation between peaks to merge.
        :param selected_line: Selected line ID.
        :param ga_group_name: Group name.

        :return: Output save message.
        """
        # Update self.params from dash component values
        with fetch_active_workspace(self.params.geoh5) as workspace:
            param_dict = self.get_params_dict(locals())
            param_dict.update(
                {
                    "geoh5": workspace,
                    "objects": self.params.objects,
                    "line_field": self.params.line_field,
                    "monitoring_directory": self.params.monitoring_directory,
                }
            )

            if masking_data == "None":
                param_dict["masking_data"] = None

            if self.property_groups is not None:
                p_g_new = {
                    p_g.name: p_g for p_g in param_dict["objects"].property_groups
                }
                for key, value in self.property_groups.items():
                    param_dict[f"group_{value['param']}_data"] = p_g_new[key]
                    param_dict[f"group_{value['param']}_color"] = value["color"]

        # Write output uijson.
        new_params = PeakFinderParams(**param_dict)
        name = workspace.h5file.stem.replace(".ui", "")
        new_params.write_input_file(
            name=name + ".ui.json",
            path=workspace.h5file.parent,
            validate=False,
        )

        driver = PeakFinderDriver(new_params)
        driver.run()

        return ["Saved to " + str(workspace.h5file)]

    @property
    def params(self) -> PeakFinderParams:
        """
        Application parameters
        """
        return self._params

    @params.setter
    def params(self, params: PeakFinderParams):
        if not isinstance(params, PeakFinderParams):
            raise TypeError(
                f"Input parameters must be an instance of {PeakFinderParams}"
            )

        self._params = params


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
