#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app project.
#
#  All rights reserved.
#

# pylint: disable=duplicate-code

from __future__ import annotations
import sys
from typing import cast
import numpy as np
from curve_apps.trend_lines.driver import TrendLinesDriver
from curve_apps.trend_lines.params import TrendLineParameters
from dask import compute, delayed, config
from dask.diagnostics import ProgressBar
from geoapps_utils.driver.driver import BaseDriver
from geoapps_utils.utils.conversions import hex_to_rgb
from geoapps_utils.utils.formatters import string_name
from geoh5py import Workspace
from geoh5py.data import NumericData, ReferencedData
from geoh5py.groups import ContainerGroup, PropertyGroup
from geoh5py.objects import Curve, Points
from geoh5py.shared.utils import fetch_active_workspace
from tqdm import tqdm

from peak_finder.constants import validations
from peak_finder.line_anomaly import LineAnomaly
from peak_finder.params import PeakFinderParams

config.set(scheduler="processes")


@delayed
def line_computation(line_anomaly: LineAnomaly):
    getattr(line_anomaly, "anomalies")
    return line_anomaly


class PeakFinderDriver(BaseDriver):
    _params_class: PeakFinderParams = PeakFinderParams  # type: ignore
    _validations = validations

    def __init__(self, params: PeakFinderParams):
        super().__init__(params)
        self.params: PeakFinderParams = params

    @staticmethod
    def compute_lines(  # pylint: disable=R0913, R0914
        *,
        survey: Curve,
        line_indices_dict: dict[str, dict],
        line_ids: list[int] | np.ndarray,
        property_groups: list[PropertyGroup],
        smoothing: float,
        min_amplitude: float,
        min_value: float,
        min_width: float,
        max_migration: float,
        min_channels: int,
        n_groups: int,
        max_separation: float,
        parallelized: bool = True,
    ) -> list[LineAnomaly]:
        """
        Compute anomalies for a list of line ids.

        :param survey: Survey object.
        :param line_indices_dict: Dict of line indices.
        :param line_ids: List of line ids.
        :param property_groups: Property groups to use for grouping anomalies.
        :param smoothing: Smoothing factor.
        :param min_amplitude: Minimum amplitude of anomaly as percent.
        :param min_value: Minimum data value of anomaly.
        :param min_width: Minimum width of anomaly in meters.
        :param max_migration: Maximum peak migration.
        :param min_channels: Minimum number of channels in anomaly.
        :param n_groups: Number of groups to use for grouping anomalies.
        :param max_separation: Maximum separation between anomalies in meters.
        """

        groups_channels = {}
        for group in property_groups:
            channels = {}
            if group.properties is None:
                continue

            for prop in group.properties:
                survey_data = survey.get_entity(prop)[0]
                if (
                    not isinstance(survey_data, NumericData)
                    or survey_data.values is None
                ):
                    continue

                channels[survey_data.uid] = survey_data.values

            groups_channels[group.name] = channels

        anomalies = []
        for line_id in tqdm(list(line_ids)):
            line_start = line_indices_dict[line_id]["line_start"]
            for indices in line_indices_dict[line_id]["line_indices"]:
                line_class = LineAnomaly(
                    vertices=survey.vertices,
                    cells=survey.cells,
                    line_id=line_id,
                    line_indices=indices,
                    line_start=line_start,
                    property_groups=groups_channels,
                    smoothing=smoothing,
                    min_amplitude=min_amplitude,
                    min_value=min_value,
                    min_width=min_width,
                    max_migration=max_migration,
                    min_channels=min_channels,
                    n_groups=n_groups,
                    max_separation=max_separation,
                    minimal_output=True,
                )  # type: ignore

                if parallelized:
                    anomalies += [line_computation(line_class)]
                else:
                    anomalies += [line_class]

        return anomalies

    @staticmethod
    def get_line_indices(  # pylint: disable=too-many-locals
        survey_obj: Curve,
        line_field_obj: ReferencedData,
        line_ids: list[int],
    ) -> dict:
        """
        Get line indices for plotting.

        :param survey_obj: Survey object.
        :param line_field_obj: Line field object.
        :param line_ids: Line IDs.

        :return: Line indices for each line ID given.
        """
        if (
            not isinstance(survey_obj, Curve)
            or survey_obj.vertices is None
            or line_field_obj.values is None
        ):
            return {}

        line_length = len(line_field_obj.values)

        indices_dict: dict = {}
        for line_id in line_ids:
            line_bool = line_field_obj.values == line_id
            full_line_indices = np.where(line_bool)[0]

            indices_dict[line_id] = {"line_indices": []}

            parts = np.unique(survey_obj.parts[full_line_indices])

            for part in parts:
                active_indices = np.where(
                    (line_field_obj.values == line_id) & (survey_obj.parts == part)
                )[0]

                line_indices = np.zeros(line_length, dtype=bool)
                line_indices[active_indices] = True

                indices_dict[line_id]["line_indices"].append(line_indices)

        # Just on masked parts of line
        for line_id, indices in indices_dict.items():
            # Get line start
            line_start = None
            if len(indices["line_indices"]) > 0:
                locs = survey_obj.vertices
                line_segment = np.any(indices["line_indices"], axis=0)

                if isinstance(locs, np.ndarray) and locs.shape[1] > 1:
                    if np.std(locs[line_segment][:, 1]) > np.std(
                        locs[line_segment][:, 0]
                    ):
                        line_start = np.argmin(locs[line_segment, 1])
                        line_start = locs[line_segment][line_start]
                    else:
                        line_start = np.argmin(locs[line_segment, 0])
                        line_start = locs[line_segment][line_start]
            indices["line_start"] = line_start

        return indices_dict

    def run(self):  # pylint: disable=R0912, R0914, too-many-statements
        with fetch_active_workspace(self.params.geoh5, mode="r+"):
            survey = self.params.objects

            if survey is None:
                raise ValueError("Survey object not found.")

            output_group = ContainerGroup.create(
                self.params.geoh5, name=string_name(self.params.ga_group_name)
            )

            channel_groups = self.params.get_property_groups()
            # Create reference values and color_map
            group_map, color_map = {0: "Unknown"}, [[0, 0, 0, 0, 0]]
            for ind, (name, group) in enumerate(channel_groups.items()):
                group_map[ind + 1] = name
                color_map += [[ind + 1] + hex_to_rgb(group["color"]) + [0]]

            active_channels = {}
            for group in channel_groups.values():
                for channel in group["properties"]:
                    obj = self.params.geoh5.get_entity(channel)[0]
                    active_channels[channel] = {"name": obj.name}

            for uid, channel_params in active_channels.items():
                obj = self.params.geoh5.get_entity(uid)[0]
                values = obj.values

                if values is None:
                    continue

                channel_params["values"] = (
                    values.copy() * (-1.0) ** self.params.flip_sign
                )

            print("Submitting parallel jobs:")
            property_groups = [
                survey.fetch_property_group(name=name) for name in channel_groups
            ]

            if self.params.masking_data is not None:
                masking_array = self.params.masking_data.values

                workspace = Workspace()
                survey = cast(Curve, survey.copy(parent=workspace))

                if False in masking_array:
                    survey.remove_vertices(~masking_array)

                if self.params.line_field is not None:
                    new_line_id = survey.get_entity(self.params.line_field.uid)[0]
                else:
                    new_line_id = self.params.get_line_field(survey)

                if isinstance(new_line_id, ReferencedData):
                    self.params.line_field = new_line_id

            line_field_obj = self.params.get_line_field(survey)

            if (
                not isinstance(line_field_obj, ReferencedData)
                or line_field_obj.value_map is None
            ):
                raise ValueError("Line field not found.")

            line_ids = list(line_field_obj.value_map().keys())
            indices_dict = PeakFinderDriver.get_line_indices(
                survey, line_field_obj, line_ids
            )
            anomalies = PeakFinderDriver.compute_lines(
                survey=survey,
                line_indices_dict=indices_dict,
                line_ids=line_ids,
                property_groups=property_groups,
                smoothing=self.params.smoothing,
                min_amplitude=self.params.min_amplitude,
                min_value=self.params.min_value,
                min_width=self.params.min_width,
                max_migration=self.params.max_migration,
                min_channels=self.params.min_channels,
                n_groups=self.params.n_groups,
                max_separation=self.params.max_separation,
            )

            (
                channel_groups,
                amplitudes,
                centers,
                starts,
                ends,
            ) = ([], [], [], [], [])

            (
                anom_locs,
                inflect_up,
                inflect_down,
                anom_start,
                anom_end,
                peaks,
                line_ids,
            ) = ([], [], [], [], [], [], [])

            print("Processing and collecting results:")

            with ProgressBar():
                results = compute(anomalies)[0]
            # pylint: disable=R1702
            for line_anomaly in tqdm(results):
                if line_anomaly.anomalies is None or line_anomaly.centers is None:
                    continue

                centers.append(line_anomaly.centers)
                amplitudes.append(line_anomaly.amplitudes)
                starts.append(line_anomaly.starts)
                ends.append(line_anomaly.ends)
                line_ids.append(
                    np.ones(len(line_anomaly.centers)) * line_anomaly.line_id
                )
                channel_groups.append(line_anomaly.group_ids)

                if self.params.structural_markers:
                    for line_group in line_anomaly.anomalies:
                        for group in line_group.groups:
                            locs = np.vstack(
                                [
                                    getattr(group.position, f"{k}_locations")
                                    for k in "xyz"
                                ]
                            ).T
                            inds_map = group.position.map_locations

                            for anom in group.anomalies:
                                anom_locs.append(locs[inds_map[anom.peak]])
                                inflect_down.append(inds_map[anom.inflect_down])
                                inflect_up.append(inds_map[anom.inflect_up])
                                anom_start.append(inds_map[anom.start])
                                anom_end.append(inds_map[anom.end])
                                peaks.append(inds_map[anom.peak])

            print("Exporting . . .")
            group_points = None
            if centers:
                group_points = Points.create(
                    self.params.geoh5,
                    name="Anomaly Groups",
                    vertices=np.vstack(centers),
                    parent=output_group,
                )

                group_points.entity_type.name = self.params.ga_group_name
                group_points.add_data(
                    {
                        "amplitude": {"values": np.hstack(amplitudes)},
                        "start": {"values": np.hstack(starts).astype(np.int32)},
                        "end": {"values": np.hstack(ends).astype(np.int32)},
                    }
                )
                channel_group_data = group_points.add_data(
                    {
                        "channel_group": {
                            "type": "referenced",
                            "values": np.hstack(channel_groups) + 1,
                            "value_map": group_map,
                        }
                    }
                )
                channel_group_data.entity_type.color_map = np.vstack(color_map)
                line_id_data = group_points.add_data(
                    {
                        line_field_obj.name: {
                            "values": np.hstack(line_ids).astype(np.int32),
                            "entity_type": line_field_obj.entity_type,
                        }
                    }
                )

                if self.params.trend_lines:
                    inputs = {
                        "geoh5": self.params.geoh5,
                        "entity": group_points,
                        "data": channel_group_data,
                        "parts": line_id_data,
                        "export_as": "Trend Lines",
                        "damping": 1,
                    }

                    params = TrendLineParameters.build(inputs)
                    driver = TrendLinesDriver(params)
                    out_trend = driver.make_curve()

                    if out_trend is not None:
                        driver.add_ui_json(out_trend)

            if self.params.structural_markers and any(anom_locs):
                anom_points = Points.create(
                    self.params.geoh5,
                    name="Anomalies",
                    vertices=np.vstack(anom_locs),
                    parent=output_group,
                )
                anom_points.add_data(
                    {
                        "start": {
                            "values": np.vstack(anom_start).flatten().astype(np.int32)
                        },
                        "end": {
                            "values": np.vstack(anom_end).flatten().astype(np.int32)
                        },
                        "upward inflection": {
                            "values": np.vstack(inflect_up).flatten().astype(np.int32)
                        },
                        "downward inflection": {
                            "values": np.vstack(inflect_down).flatten().astype(np.int32)
                        },
                    }
                )

            self.update_monitoring_directory(output_group)

    @property
    def params(self) -> PeakFinderParams:
        """Application parameters."""
        return self._params

    @params.setter
    def params(self, val: PeakFinderParams):
        if not isinstance(val, PeakFinderParams):
            raise TypeError("Parameters must be of type BaseParams.")
        self._params = val


if __name__ == "__main__":
    FILE = sys.argv[1]
    PeakFinderDriver.start(FILE)
