#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app project.
#
#  All rights reserved.
#


from __future__ import annotations

import sys

import numpy as np
from dask import compute, delayed
from dask.diagnostics import ProgressBar
from geoapps_utils.conversions import hex_to_rgb
from geoapps_utils.driver.driver import BaseDriver
from geoapps_utils.formatters import string_name
from geoh5py import Workspace
from geoh5py.data import ReferencedData
from geoh5py.groups import ContainerGroup, PropertyGroup
from geoh5py.objects import Curve, Points
from geoh5py.shared.utils import fetch_active_workspace
from tqdm import tqdm

from peak_finder.constants import validations
from peak_finder.line_anomaly import LineAnomaly
from peak_finder.params import PeakFinderParams


class PeakFinderDriver(BaseDriver):
    _params_class: PeakFinderParams = PeakFinderParams  # type: ignore
    _validations = validations

    def __init__(self, params: PeakFinderParams):
        super().__init__(params)
        self.params: PeakFinderParams = params

    @staticmethod
    def get_line_indices(  # pylint: disable=R0914
        survey: Curve,
        line_field: ReferencedData,
        line_ids: list[int],
    ) -> dict:
        """
        Get line indices for a list of line ids.

        :param survey: Survey object.
        :param line_field: Line field.
        :param line_ids: List of line ids.

        :return: Dictionary of line indices with line IDs as keys.
        """
        indices_dict: dict[str, np.ndarray] = {}
        for line_id in line_ids:
            indices_dict[str(line_id)] = []

            line_bool = line_field.values == line_id
            full_line_indices = np.where(line_bool)[0]
            if len(full_line_indices) < 2:
                continue

            parts = np.unique(survey.parts[full_line_indices])

            for part in parts:
                line_indices = np.where(
                    (line_field.values == line_id) & (survey.parts == part)
                )[0]

                indices_dict[str(line_id)] += [line_indices]

        return indices_dict

    @staticmethod
    def compute_lines(  # pylint: disable=R0913, R0914
        survey: Curve,
        line_indices: list[int] | np.ndarray,
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
    ) -> list[LineAnomaly]:
        """
        Compute anomalies for a list of line ids.

        :param survey: Survey object.
        :param line_indices: List of line indices.
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
        line_computation = delayed(LineAnomaly, pure=True)

        anomalies = []
        for line_id in tqdm(list(line_ids)):
            for indices in line_indices[str(line_id)]:  # type: ignore
                masking_offset = np.min(indices)
                anomalies += [
                    line_computation(
                        entity=survey,
                        line_id=line_id,
                        line_indices=indices,
                        property_groups=property_groups,
                        smoothing=smoothing,
                        min_amplitude=min_amplitude,
                        min_value=min_value,
                        min_width=min_width,
                        max_migration=max_migration,
                        min_channels=min_channels,
                        n_groups=n_groups,
                        max_separation=max_separation,
                        minimal_output=True,
                        masking_offset=masking_offset,
                    )
                ]
        return anomalies

    def run(self):  # pylint: disable=R0912, R0914
        with fetch_active_workspace(self.params.geoh5, mode="r+"):
            survey = self.params.objects

            output_group = ContainerGroup.create(
                self.params.geoh5, name=string_name(self.params.ga_group_name)
            )

            channel_groups = self.params.get_property_groups()

            active_channels = {}
            for group in channel_groups.values():
                for channel in group["properties"]:
                    obj = self.params.geoh5.get_entity(channel)[0]
                    active_channels[channel] = {"name": obj.name}

            for uid, channel_params in active_channels.items():
                obj = self.params.geoh5.get_entity(uid)[0]
                channel_params["values"] = (
                    obj.values.copy() * (-1.0) ** self.params.flip_sign
                )

            print("Submitting parallel jobs:")
            property_groups = [
                survey.find_or_create_property_group(name=name)
                for name in channel_groups
            ]

            survey_obj = self.params.objects
            if self.params.masking_data is not None:
                masking_array = self.params.masking_data.values

                workspace = Workspace()
                survey_obj = survey_obj.copy(parent=workspace)
                survey_obj.remove_vertices(~masking_array)
                line_field_obj = survey_obj.get_data(self.params.line_field.uid)[0]
            else:
                line_field_obj = self.params.line_field

            line_indices = self.get_line_indices(
                survey=survey_obj,
                line_field=line_field_obj,
                line_ids=[self.params.line_id],
            )
            anomalies = PeakFinderDriver.compute_lines(
                survey=survey_obj,
                line_indices=line_indices,
                line_ids=[self.params.line_id],
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
                channel_group,
                tau,
                migration,
                azimuth,
                group_center,
                amplitude,
                inflect_up,
                inflect_down,
                start,
                end,
                skew,
                peaks,
            ) = ([], [], [], [], [], [], [], [], [], [], [], [])

            print("Processing and collecting results:")
            with ProgressBar():
                results = compute(anomalies)

            for line in tqdm(results):  # pylint: disable=R1702
                for line_anomaly in line:
                    if line_anomaly.anomalies is None:
                        continue
                    for line_group in line_anomaly.anomalies:
                        for group in line_group.groups:
                            if group.linear_fit is None:
                                tau += [0]
                            else:
                                tau += [np.abs(group.linear_fit[0] ** -1.0)]
                            channel_group.append(
                                property_groups.index(group.property_group) + 1
                            )
                            migration.append(group.migration)
                            amplitude.append(group.amplitude)
                            azimuth.append(group.azimuth)
                            skew.append(group.skew)
                            group_center.append(group.group_center)
                            for anom in group.anomalies:
                                inflect_down.append(anom.inflect_down)
                                inflect_up.append(anom.inflect_up)
                                start.append(anom.start)
                                end.append(anom.end)
                                peaks.append(anom.peak)

            print("Exporting . . .")
            if group_center:
                channel_group = np.hstack(channel_group)  # Start count at 1

                # Create reference values and color_map
                group_map, color_map = {}, []
                for ind, (name, group) in enumerate(channel_groups.items()):
                    group_map[ind + 1] = name
                    color_map += [[ind + 1] + hex_to_rgb(group["color"]) + [1]]

                color_map = np.core.records.fromarrays(
                    np.vstack(color_map).T,
                    names=["Value", "Red", "Green", "Blue", "Alpha"],
                )
                points = Points.create(
                    self.params.geoh5,
                    name="PointMarkers",
                    vertices=np.vstack(group_center),
                    parent=output_group,
                )
                points.entity_type.name = self.params.ga_group_name
                migration = np.hstack(migration)
                dip = migration / migration.max()
                dip = np.rad2deg(np.arccos(dip))
                skew = np.hstack(skew)
                azimuth = np.hstack(azimuth)
                amplitude = np.hstack(amplitude)
                points.add_data(
                    {
                        "amplitude": {"values": amplitude},
                        "skew": {"values": skew},
                    }
                )

                channel_group_data = points.add_data(
                    {
                        "channel_group": {
                            "type": "referenced",
                            "values": np.hstack(channel_group),
                            "value_map": group_map,
                        }
                    }
                )
                channel_group_data.entity_type.color_map = {
                    "name": "Time Groups",
                    "values": color_map,
                }

                # Add structural markers
                if self.params.structural_markers:
                    inflect_pts = Points.create(
                        self.params.geoh5,
                        name="Inflections_Up",
                        vertices=np.vstack(inflect_up),
                        parent=output_group,
                    )
                    channel_group_data = inflect_pts.add_data(
                        {
                            "channel_group": {
                                "type": "referenced",
                                "values": np.repeat(
                                    np.hstack(channel_group),
                                    [i.shape[0] for i in inflect_up],
                                ),
                                "value_map": group_map,
                            }
                        }
                    )
                    channel_group_data.entity_type.color_map = {
                        "name": "Time Groups",
                        "values": color_map,
                    }
                    inflect_pts = Points.create(
                        self.params.geoh5,
                        name="Inflections_Down",
                        vertices=np.vstack(inflect_down),
                        parent=output_group,
                    )
                    channel_group_data.copy(parent=inflect_pts)

                    start_pts = Points.create(
                        self.params.geoh5,
                        name="Starts",
                        vertices=np.vstack(start),
                        parent=output_group,
                    )
                    channel_group_data.copy(parent=start_pts)

                    end_pts = Points.create(
                        self.params.geoh5,
                        name="Ends",
                        vertices=np.vstack(end),
                        parent=output_group,
                    )
                    channel_group_data.copy(parent=end_pts)

                    Points.create(
                        self.params.geoh5,
                        name="Peaks",
                        vertices=np.vstack(peaks),
                        parent=output_group,
                    )
            self.update_monitoring_directory(output_group)


if __name__ == "__main__":
    FILE = sys.argv[1]
    PeakFinderDriver.start(FILE)
