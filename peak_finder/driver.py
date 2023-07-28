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
from geoapps_utils import geophysical_systems
from geoapps_utils.conversions import hex_to_rgb
from geoapps_utils.driver.driver import BaseDriver
from geoapps_utils.formatters import string_name
from geoh5py.groups import ContainerGroup
from geoh5py.objects import Curve, Points
from geoh5py.shared.utils import fetch_active_workspace
from tqdm import tqdm

from peak_finder.constants import validations
from peak_finder.line_anomaly import LineAnomaly
from peak_finder.params import PeakFinderParams
from peak_finder.utils import default_groups_from_property_group


class PeakFinderDriver(BaseDriver):
    _params_class: PeakFinderParams = PeakFinderParams  # type: ignore
    _validations = validations

    def __init__(self, params: PeakFinderParams):
        super().__init__(params)
        self.params: PeakFinderParams = params

    def run(self):  # pylint: disable=R0912, R0914, R0915 # noqa: C901
        with fetch_active_workspace(self.params.geoh5, mode="r+"):
            survey = self.params.objects
            prop_group = [
                pg for pg in survey.property_groups if pg.uid == self.params.data.uid
            ]

            if self.params.tem_checkbox:
                system = geophysical_systems.parameters()[self.params.system]
                normalization = system["normalization"]
            else:
                normalization = [1]

            output_group = ContainerGroup.create(
                self.params.geoh5, name=string_name(self.params.ga_group_name)
            )

            line_field = self.params.line_field
            lines = np.unique(line_field.values)

            if self.params.group_auto and any(prop_group):
                channel_groups = default_groups_from_property_group(prop_group[0])
            else:
                channel_groups = self.params.groups_from_free_params()

            active_channels = {}
            for group in channel_groups.values():
                for channel in group["properties"]:
                    obj = self.params.geoh5.get_entity(channel)[0]
                    active_channels[channel] = {"name": obj.name}

            for uid, channel_params in active_channels.items():
                obj = self.params.geoh5.get_entity(uid)[0]
                if self.params.tem_checkbox:
                    channel = [ch for ch in system["channels"] if ch in obj.name]
                    if any(channel):
                        channel_params["time"] = system["channels"][channel[0]]
                    else:
                        continue
                channel_params["values"] = (
                    obj.values.copy() * (-1.0) ** self.params.flip_sign
                )

            print("Submitting parallel jobs:")
            property_groups = [
                survey.find_or_create_property_group(name=name)
                for name in channel_groups
            ]

            anomalies = []
            for line_id in tqdm(list(lines)):
                line_indices = np.where(line_field.values == line_id)[0]

                line_computation = delayed(LineAnomaly, pure=True)

                anomalies += [
                    line_computation(
                        entity=survey,
                        line_indices=line_indices,
                        property_groups=property_groups,
                        data_normalization=normalization,
                        smoothing=self.params.smoothing,
                        min_amplitude=self.params.min_amplitude,
                        min_value=self.params.min_value,
                        min_width=self.params.min_width,
                        max_migration=self.params.max_migration,
                        min_channels=self.params.min_channels,
                        minimal_output=True,
                    )
                ]
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

            for line in tqdm(results):
                for line_anomaly in line:
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
                points.add_data(
                    {
                        "amplitude": {"values": np.hstack(amplitude)},
                        "skew": {"values": skew},
                    }
                )

                if self.params.tem_checkbox:
                    points.add_data(
                        {
                            "tau": {"values": np.hstack(tau)},
                            "azimuth": {"values": azimuth},
                            "dip": {"values": dip},
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

                if self.params.tem_checkbox:
                    group = points.find_or_create_property_group(
                        name="AzmDip", property_group_type="Dip direction & dip"
                    )
                    group.properties = [
                        points.get_data("azimuth")[0].uid,
                        points.get_data("dip")[0].uid,
                    ]

                # Add structural markers
                if self.params.structural_markers:
                    if self.params.tem_checkbox:
                        markers = []

                        def rotation_2d(angle):
                            rot = np.r_[
                                np.c_[
                                    np.cos(np.pi * angle / 180),
                                    -np.sin(np.pi * angle / 180),
                                ],
                                np.c_[
                                    np.sin(np.pi * angle / 180),
                                    np.cos(np.pi * angle / 180),
                                ],
                            ]
                            return rot

                        for azm, xyz, mig in zip(
                            np.hstack(azimuth).tolist(),
                            np.vstack(group_center).tolist(),
                            migration.tolist(),
                        ):
                            marker = np.r_[
                                np.c_[-0.5, 0.0] * 50,
                                np.c_[0.5, 0] * 50,
                                np.c_[0.0, 0.0],
                                np.c_[0.0, 1.0] * mig,
                            ]

                            marker = (
                                np.c_[
                                    np.dot(rotation_2d(-azm), marker.T).T, np.zeros(4)
                                ]
                                + xyz
                            )
                            markers.append(marker.squeeze())

                        curves = Curve.create(
                            self.params.geoh5,
                            name="TickMarkers",
                            vertices=np.vstack(markers),
                            cells=np.arange(len(markers) * 4, dtype="uint32").reshape(
                                (-1, 2)
                            ),
                            parent=output_group,
                        )
                        channel_group_data = curves.add_data(
                            {
                                "channel_group": {
                                    "type": "referenced",
                                    "values": np.kron(
                                        np.hstack(channel_group), np.ones(4)
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
