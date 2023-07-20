#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app project.
#
#  All rights reserved.
#

from __future__ import annotations

import numpy as np
from geoh5py.groups import PropertyGroup
from geoh5py.objects import Curve

from peak_finder.anomaly_group import AnomalyGroup
from peak_finder.base.utils import running_mean
from peak_finder.line_data import LineData
from peak_finder.line_position import LinePosition


class LineGroup:
    """
    Contains list of AnomalyGroup objects.
    """

    def __init__(self, dataset: LineData, groups: list[AnomalyGroup]):
        """
        :param dataset: LineData with list of all anomalies.
        :param groups: List of anomaly groups.
        """
        self._groups = groups
        self._dataset = dataset

    @property
    def groups(self) -> list[AnomalyGroup]:
        """
        List of anomaly groups.
        """
        return self._groups

    @groups.setter
    def groups(self, value):
        self._groups = value

    @property
    def dataset(self) -> LineData:
        """
        All anomalies.
        """
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value


class LineAnomaly:
    """
    Contains list of LineGroup objects.
    """

    def __init__(
        self,
        entity: Curve,
    ):
        """
        :param entity: Survey object.
        """
        self._entity = entity
        self._property_group: PropertyGroup | None = None
        self._position: LinePosition | None = None
        self._anomalies: list[LineGroup] = []

        self.locations = self.entity.vertices

    @property
    def entity(self) -> Curve:
        """
        Survey object.
        """
        return self._entity

    @entity.setter
    def entity(self, value):
        self._entity = value

    @property
    def locations(self) -> np.ndarray | None:
        """
        Survey vertices.
        """
        return self._locations

    @locations.setter
    def locations(self, value):
        self._locations = value

    @property
    def anomalies(self) -> list[LineGroup] | None:
        """
        List of line groups.
        """
        return self._anomalies

    @anomalies.setter
    def anomalies(self, value):
        self._anomalies = value

    @property
    def position(self) -> LinePosition | None:
        """
        Line position and interpolation.
        """
        return self._position

    @position.setter
    def position(self, value):
        self._position = value

    def find_anomalies(
        self,
        line_indices: np.ndarray,
        channels: dict,
        channel_groups: dict,
        smoothing: int,
        data_normalization: list | str,
        min_amplitude: int,
        min_value: float,
        min_width: float,
        max_migration: float,
        min_channels: int,
        use_residual: bool = False,
        minimal_output: bool = False,
        return_profile: bool = False,
    ) -> tuple[LineGroup | None, LinePosition] | LineGroup | None:
        """
        Find all anomalies along a line profile of data.
        Anomalies are detected based on the lows, inflection points and peaks.
        Neighbouring anomalies are then grouped and assigned a channel_group label.

        :param line_indices: Indices of vertices for line profile.
        :param channels: Channels.
        :param channel_groups: Property groups to use for grouping anomalies.
        :param smoothing: Smoothing factor.
        :param data_normalization: Value(s) to normalize data by.
        :param min_amplitude: Minimum amplitude of anomaly as percent.
        :param min_value: Minimum data value of anomaly.
        :param min_width: Minimum width of anomaly in meters.
        :param max_migration: Maximum peak migration.
        :param min_channels: Minimum number of channels in anomaly.
        :param use_residual: Whether to use the residual of the smoothing data.
        :param minimal_output: Whether to return minimal output.
        :param return_profile: Whether to return the line profile.

        :return: List of groups and line profile.
        """
        if self.locations is None:
            return None

        self.position = LinePosition(
            locations=self.locations[line_indices],
            smoothing=smoothing,
            residual=use_residual,
        )
        locs = self.position.locations_resampled
        if locs is None:
            return None

        if data_normalization == "ppm":
            data_normalization = [1e-6]

        # Compute azimuth
        mat = np.c_[self.position.interp_x(locs), self.position.interp_y(locs)]
        angles = np.arctan2(mat[1:, 1] - mat[:-1, 1], mat[1:, 0] - mat[:-1, 0])
        angles = np.r_[angles[0], angles].tolist()
        azimuth = (450.0 - np.rad2deg(running_mean(angles, width=5))) % 360.0

        line_data = LineData(self.position, min_amplitude, min_width, max_migration)

        # Used in group_anomalies
        data_uid = list(channels)
        property_groups = list(channel_groups.values())
        group_prop_size = np.r_[
            [len(grp["properties"]) for grp in channel_groups.values()]
        ]

        # Iterate over channels and add to anomalies
        for channel, (uid, params) in enumerate(channels.items()):
            if "values" not in list(params):
                continue
            # Update profile with current line values
            # self.position.values = params["values"][line_indices].copy()
            values = params["values"][line_indices].copy()
            self.position.values = values

            # Get indices for peaks and inflection points for line
            peaks, lows, inflect_up, inflect_down = self.position.get_peak_indices(
                values, min_value
            )
            # Iterate over peaks and add to anomalies
            line_data.iterate_over_peaks(
                values,
                channel,
                channel_groups,
                uid,
                peaks,
                lows,
                inflect_up,
                inflect_down,
                locs,
            )
        line_data.anomalies = np.array(line_data.anomalies)
        if len(line_data.anomalies) == 0:
            line_group = None
        else:
            # Group anomalies
            groups = line_data.group_anomalies(
                channel_groups,
                group_prop_size,
                min_channels,
                property_groups,
                data_uid,
                azimuth,
                data_normalization,
                channels,
                minimal_output,
            )
            line_group = LineGroup(line_data, groups)

        if return_profile:
            return line_group, self.position
        return line_group
