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

from peak_finder.base.utils import running_mean
from peak_finder.line_data import LineData
from peak_finder.line_group import LineGroup
from peak_finder.line_position import LinePosition


class LineAnomaly:  # pylint: disable=R0902
    """
    Contains list of LineGroup objects.
    """

    def __init__(  # pylint: disable=R0913
        self,
        entity: Curve,
        line_indices: list[int],
        smoothing: int = 1,
        data_normalization: tuple | list | str = (1.0,),
        min_amplitude: int = 25,
        min_value: float = -np.inf,
        min_width: float = 200.0,
        max_migration: float = 50.0,
        min_channels: int = 3,
        use_residual: bool = False,
        minimal_output: bool = False,
        return_profile: bool = False,
    ):
        """
        :param entity: Survey object.
        :param line_indices: Indices of vertices for line profile.
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
        """
        self._entity = entity
        self._line_indices = line_indices
        self._smoothing = smoothing
        self._data_normalization = data_normalization
        self._min_amplitude = min_amplitude
        self._min_value = min_value
        self._min_width = min_width
        self._max_migration = max_migration
        self._min_channels = min_channels
        self._use_residual = use_residual
        self._minimal_output = minimal_output
        self._return_profile = return_profile

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
    def line_indices(self) -> list[int]:
        """ """
        return self._line_indices

    @line_indices.setter
    def line_indices(self, value):
        self._line_indices = value

    @property
    def smoothing(self) -> int:
        """ """
        return self._smoothing

    @smoothing.setter
    def smoothing(self, value):
        self._smoothing = value

    @property
    def data_normalization(self) -> tuple | list | str:
        """ """
        return self._data_normalization

    @data_normalization.setter
    def data_normalization(self, value):
        self._data_normalization = value

    @property
    def min_amplitude(self) -> int:
        """ """
        return self._min_amplitude

    @min_amplitude.setter
    def min_amplitude(self, value):
        self._min_amplitude = value

    @property
    def min_value(self) -> float:
        """ """
        return self._min_value

    @min_value.setter
    def min_value(self, value):
        self._min_value = value

    @property
    def min_width(self) -> float:
        """ """
        return self._min_width

    @min_width.setter
    def min_width(self, value):
        self._min_width = value

    @property
    def max_migration(self) -> float:
        """ """
        return self._max_migration

    @max_migration.setter
    def max_migration(self, value):
        self._max_migration = value

    @property
    def min_channels(self) -> int:
        """ """
        return self._min_channels

    @min_channels.setter
    def min_channels(self, value):
        self._min_channels = value

    @property
    def use_residual(self) -> bool:
        """ """
        return self._use_residual

    @use_residual.setter
    def use_residual(self, value):
        self._use_residual = value

    @property
    def minimal_output(self) -> bool:
        """ """
        return self._minimal_output

    @minimal_output.setter
    def minimal_output(self, value):
        self._minimal_output = value

    @property
    def return_profile(self) -> bool:
        """ """
        return self._return_profile

    @return_profile.setter
    def return_profile(self, value):
        self._return_profile = value

    @property
    def locations(self) -> np.ndarray | None:
        """
        Survey vertices.
        """
        return self._locations

    @locations.setter
    def locations(self, value):
        self._locations = value
        self.position = LinePosition(
            locations=self.locations[self.line_indices],
            smoothing=self.smoothing,
            residual=self.use_residual,
        )

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

    def compute_azimuth(self):
        """ """
        # Compute azimuth
        locs = self.position.locations_resampled
        mat = np.c_[self.position.interp_x(locs), self.position.interp_y(locs)]
        angles = np.arctan2(mat[1:, 1] - mat[:-1, 1], mat[1:, 0] - mat[:-1, 0])
        angles = np.r_[angles[0], angles].tolist()
        azimuth = (450.0 - np.rad2deg(running_mean(angles, width=5))) % 360.0
        return azimuth

    def find_anomalies(  # pylint: disable=R0914
        self,
        channels: dict,
        channel_groups: dict,
    ) -> tuple[LineGroup | None, LinePosition] | LineGroup | None:
        """
        Find all anomalies along a line profile of data.
        Anomalies are detected based on the lows, inflection points and peaks.
        Neighbouring anomalies are then grouped and assigned a channel_group label.

        :param channels: Channels.
        :param channel_groups: Property groups to use for grouping anomalies.

        :return: List of groups and line profile.
        """
        if self.locations is None or self.position is None:
            return None

        locs = self.position.locations_resampled

        if locs is None:
            return None

        if self.data_normalization == "ppm":
            self.data_normalization = [1e-6]

        azimuth = self.compute_azimuth()

        line_dataset = []

        # Used in group_anomalies
        data_uid = list(channels)
        property_groups = list(channel_groups.values())
        group_prop_size = np.r_[
            [len(grp["properties"]) for grp in channel_groups.values()]
        ]

        # Iterate over channels and add to anomalies
        for channel, (uid, params) in enumerate(channels.items()):
            line_data = LineData(
                channel,
                uid,
                self.position,
                self.min_amplitude,
                self.min_width,
                self.max_migration,
            )

            if "values" not in list(params):
                continue

            # Update profile with current line values
            values = params["values"][self.line_indices].copy()
            self.position.values = values
            values = self.position.values_resampled

            # Get indices for peaks and inflection points for line
            peaks, lows, inflect_up, inflect_down = self.position.get_peak_indices(
                values, self.min_value
            )
            # Iterate over peaks and add to anomalies
            line_data.add_anomalies(
                values,
                channel_groups,
                peaks,
                lows,
                inflect_up,
                inflect_down,
                locs,
            )
            line_data.anomalies = np.array(line_data.anomalies)
            line_dataset.append(line_data)

        if len(line_dataset) == 0:
            line_group = None
        else:
            # Group anomalies
            line_group = LineGroup(
                line_dataset=line_dataset,
                max_migration=self.max_migration,
            )
            line_group.group_anomalies(
                channel_groups,
                group_prop_size,
                self.min_channels,
                property_groups,
                data_uid,
                azimuth,
                self.data_normalization,
                channels,
                self.minimal_output,
            )

        if self.return_profile:
            return line_group, self.position
        return line_group
