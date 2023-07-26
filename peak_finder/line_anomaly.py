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
        channels: dict,
        channel_groups: dict,
        smoothing: int = 1,
        data_normalization: tuple | list | str = (1.0,),
        min_amplitude: int = 25,
        min_value: float = -np.inf,
        min_width: float = 200.0,
        max_migration: float = 50.0,
        min_channels: int = 3,
        use_residual: bool = False,
        minimal_output: bool = False,
    ):
        """
        :param entity: Survey object.
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
        """
        self._entity = entity
        self._line_indices = line_indices
        self._channels = channels
        self._channel_groups = channel_groups
        self._smoothing = smoothing
        self._data_normalization = data_normalization
        self._min_amplitude = min_amplitude
        self._min_value = min_value
        self._min_width = min_width
        self._max_migration = max_migration
        self._min_channels = min_channels
        self._use_residual = use_residual
        self._minimal_output = minimal_output

        self._property_group: PropertyGroup | None = None
        self._position: LinePosition | None = None
        self._anomalies: list[LineGroup] | None = None
        self._locations: np.ndarray | None = None

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
        """
        Indices of vertices for line profile.
        """
        return self._line_indices

    @line_indices.setter
    def line_indices(self, value):
        self._line_indices = value

    @property
    def channels(self) -> dict:
        """
        Dict of active channels, uids and values.
        """
        return self._channels

    @channels.setter
    def channels(self, value):
        self._channels = value

    @property
    def channel_groups(self) -> dict:
        """
        Dict of property groups.
        """
        return self._channel_groups

    @channel_groups.setter
    def channel_groups(self, value):
        self._channel_groups = value

    @property
    def smoothing(self) -> int:
        """
        Smoothing factor.
        """
        return self._smoothing

    @smoothing.setter
    def smoothing(self, value):
        self._smoothing = value

    @property
    def data_normalization(self) -> tuple | list | str:
        """
        Value(s) to normalize data by.
        """
        return self._data_normalization

    @data_normalization.setter
    def data_normalization(self, value):
        self._data_normalization = value

    @property
    def min_amplitude(self) -> int:
        """
        Minimum amplitude of anomaly as percent.
        """
        return self._min_amplitude

    @min_amplitude.setter
    def min_amplitude(self, value):
        self._min_amplitude = value

    @property
    def min_value(self) -> float:
        """
        Minimum data value of anomaly.
        """
        return self._min_value

    @min_value.setter
    def min_value(self, value):
        self._min_value = value

    @property
    def min_width(self) -> float:
        """
        Minimum width of anomaly.
        """
        return self._min_width

    @min_width.setter
    def min_width(self, value):
        self._min_width = value

    @property
    def max_migration(self) -> float:
        """
        Max migration for anomaly group.
        """
        return self._max_migration

    @max_migration.setter
    def max_migration(self, value):
        self._max_migration = value

    @property
    def min_channels(self) -> int:
        """
        Minimum number of channels in anomaly group.
        """
        return self._min_channels

    @min_channels.setter
    def min_channels(self, value):
        self._min_channels = value

    @property
    def use_residual(self) -> bool:
        """
        Whether to use the residual of the smoothing data.
        """
        return self._use_residual

    @use_residual.setter
    def use_residual(self, value):
        self._use_residual = value

    @property
    def minimal_output(self) -> bool:
        """
        Whether to return minimal output for anomaly groups.
        """
        return self._minimal_output

    @minimal_output.setter
    def minimal_output(self, value):
        self._minimal_output = value

    @property
    def locations(self) -> np.ndarray | None:
        """
        Survey vertices.
        """
        if self._locations is None:
            self._locations = self.entity.vertices
        return self._locations

    @property
    def anomalies(self) -> list[LineGroup] | None:
        """
        List of line groups.
        """
        if self._anomalies is None:
            self._anomalies = self.find_anomalies()
        return self._anomalies

    @property
    def position(self) -> LinePosition | None:
        """
        Line position and interpolation.
        """
        if self._position is None:
            self._position = LinePosition(
                locations=self.locations[self.line_indices],
                smoothing=self.smoothing,
                residual=self.use_residual,
            )
        return self._position

    def find_anomalies(  # pylint: disable=R0914
        self,
    ) -> list[LineGroup] | None:
        """
        Find all anomalies along a line profile of data.
        Anomalies are detected based on the lows, inflection points and peaks.
        Neighbouring anomalies are then grouped and assigned a channel_group label.

        :return: List of groups and line profile.
        """
        if self.locations is None or self.position is None:
            return None

        locs = self.position.locations_resampled

        if locs is None:
            return None

        if self.data_normalization == "ppm":
            self.data_normalization = [1e-6]

        line_dataset = []
        # Iterate over channels and add to anomalies
        for channel, (uid, params) in enumerate(self.channels.items()):
            if "values" not in list(params):
                continue

            # Make LineData with current channel values
            values = params["values"][self.line_indices].copy()
            line_data = LineData(
                values,
                channel,
                uid,
                self.position,
                self.min_amplitude,
                self.min_width,
                self.max_migration,
                self.min_value,
            )

            line_dataset.append(line_data)

        if len(line_dataset) == 0:
            return None

        # Group anomalies
        line_groups = []
        for property_group in list(self.channel_groups.values()):
            line_group = LineGroup(
                position=self.position,
                line_dataset=line_dataset,
                property_group=property_group,
                max_migration=self.max_migration,
                channel_groups=self.channel_groups,
                min_channels=self.min_channels,
                data_normalization=self.data_normalization,
                channels=self.channels,
                minimal_output=self.minimal_output,
            )
            line_groups.append(line_group)

        return line_groups
