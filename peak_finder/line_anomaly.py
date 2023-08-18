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
    Main class for finding anomalies.

    Contains list of LineGroup objects.
    """

    _entity: Curve
    _line_indices: list[int] | np.ndarray
    _max_migration: float
    _minimal_output: bool
    _min_amplitude: int
    _min_channels: int
    _min_value: float
    _min_width: float
    _property_groups: list[PropertyGroup]
    _smoothing: int
    _use_residual: bool

    def __init__(  # pylint: disable=R0913
        self,
        entity,
        line_indices,
        property_groups,
        max_migration=50.0,
        minimal_output=False,
        min_amplitude=25,
        min_channels=3,
        min_value=-np.inf,
        min_width=200.0,
        smoothing=1,
        use_residual=False,
    ):
        """
        :param entity: Survey object.
        :param line_indices: Indices of vertices for line profile.
        :param property_groups: Property groups to use for grouping anomalies.
        :param smoothing: Smoothing factor.
        :param min_amplitude: Minimum amplitude of anomaly as percent.
        :param min_value: Minimum data value of anomaly.
        :param min_width: Minimum width of anomaly in meters.
        :param max_migration: Maximum peak migration.
        :param min_channels: Minimum number of channels in anomaly.
        :param use_residual: Whether to use the residual of the smoothing data.
        :param minimal_output: Whether to return minimal output.
        """
        self._position: LinePosition | None = None
        self._anomalies: list[LineGroup] | None = None
        self._locations: np.ndarray | None = None

        self.entity = entity
        self.line_indices = line_indices
        self.smoothing = smoothing
        self.min_amplitude = min_amplitude
        self.min_value = min_value
        self.min_width = min_width
        self.max_migration = max_migration
        self.min_channels = min_channels
        self.use_residual = use_residual
        self.minimal_output = minimal_output
        self.property_groups = property_groups

    @property
    def entity(self) -> Curve:
        """
        Survey object.
        """
        return self._entity

    @entity.setter
    def entity(self, value):
        if not isinstance(value, Curve):
            raise TypeError("Entity must be a Curve.")

        self._entity = value

    @property
    def line_indices(self) -> list[int] | None:
        """
        Indices of vertices for line profile.
        """
        return self._line_indices

    @line_indices.setter
    def line_indices(self, value):
        if not isinstance(value, (list, np.ndarray)):
            raise TypeError("Line indices must be a list or numpy array.")

        self._line_indices = value

    @property
    def channels(self) -> list:
        """
        List of active channels.
        """
        return self._channels

    @property
    def property_groups(self) -> list[PropertyGroup]:
        """
        List of property groups.
        """
        return self._property_groups

    @property_groups.setter
    def property_groups(self, value):
        if not isinstance(value, list) or not all(
            isinstance(item, PropertyGroup) for item in value
        ):
            raise TypeError("Property groups must be a list of PropertyGroups.")

        self._property_groups = value
        channels = []
        for group in self._property_groups:
            channels += [self.entity.get_entity(uid)[0] for uid in group.properties]

        self._channels = list(set(channels))

    @property
    def smoothing(self) -> int:
        """
        Smoothing factor.
        """
        return self._smoothing

    @smoothing.setter
    def smoothing(self, value: int):
        if not isinstance(value, int):
            raise TypeError("Smoothing must be an integer.")

        self._smoothing = value

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
        if self._position is None and self.locations is not None:
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
        if self.position is None:
            return None

        locs = self.position.locations_resampled

        if locs is None:
            return None

        line_dataset = {}
        # Iterate over channels and add to anomalies
        for data in self.channels:
            if data.values is None:
                continue

            # Make LineData with current channel values
            line_data = LineData(
                data,
                self.position,
                self.min_amplitude,
                self.min_width,
                self.max_migration,
                self.min_value,
            )

            line_dataset[data.uid] = line_data

        if len(line_dataset) == 0:
            return None

        # Group anomalies
        line_groups = []
        for property_group in self.property_groups:
            line_group = LineGroup(
                position=self.position,
                line_dataset=line_dataset,
                property_group=property_group,
                max_migration=self.max_migration,
                min_channels=self.min_channels,
                minimal_output=self.minimal_output,
            )
            line_groups.append(line_group)

        return line_groups
