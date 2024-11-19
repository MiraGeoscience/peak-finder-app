#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app project.
#
#  All rights reserved.
#


from __future__ import annotations
from uuid import UUID
import numpy as np


from peak_finder.line_data import LineData
from peak_finder.line_group import LineGroup
from peak_finder.line_position import LinePosition


class LineAnomaly:  # pylint: disable=R0902, duplicate-code
    """
    Main class for finding anomalies.

    Contains list of LineGroup objects.
    """

    def __init__(  # pylint: disable=R0913, R0914
        self,
        *,
        vertices,
        cells,
        line_id,
        line_indices: np.ndarray,
        line_start: np.ndarray,
        property_groups: dict[str, dict[UUID, list[np.ndarray]]],
        max_migration=50.0,
        minimal_output=False,
        min_amplitude=25,
        min_channels=3,
        min_value=-np.inf,
        min_width=200.0,
        smoothing=1,
        n_groups=1,
        max_separation=100.0,
        use_residual=False,
    ):
        """
        :param vertices: Array of vertices.
        :param cells: Array of cells.
        :param line_id: Line ID.
        :param line_indices: Indices of vertices for line profile.
        :param line_start: Start location of line.
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
        self.locations = vertices
        self.cells = cells
        self.line_id = line_id
        self.line_indices = line_indices
        self.line_start = line_start
        self.smoothing = smoothing
        self.min_amplitude = min_amplitude
        self.min_value = min_value
        self.min_width = min_width
        self.max_migration = max_migration
        self.min_channels = min_channels
        self.use_residual = use_residual
        self.minimal_output = minimal_output
        self.property_groups = self.validate_property_groups(property_groups)
        self.n_groups = n_groups
        self.max_separation = max_separation

        # Metrics
        self.centers: np.ndarray | None = None
        self.starts: np.ndarray | None = None
        self.ends: np.ndarray | None = None
        self.amplitudes: np.ndarray | None = None
        self.group_ids: np.ndarray | None = None

    @property
    def line_id(self) -> int | None:
        """
        Line ID.
        """
        return self._line_id

    @line_id.setter
    def line_id(self, value: int):
        self._line_id = value

    @property
    def line_indices(self) -> np.ndarray:
        """
        Indices of vertices for line profile.
        """
        return self._line_indices

    @line_indices.setter
    def line_indices(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise TypeError("Line indices must be a numpy array.")

        self._line_indices = value

    @property
    def line_start(self) -> np.ndarray:
        """
        Index for start of the line.
        """
        return self._line_start

    @line_start.setter
    def line_start(self, value):
        self._line_start = value

    @staticmethod
    def validate_property_groups(value):
        if not isinstance(value, dict) or not all(
            isinstance(item, dict) for item in value.values()
        ):
            raise TypeError("Property groups must be a dict.")

        return value

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
    def n_groups(self) -> int:
        """
        Number of peaks to merge.
        """
        return self._n_groups

    @n_groups.setter
    def n_groups(self, value):
        self._n_groups = value

    @property
    def max_separation(self) -> int:
        """
        Max separation between peaks to merge.
        """
        return self._max_separation

    @max_separation.setter
    def max_separation(self, value):
        self._max_separation = value

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
    def locations(self) -> np.ndarray:
        """
        Vertices for the underlying curve object.
        """
        return self._locations

    @locations.setter
    def locations(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise TypeError("Vertices must be a numpy array.")

        self._locations = value

    @property
    def cells(self) -> np.ndarray:
        """
        Curve object cells.
        """
        return self._cells

    @cells.setter
    def cells(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise TypeError("Curve indices must be a numpy array.")

        self._cells = value

    @property
    def anomalies(self) -> list[LineGroup] | None:
        """
        List of line groups.
        """
        if self._anomalies is None:
            self._anomalies = self.find_anomalies()
            self._store_metrics()

        return self._anomalies

    def _store_metrics(self):
        """
        Store metrics for line anomalies.
        """
        if self._anomalies is None:
            return

        centers = []
        starts = []
        ends = []
        amplitudes = []
        group_ids = []
        group_names = list(self.property_groups)
        for anomaly in self._anomalies:
            for group in anomaly.groups:
                centers.append(group.center)
                starts.append(group.start)
                ends.append(group.end)
                amplitudes.append(group.amplitude)
                group_ids.append(group_names.index(anomaly.property_group))

        if len(centers) == 0:
            return

        self.centers = np.vstack(centers)
        self.starts = np.hstack(starts)
        self.ends = np.hstack(ends)
        self.amplitudes = np.hstack(amplitudes)
        self.group_ids = np.hstack(group_ids)

    @property
    def position(self) -> LinePosition | None:
        """
        Line position and interpolation.
        """
        if self._position is None:
            bool_cells = np.all(self.line_indices[self.cells], axis=1)

            active_cells = self.cells[bool_cells]

            if active_cells.size == 0:
                return None

            sorting = np.concatenate((active_cells[:, 0], [active_cells[-1, 1]]))

            self._position = LinePosition(
                locations=self.locations,
                line_indices=self.line_indices,
                line_start=self.line_start,
                sorting=sorting,
                smoothing=self.smoothing,
                residual=self.use_residual,
            )
        return self._position

    def find_anomalies(  # pylint: disable=R0914
        self,
    ) -> list[LineGroup]:
        """
        Find all anomalies along a line profile of data.
        Anomalies are detected based on the lows, inflection points and peaks.
        Neighbouring anomalies are then grouped and assigned a channel_group label.

        :return: List of groups and line profile.
        """
        # Group anomalies
        if self.position is None:
            return []

        line_groups: list[LineGroup] = []
        for name, data_dict in self.property_groups.items():
            line_dataset = [
                LineData(
                    uid,
                    data,
                    self.position,
                    min_amplitude=self.min_amplitude,
                    min_width=self.min_width,
                    max_migration=self.max_migration,
                    min_value=self.min_value,
                )
                for uid, data in data_dict.items()
            ]

            if len(line_dataset) == 0:
                continue

            line_group = LineGroup(
                position=self.position,
                line_dataset=line_dataset,
                property_group=name,
                max_migration=self.max_migration,
                min_channels=self.min_channels,
                n_groups=self.n_groups,
                max_separation=self.max_separation,
                minimal_output=self.minimal_output,
            )
            line_groups.append(line_group)

        return line_groups
