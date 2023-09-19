#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app project.
#
#  All rights reserved.
#

# pylint: disable=too-many-arguments

from __future__ import annotations

import uuid

import numpy as np
from geoh5py.groups import PropertyGroup

from peak_finder.anomaly_group import AnomalyGroup
from peak_finder.line_data import LineData
from peak_finder.line_position import LinePosition


class LineGroup:
    """
    Contains list of AnomalyGroup objects.
    """

    def __init__(
        self,
        position: LinePosition,
        line_dataset: dict[uuid.UUID, LineData],
        property_group: PropertyGroup,
        max_migration: float,
        min_channels: int,
        n_groups: int,
        max_separation: float,
        minimal_output: bool,
    ):
        """
        :param line_dataset: List of line data with all anomalies.
        :param groups: List of anomaly groups.
        :param max_migration: Maximum peak migration.
        """
        self._position = position
        self._line_dataset = line_dataset
        self.property_group = property_group
        self._max_migration = max_migration
        self._min_channels = min_channels
        self._n_groups = n_groups
        self._max_separation = max_separation
        self._minimal_output = minimal_output
        self._channels: dict[uuid.UUID, LineData] | None = None
        self._groups: list[AnomalyGroup] | None = None

    @property
    def groups(self) -> list[AnomalyGroup] | None:
        """
        List of anomaly groups.
        """
        if self._groups is None:
            self._groups = self.compute()
        return self._groups

    @property
    def position(self) -> LinePosition:
        """
        Line Position.
        """
        return self._position

    @position.setter
    def position(self, value):
        self._position = value

    @property
    def line_dataset(self) -> dict[uuid.UUID, LineData]:
        """
        List of line data.
        """
        return self._line_dataset

    @line_dataset.setter
    def line_dataset(self, value):
        self._line_dataset = value

    @property
    def property_group(self) -> PropertyGroup:
        """
        Property group.
        """
        return self._property_group

    @property_group.setter
    def property_group(self, value):
        self._property_group = value

    @property
    def max_migration(self) -> float:
        """
        Maximum peak migration.
        """
        return self._max_migration

    @max_migration.setter
    def max_migration(self, value):
        self._max_migration = value

    @property
    def min_channels(self) -> int:
        """
        Minimum number of channels in anomaly.
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
    def max_separation(self) -> float:
        """
        Max separation between peaks to merge.
        """
        return self._max_separation

    @max_separation.setter
    def max_separation(self, value):
        self._max_separation = value

    @property
    def channels(self) -> dict | None:
        """
        Dict of active channels and values.
        """
        if self._channels is None:
            channels = {}
            for uid in self.property_group.properties:  # type: ignore
                channels[uid] = self.line_dataset[uid]
            self._channels = channels

        return self._channels

    @property
    def minimal_output(self) -> bool:
        """
        Whether to return minimal output.
        """
        return self._minimal_output

    @minimal_output.setter
    def minimal_output(self, value):
        self._minimal_output = value

    def get_near_peaks(
        self,
        ind: int,
        full_channel: np.ndarray,
        peaks_position: np.ndarray,
    ) -> np.ndarray:
        """
        Get indices of peaks within the migration distance.

        :param ind: Index of anomaly.
        :param full_channel: Array of channels.
        :param peaks_position: Array of peak positions.

        :return: Array of indices of peaks within the migration distance.
        """
        dist = np.abs(peaks_position[ind] - peaks_position)
        # Find anomalies across channels within horizontal range
        near = np.where(dist < self.max_migration)[0]

        # Reject from group if channel gap > 1
        u_gates, u_count = np.unique(full_channel[near], return_counts=True)
        if len(u_gates) > 1 and np.any((u_gates[1:] - u_gates[:-1]) > 2):
            cutoff = u_gates[np.where((u_gates[1:] - u_gates[:-1]) > 2)[0][0]]
            near = near[full_channel[near] <= cutoff]  # Remove after cutoff

        # Check for multiple nearest peaks on single channel
        # and keep the nearest
        u_gates, u_count = np.unique(full_channel[near], return_counts=True)
        for gate in u_gates[np.where(u_count > 1)]:
            mask = np.ones_like(near, dtype="bool")
            sub_ind = full_channel[near] == gate
            sub_ind[np.where(sub_ind)[0][np.argmin(dist[near][sub_ind])]] = False
            mask[sub_ind] = False
            near = near[mask]

        return near

    def get_anomaly_attributes(self, line_data_subset: list[LineData]) -> tuple:
        """
        Get full lists of anomaly attributes from line_dataset.

        :param line_data_subset: List of line data corresponding to property group.

        :return: Full list of anomalies, group ids, channels, peak positions,
            peak values, and channel groups.
        """
        locs = self.position.locations_resampled

        full_anomalies = []
        full_channels = []
        full_peak_positions = []
        full_peak_values = []
        for ind, line_data in enumerate(line_data_subset):
            values = line_data.values_resampled
            for anom in line_data.anomalies:
                full_anomalies.append(anom)
                full_channels.append(ind)
                full_peak_positions.append(locs[anom.peak])
                full_peak_values.append(values[anom.peak])
        full_channels = np.array(full_channels)
        full_peak_positions = np.array(full_peak_positions)
        return (
            full_anomalies,
            full_channels,
            full_peak_positions,
            full_peak_values,
        )

    def group_n_groups(self, groups):  # pylint: disable=R0914
        """
        Look for anomalies connected by their ends.
        """
        # Sort anomalies by amplitude
        all_amplitudes = np.array([group.amplitude for group in groups])
        sorted_ind = np.argsort(-all_amplitudes)

        locs = self.position.locations_resampled
        all_starts = np.vstack([locs[group.start] for group in groups])

        merged = []
        ignore = []
        for ind in sorted_ind:
            if groups[ind] in ignore:
                continue
            # Find closest groups
            rad = np.linalg.norm(locs[groups[ind].end] - all_starts, axis=1)
            in_range = np.where(rad < self.max_separation)[0]
            if ind in in_range:
                # Remove the group itself
                in_range = np.delete(in_range, np.where(in_range == ind)[0][0])
            # Sort in_range
            sorted_in_range = in_range[np.argsort(rad[in_range])]

            groups_to_merge = [groups[ind]]
            i = 0
            while len(groups_to_merge) < self.n_groups and i < len(sorted_in_range):
                val = sorted_in_range[i]
                i += 1
                if groups[val] in ignore:
                    continue

                groups_to_merge.append(groups[val])
                ignore.append(groups[val])

                if groups[val] in merged:
                    merged.remove(groups[val])
                if groups[ind] not in ignore:
                    ignore.append(groups[ind])

            # Merge groups
            if len(groups_to_merge) == 1:
                new_group = groups[ind]
            else:
                new_group = AnomalyGroup(
                    self.position,
                    np.concatenate(tuple(group.anomalies for group in groups_to_merge)),
                    self.property_group,
                    np.concatenate(
                        tuple(group.full_azimuth for group in groups_to_merge)
                    ),
                    self.channels,
                    np.concatenate(
                        tuple(group.full_peak_values for group in groups_to_merge)
                    ),
                )
            merged.append(new_group)

        return merged

    def compute(  # pylint: disable=R0913, R0914
        self,
    ) -> list[AnomalyGroup] | None:
        """
        Group anomalies.

        :return: List of groups of anomalies.
        """
        groups = []
        group_id = -1
        azimuth = self.position.compute_azimuth()

        if self.channels is None:
            return None

        # Get full lists of anomaly attributes
        (
            full_anomalies,
            full_channels,
            full_peak_positions,
            full_peak_values,
        ) = self.get_anomaly_attributes(list(self.channels.values()))

        full_group_ids = np.ones(len(full_anomalies), dtype="bool") * -1
        for ind, _ in enumerate(full_anomalies):
            # Skip if already labeled
            if full_group_ids[ind] != -1:
                continue

            group_id += 1

            # Find nearest peaks
            near = self.get_near_peaks(
                ind,
                full_channels,
                full_peak_positions,
            )

            if (
                len(np.unique(full_channels[near])) < self.min_channels
                or len(near) == 0
            ):
                continue

            full_group_ids[near] = group_id

            # Make AnomalyGroup
            near_anomalies = np.array(full_anomalies)[near]
            near_values = np.array(full_peak_values)[near]

            group = AnomalyGroup(
                self.position,
                near_anomalies,
                self.property_group,
                azimuth,
                self.channels,
                near_values,
            )
            groups += [group]

        if self.n_groups > 1:
            groups = self.group_n_groups(groups)

        return groups
