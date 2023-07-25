#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app project.
#
#  All rights reserved.
#

from __future__ import annotations

import copy

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
        line_dataset: list[LineData],
        property_group: dict,
        max_migration: float,
        channel_groups: dict,
        min_channels: int,
        data_normalization: list | str | tuple,
        channels: dict,
        minimal_output: bool,
    ):
        """
        :param line_dataset: List of line data with all anomalies.
        :param groups: List of anomaly groups.
        :param max_migration: Maximum peak migration.
        """
        self._position = position
        self._line_dataset = line_dataset
        self._property_group = property_group
        self._max_migration = max_migration
        self._channel_groups = channel_groups
        self._min_channels = min_channels
        self._data_normalization = data_normalization
        self._channels = channels
        self._minimal_output = minimal_output

        self._groups = None

    @property
    def groups(self) -> list[AnomalyGroup] | None:
        """
        List of anomaly groups.
        """
        if self._groups is None:
            self._groups = self.group_anomalies()
        return self._groups

    @groups.setter
    def groups(self, value):
        self._groups = value

    @property
    def position(self) -> LinePosition:
        """
        """
        return self._position

    @position.setter
    def position(self, value):
        self._position = value

    @property
    def line_dataset(self) -> list[LineData]:
        """
        List of line data.
        """
        return self._line_dataset

    @line_dataset.setter
    def line_dataset(self, value):
        self._line_dataset = value

    @property
    def property_group(self) -> dict:
        """
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
    def channel_groups(self) -> dict:
        """
        """
        return self._channel_groups

    @channel_groups.setter
    def channel_groups(self, value):
        self._channel_groups = value

    @property
    def min_channels(self) -> int:
        """
        """
        return self._min_channels

    @min_channels.setter
    def min_channels(self, value):
        self._min_channels = value

    @property
    def data_normalization(self) -> list | str | tuple:
        """
        """
        return self._data_normalization

    @data_normalization.setter
    def data_normalization(self, value):
        self._data_normalization = value

    @property
    def channels(self) -> dict:
        """
        """
        return self._channels

    @channels.setter
    def channels(self, value):
        self._channels = value

    @property
    def minimal_output(self) -> bool:
        """
        """
        return self._minimal_output

    @minimal_output.setter
    def minimal_output(self, value):
        self._minimal_output = value

    def get_near_peaks(
        self,
        i,
        full_group: np.ndarray,
        full_channel: np.ndarray,
        peaks_position: np.ndarray,
    ) -> np.ndarray:
        """
        Get indices of peaks within the migration distance.

        :param anomaly: Anomaly.
        :param locs: Line vertices.
        :param full_group: Array of group ids.
        :param full_channel: Array of channels.
        :param peaks_position: Array of peak positions.

        :return: Array of indices of peaks within the migration distance.
        """
        dist = np.abs(peaks_position[i] - peaks_position)
        # Find anomalies across channels within horizontal range
        near = np.where((dist < self.max_migration) & (full_group == -1))[0]

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

    def get_anomaly_attributes(self, line_data_subset):
        """
        Get full lists of anomaly attributes from line_dataset.

        :return: Full list of anomalies, group ids, channels, peak positions, and channel groups.
        """
        locs = self.position.locations_resampled

        full_anomalies = []
        full_group_ids = []
        full_channels = []
        full_peak_positions = []
        for line_data in line_data_subset:
            for anom in line_data.anomalies:
                full_anomalies.append(anom)
                full_group_ids.append(anom.group)
                full_channels.append(anom.channel)
                full_peak_positions.append(locs[anom.peak])
        full_channels = np.array(full_channels)
        full_group_ids = np.array(full_group_ids)
        full_peak_positions = np.array(full_peak_positions)
        return (
            full_anomalies,
            full_group_ids,
            full_channels,
            full_peak_positions,
        )

    def group_anomalies(  # pylint: disable=R0913, R0914
        self,
    ) -> list[AnomalyGroup]:
        """
        Group anomalies.

        :return: List of groups of anomalies.
        """
        azimuth = self.position.compute_azimuth()
        group_prop_size = np.r_[
            [len(grp["properties"]) for grp in self.channel_groups.values()]
        ]

        line_data_subset = []
        for line_data in self.line_dataset:
            if line_data.uid in self.property_group["properties"]:
                line_data_subset.append(line_data)

        # Get full lists of anomaly attributes
        (
            full_anomalies,
            full_group_ids,
            full_channels,
            full_peak_positions,
        ) = self.get_anomaly_attributes(line_data_subset)

        groups = []
        group_id = -1

        for ind, anom in enumerate(full_anomalies):
            # Skip if already labeled
            if anom.group != -1:
                continue
            group_id += 1

            # Find nearest peaks
            near = self.get_near_peaks(
                ind,
                full_group_ids,
                full_channels,
                full_peak_positions,
            )

            """
            score = np.zeros(len(self.channel_groups))
            for i in near:
                score[full_channel_groups[i]] += 1

            # Find groups with the largest channel overlap
            max_scores = np.where(score == score.max())[0]

            # in_group = max_scores
            in_group = max_scores[
                np.argmax(score[max_scores] / group_prop_size[max_scores])
            ]
            if score[in_group] < self.min_channels:
                continue
            """

            if len(near) == 0:
                continue

            # Update group_id on anomalies
            full_group_ids[near] = group_id
            for i in near:
                full_anomalies[i].group = group_id

            # Make AnomalyGroup
            near_anomalies = np.array(full_anomalies)[near]
            group = AnomalyGroup(
                self.position,
                copy.deepcopy(near_anomalies),
                self.property_group,
                azimuth,
                self.channels
            )
            # Normalize peak values
            for group_anom in group.anomalies:
                group_anom.peak_values *= np.prod(self.data_normalization)

            if self.minimal_output:
                group.minimal_output()

            groups += [group]

        return groups
