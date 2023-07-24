#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app project.
#
#  All rights reserved.
#

from __future__ import annotations

import copy

import numpy as np

from peak_finder.anomaly_group import AnomalyGroup
from peak_finder.line_data import LineData
from peak_finder.line_position import LinePosition


class LineGroup:
    """
    Contains list of AnomalyGroup objects.
    """

    def __init__(
        self,
        line_dataset: list[LineData],
        groups: list[AnomalyGroup] | None = None,
        max_migration: float = 50.0,
    ):
        """
        :param line_dataset: List of line data with all anomalies.
        :param groups: List of anomaly groups.
        :param max_migration: Maximum peak migration.
        """
        self._line_dataset = line_dataset
        self._groups = groups
        self._max_migration = max_migration

    @property
    def groups(self) -> list[AnomalyGroup] | None:
        """
        List of anomaly groups.
        """
        return self._groups

    @groups.setter
    def groups(self, value):
        self._groups = value

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
    def max_migration(self) -> float:
        """
        Maximum peak migration.
        """
        return self._max_migration

    @max_migration.setter
    def max_migration(self, value):
        self._max_migration = value

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

    def get_anomaly_attributes(self, locs):
        """
        Get full lists of anomaly attributes from line_dataset.

        :param locs: Line vertices.

        :return: Full list of anomalies, group ids, channels, peak positions, and channel groups.
        """
        full_anomalies = []
        full_group_ids = []
        full_channels = []
        full_peak_positions = []
        full_channel_groups = []
        for line_data in self.line_dataset:
            for anom in line_data.anomalies:
                full_anomalies.append(anom)
                full_group_ids.append(anom.group)
                full_channels.append(anom.channel)
                full_peak_positions.append(locs[anom.peak])
                full_channel_groups.append(anom.channel_group)
        full_channels = np.array(full_channels)
        full_group_ids = np.array(full_group_ids)
        full_peak_positions = np.array(full_peak_positions)
        return (
            full_anomalies,
            full_group_ids,
            full_channels,
            full_peak_positions,
            full_channel_groups,
        )

    def group_anomalies(  # pylint: disable=R0913, R0914
        self,
        position: LinePosition,
        channel_groups: dict,
        min_channels: int,
        property_group: dict,
        group_prop_size: list,
        azimuth: np.ndarray,
        data_normalization: list | str | tuple,
        channels: dict,
        minimal_output: bool,
    ):
        """
        Group anomalies.

        :param position: Line position.
        :param channel_groups: Channel groups.
        :param group_prop_size: Group property size.
        :param min_channels: Minimum number of channels in a group.
        :param property_group: Property group.
        :param azimuth: Azimuth.
        :param data_normalization: Data normalization factor.
        :param channels: Channels.
        :param minimal_output: Whether to return minimal output.

        :return: List of groups of anomalies.
        """
        # LinePosition information
        locs = position.locations_resampled

        # Get full lists of anomaly attributes
        (
            full_anomalies,
            full_group_ids,
            full_channels,
            full_peak_positions,
            full_channel_groups,
        ) = self.get_anomaly_attributes(locs)

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

            score = np.zeros(len(channel_groups))
            for i in near:
                score[full_channel_groups[i]] += 1

            # Find groups with the largest channel overlap
            max_scores = np.where(score == score.max())[0]

            # in_group = max_scores
            in_group = max_scores[
                np.argmax(score[max_scores] / group_prop_size[max_scores])
            ]
            if score[in_group] < min_channels:
                continue

            if len(near) == 0:
                continue

            # Update group_id on anomalies
            full_group_ids[near] = group_id
            for i in near:
                full_anomalies[i].group = group_id

            # Make AnomalyGroup
            near_anomalies = np.array(full_anomalies)[near]
            group = AnomalyGroup(copy.deepcopy(near_anomalies), property_group)

            peaks = group.get_list_attr("peak")
            group_center_sort = np.argsort(locs[peaks])
            azimuth_near = azimuth[peaks]

            # Set group attributes
            group.azimuth = group.compute_dip_direction(
                azimuth, peaks, group_center_sort
            )
            group.migration = np.abs(
                locs[peaks[group_center_sort[-1]]] - locs[peaks[group_center_sort[0]]]
            )
            group.skew = group.compute_skew(
                locs,
                peaks,
                group_center_sort,
                azimuth_near,
            )

            # Normalize peak values
            for group_anom in group.anomalies:
                group_anom.peak_values *= np.prod(data_normalization)

            group.amplitude = np.sum([anom.amplitude for anom in group.anomalies])
            group.linear_fit = group.compute_linear_fit(channels)
            group.group_center = (
                np.mean(
                    position.interpolate_array(peaks[group_center_sort[0]]),
                    axis=0,
                ),
            )

            if minimal_output:
                group.minimal_output(position)

            groups += [group]

        self.groups = groups
