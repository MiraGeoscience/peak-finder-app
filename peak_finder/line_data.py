#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app project.
#
#  All rights reserved.
#

from __future__ import annotations

from uuid import UUID

import numpy as np

from peak_finder.anomaly import Anomaly
from peak_finder.anomaly_group import AnomalyGroup
from peak_finder.line_position import LinePosition


class LineData:
    """
    Contains full list of Anomaly objects.
    """

    def __init__(
        self,
        position: LinePosition,
        min_amplitude: int,
        min_width: float,
        max_migration: float,
    ):
        self._position = position
        self._min_amplitude = min_amplitude
        self._min_width = min_width
        self._max_migration = max_migration

        self._anomalies: list[Anomaly] = []

    @property
    def min_amplitude(self) -> int:
        """
        Minimum amplitude of anomaly.
        """
        return self._min_amplitude

    @min_amplitude.setter
    def min_amplitude(self, value):
        self._min_amplitude = value

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
        Max migration of anomaly.
        """
        return self._max_migration

    @max_migration.setter
    def max_migration(self, value):
        self._max_migration = value

    @property
    def position(self) -> LinePosition:
        """
        Line vertices and interpolation functions.
        """
        return self._position

    @position.setter
    def position(self, value):
        self._position = value

    @property
    def anomalies(self) -> list[Anomaly]:
        """
        Full list of anomalies.
        """
        return self._anomalies

    @anomalies.setter
    def anomalies(self, value):
        self._anomalies = value

    def get_list_attr(self, attr: str) -> list | np.ndarray:
        """
        Get list of anomaly attributes.

        :param attr: Attribute name.

        :return: List or np.ndarray of attribute values.
        """
        if attr == "channel_group":
            return [getattr(a, attr) for a in self.anomalies]
        return np.array([getattr(a, attr) for a in self.anomalies])

    def get_amplitude_and_width(
        self, locs: np.ndarray, values: np.ndarray, peak: int, start: int, end: int
    ) -> tuple[float, float, float]:
        """
        Get amplitude and width of anomaly.

        :param locs: Line vertices.
        :param values: Line values.
        :param peak: Index of peak of anomaly.
        :param start: Index of start of anomaly.
        :param end: Index of end of anomaly.

        :return: Amplitude and width of anomaly.
        """
        # Amplitude threshold
        delta_amp = (
            np.abs(
                np.min(
                    [
                        values[peak]  # pylint: disable=unsubscriptable-object
                        - values[start],  # pylint: disable=unsubscriptable-object
                        values[peak]  # pylint: disable=unsubscriptable-object
                        - values[end],  # pylint: disable=unsubscriptable-object
                    ]
                )
            )
            / (np.std(values) + 2e-32)
        ) * 100.0

        # Width threshold
        delta_x = locs[end] - locs[start]

        # Amplitude
        amplitude = (
            np.sum(np.abs(values[start:end]))  # pylint: disable=unsubscriptable-object
            * self.position.sampling
        )

        return delta_amp, delta_x, amplitude

    def get_peak_inds(
        self,
        locs: np.ndarray,
        peak: int,
        inds: np.ndarray,
        shift: int,
    ) -> np.ndarray:
        """
        Get indices for critical points.

        :param locs: Line vertices.
        :param peak: Index of peak of anomaly.
        :param inds: Indices to index locs.
        :param shift: Shift value.

        :return: Indices of critical points.
        """
        return np.median(
            [
                0,
                inds.shape[0] - 1,
                np.searchsorted(locs[inds], locs[peak]) - shift,
            ]
        ).astype(int)

    def iterate_over_peaks(
        self,
        values,
        channel: int,
        channel_groups: dict,
        uid: UUID,
        peaks_inds: np.ndarray,
        lows_inds: np.ndarray,
        inflect_up_inds: np.ndarray,
        inflect_down_inds: np.ndarray,
        locs: np.ndarray,
    ):
        """
        Iterate over peaks and add to anomalies.

        :param channel: Channel.
        :param channel_groups: Channel groups.
        :param uid: Channel uid.
        :param peaks_inds: Peak indices.
        :param lows_inds: Minima indices.
        :param inflect_up_inds: Upward inflection indices.
        :param inflect_down_inds: Downward inflection indices.
        :param locs: Locations.
        """
        if (
            len(peaks_inds) == 0
            or len(lows_inds) < 2
            or len(inflect_up_inds) < 2
            or len(inflect_down_inds) < 2
        ):
            return

        for peak in peaks_inds:
            # Get start of peak
            ind = self.get_peak_inds(locs, peak, lows_inds, 1)
            start = lows_inds[ind]

            # Get end of peak
            ind = self.get_peak_inds(locs, peak, lows_inds, 0)
            end = np.min([locs.shape[0] - 1, lows_inds[ind]])

            # Inflection points
            ind = self.get_peak_inds(locs, peak, inflect_up_inds, 1)
            inflect_up = inflect_up_inds[ind]

            ind = self.get_peak_inds(locs, peak, inflect_down_inds, 0)
            inflect_down = np.min([locs.shape[0] - 1, inflect_down_inds[ind] + 1])

            # Check amplitude and width thresholds
            delta_amp, delta_x, amplitude = self.get_amplitude_and_width(
                locs, values, peak, start, end
            )

            if (delta_amp > self.min_amplitude) & (delta_x > self.min_width):
                new_anomaly = Anomaly(
                    channel=channel,
                    start=start,
                    end=end,
                    inflect_up=inflect_up,
                    inflect_down=inflect_down,
                    peak=peak,
                    peak_values=values[peak],
                    amplitude=amplitude,
                    group=-1,
                    channel_group=[
                        key
                        for key, channel_group in enumerate(channel_groups.values())
                        if uid in channel_group["properties"]
                    ],
                )
                self.anomalies.append(new_anomaly)  # pylint: disable=no-member

    def get_near_peaks(
        self,
        anomaly: Anomaly,
        locs: np.ndarray,
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
        dist = np.abs(locs[anomaly.peak] - peaks_position)
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

    def group_anomalies(
        self,
        channel_groups: dict,
        group_prop_size: np.ndarray,
        min_channels: int,
        property_groups: list,
        data_uid: list,
        azimuth: np.ndarray,
        data_normalization: list | str,
        channels: dict,
        minimal_output: bool,
    ) -> list[AnomalyGroup]:
        """
        Group anomalies.

        :param channel_groups: Channel groups.
        :param group_prop_size: Group property size.
        :param min_channels: Minimum number of channels in a group.
        :param property_groups: Property groups.
        :param data_uid: List of uids for channels.
        :param azimuth: Azimuth.
        :param data_normalization: Data normalization factor.
        :param channels: Channels.
        :param minimal_output: Whether to return minimal output.

        :return: List of groups of anomalies.
        """
        locs = self.position.locations_resampled

        peak_inds = self.get_list_attr("peak")
        peaks_position = locs[peak_inds]

        full_group = self.get_list_attr("group")
        full_channel = self.get_list_attr("channel")

        groups = []
        group_id = -1

        for anom in self.anomalies:
            # Skip if already labeled
            if anom.group != -1:
                continue
            group_id += 1

            # Find nearest peaks
            near = self.get_near_peaks(
                anom,
                locs,
                full_group,
                full_channel,
                peaks_position,
            )

            score = np.zeros(len(channel_groups))
            for ind in near:
                channel_groups_list = self.get_list_attr("channel_group")
                score[channel_groups_list[ind]] += 1

            # Find groups with the largest channel overlap
            max_scores = np.where(score == score.max())[0]
            # Keep the group with fewer properties
            in_group = max_scores[
                np.argmax(score[max_scores] / group_prop_size[max_scores])
            ]
            if score[in_group] < min_channels:
                continue

            channel_group = property_groups[in_group]
            # Remove anomalies not in group
            mask = [
                data_uid[self.anomalies[ind].channel] in channel_group["properties"]
                for ind in near
            ]
            near = near[mask, ...]
            if len(near) == 0:
                continue

            # Update group_id on anomalies
            full_group[near] = group_id
            for ind in near:
                self.anomalies[ind].group = group_id

            # Make AnomalyGroup
            near_anomalies = np.array(self.anomalies)[near]
            group = AnomalyGroup(near_anomalies, channel_group)

            peaks = group.get_list_attr("peak")
            cox_sort = np.argsort(locs[peaks])
            azimuth_near = azimuth[peaks]
            group.azimuth = group.compute_dip_direction(azimuth, peaks, cox_sort)
            group.migration = np.abs(
                locs[peaks[cox_sort[-1]]] - locs[peaks[cox_sort[0]]]
            )
            group.skew = group.compute_skew(
                locs,
                peaks,
                cox_sort,
                azimuth_near,
            )

            # Normalize peak values
            for group_anom in group.anomalies:
                group_anom.peak_values *= np.prod(data_normalization)

            group.amplitude = np.sum([anom.amplitude for anom in group.anomalies])
            group.linear_fit = group.compute_linear_fit(channels)
            group.cox = (
                np.mean(
                    self.position.interpolate_array(peaks[cox_sort[0]]),
                    axis=0,
                ),
            )

            if minimal_output:
                group.minimal_output(self.position)

            groups += [group]

        return groups
