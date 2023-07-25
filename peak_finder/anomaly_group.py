#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app project.
#
#  All rights reserved.
#

from __future__ import annotations

import numpy as np

from peak_finder.anomaly import Anomaly
from peak_finder.line_position import LinePosition


class AnomalyGroup:
    """
    Group of anomalies. Contains list with a subset of anomalies.
    """

    def __init__(
        self,
        position: LinePosition,
        anomalies: list[Anomaly],
        channel_group: dict,
        full_azimuth: np.ndarray,
        channels: dict,
    ):
        self._position = position
        self._anomalies = anomalies
        self._channel_group = channel_group
        self._full_azimuth = full_azimuth
        self._channels = channels

        self._linear_fit: list | None = None
        self._skew: float | None = None
        self._amplitude: float | None = None
        self._migration: float | None = None
        self._azimuth: float | None = None
        self._group_center: np.ndarray | None = None
        self._group_center_sort: np.ndarray | None = None
        self._peak: np.ndarray | None = None
        self._full_peaks: np.ndarray | None = None

    @property
    def position(self) -> LinePosition:
        """
        Line position.
        """
        return self._position

    @position.setter
    def position(self, value):
        self._position = value

    @property
    def anomalies(self) -> list[Anomaly]:
        """
        List of anomalies that are grouped together.
        """
        return self._anomalies

    @anomalies.setter
    def anomalies(self, value):
        self._anomalies = value

    @property
    def group_center(self) -> np.ndarray | None:
        """
        Group center.
        """
        if self._group_center is None:
            peaks = self.get_list_attr("peak")
            self._group_center = (
                np.mean(
                    self.position.interpolate_array(self.full_peaks[self.group_center_sort[0]]),
                    axis=0,
                ),
            )
        return self._group_center

    @group_center.setter
    def group_center(self, value):
        self._group_center = value

    @property
    def group_center_sort(self) -> np.ndarray | None:
        """
        Group center sorting indices.
        """
        if self._group_center_sort is None:
            locs = self.position.locations_resampled
            self._group_center_sort = np.argsort(locs[self.full_peaks])
        return self._group_center_sort

    @property
    def migration(self) -> float | None:
        """
        Distance migrated from anomaly.
        """
        if self._migration is None:
            locs = self.position.locations_resampled
            self._migration = np.abs(
                locs[self.full_peaks[self.group_center_sort[-1]]] - locs[self.full_peaks[self.group_center_sort[0]]]
            )
        return self._migration

    @property
    def amplitude(self) -> float | None:
        """
        Amplitude of anomalies.
        """
        if self._amplitude is None:
            self._amplitude = np.sum([anom.amplitude for anom in self.anomalies])
        return self._amplitude

    @property
    def linear_fit(self) -> list | None:
        """
        Intercept and slope of linear fit.
        """
        if self._linear_fit is None:
            self._linear_fit = self.compute_linear_fit()
        return self._linear_fit

    @property
    def skew(self) -> float | None:
        """
        Skew.
        """
        if self._skew is None:
            self._skew = self.compute_skew()
        return self._skew

    @skew.setter
    def skew(self, value):
        self._skew = value

    @property
    def channel_group(self) -> dict:
        """
        Channel group.
        """
        return self._channel_group

    @channel_group.setter
    def channel_group(self, value):
        self._channel_group = value

    @property
    def channels(self) -> dict:
        """
        """
        return self._channels

    @channels.setter
    def channels(self, value):
        self._channels = value

    @property
    def azimuth(self) -> float | None:
        """
        Azimuth of anomalies.
        """
        if self._azimuth is None:
            self._azimuth = self.compute_dip_direction()
        return self._azimuth

    @azimuth.setter
    def azimuth(self, value):
        self._azimuth = value

    @property
    def full_azimuth(self) -> float | None:
        """
        """
        return self._full_azimuth

    @full_azimuth.setter
    def full_azimuth(self, value):
        self._full_azimuth = value

    @property
    def full_peaks(self) -> np.ndarray | None:
        """
        List of peaks from all anomalies in group.
        """
        if self._full_peaks is None:
            self._full_peaks = self.get_list_attr("peak")
        return self._full_peaks

    def get_list_attr(self, attr: str) -> list | np.ndarray:
        """
        Get list of attribute from anomalies.

        :param attr: Attribute to get.

        :return: List of attribute.
        """
        if attr == "channel_group":
            return [getattr(a, attr) for a in self.anomalies]
        return np.array([getattr(a, attr) for a in self.anomalies])

    def minimal_output(self):
        """
        Interpolate anomaly properties for minimal output.
        """
        self.skew = np.mean(self.skew)
        for anom in self.anomalies:
            anom.start = self.position.interpolate_array(anom.start)
            anom.end = self.position.interpolate_array(anom.end)
            anom.inflect_up = self.position.interpolate_array(anom.inflect_up)
            anom.inflect_down = self.position.interpolate_array(anom.inflect_down)
            anom.peak = self.position.interpolate_array(anom.peak)

    def compute_dip_direction(
        self,
    ) -> float:
        """
        Compute dip direction for an anomaly group.

        :param azimuth: Azimuth values for all anomalies.
        :param group_center: Peak indices.
        :param group_center_sort: Indices to sort group center.

        :return: Dip direction.
        """
        peak_values = self.get_list_attr("peak_values")
        dip_direction = None
        if len(self.group_center) > 0:
            dip_direction = self.full_azimuth[self.full_peaks[0]]
        if peak_values[self.group_center_sort][0] < peak_values[self.group_center_sort][-1]:
            dip_direction = (dip_direction + 180) % 360.0
        return dip_direction

    def compute_skew(
        self,
    ) -> float:
        """
        Compute skew factor for an anomaly group.

        :return: Skew.
        """
        locs = self.position.locations_resampled
        peaks = self.get_list_attr("peak")
        azimuth_near = self.full_azimuth[peaks]

        inflect_up = self.get_list_attr("inflect_up")
        inflect_down = self.get_list_attr("inflect_down")

        skew = (
            locs[peaks][self.group_center_sort[0]]
            - locs[inflect_up][self.group_center_sort]
        ) / (
            locs[inflect_down][self.group_center_sort]
            - locs[peaks][self.group_center_sort[0]]
            + 1e-8
        )
        skew[azimuth_near[self.group_center_sort] > 180] = 1.0 / (
            skew[azimuth_near[self.group_center_sort] > 180] + 1e-2
        )
        # Change skew factor from [-100, 1]
        flip_skew = skew < 1
        skew[flip_skew] = 1.0 / (skew[flip_skew] + 1e-2)
        skew = 1.0 - skew
        skew[flip_skew] *= -1

        return skew

    def compute_linear_fit(
        self,
    ) -> list[float] | None:
        """
        Compute linear fit for the anomaly group.

        :param channels: Channels.

        :return: List of intercept, slope for the linear fit.
        """
        gates = np.array([a.channel for a in self.anomalies])
        values = np.array([a.peak_values for a in self.anomalies])

        times = [
            channel["time"]
            for i, channel in enumerate(self.channels.values())
            if (i in list(gates) and "time" in channel)
        ]

        linear_fit = None
        if len(times) > 2 and len(self.anomalies) > 0:
            times = np.hstack(times)[values > 0]
            if len(times) > 2:
                # Compute linear trend
                slope, intercept = np.polyfit(times, np.log(values[values > 0]), 1)
                linear_fit = [intercept, slope]

        return linear_fit
