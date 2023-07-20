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
        anomalies: list[Anomaly],
        channel_group: dict,
    ):
        self._anomalies = anomalies
        self._channel_group = channel_group

        self._linear_fit: list | None = None
        self._skew: float | None = None
        self._amplitude: float | None = None
        self._migration: float | None = None
        self._azimuth: float | None = None
        self._cox: np.ndarray | None = None

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
    def cox(self) -> np.ndarray | None:
        """
        Center of oxidized coal.
        """
        return self._cox

    @cox.setter
    def cox(self, value):
        self._cox = value

    @property
    def migration(self) -> float | None:
        """
        Distance migrated from anomaly.
        """
        return self._migration

    @migration.setter
    def migration(self, value):
        self._migration = value

    @property
    def amplitude(self) -> float | None:
        """
        Amplitude of anomalies.
        """
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value):
        self._amplitude = value

    @property
    def linear_fit(self) -> list | None:
        """
        Intercept and slope of linear fit.
        """
        return self._linear_fit

    @linear_fit.setter
    def linear_fit(self, value):
        self._linear_fit = value

    @property
    def skew(self) -> float | None:
        """
        Skew.
        """
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
    def azimuth(self) -> float | None:
        """
        Azimuth of anomalies.
        """
        return self._azimuth

    @azimuth.setter
    def azimuth(self, value):
        self._azimuth = value

    def get_list_attr(self, attr: str) -> list | np.ndarray:
        """
        Get list of attribute from anomalies.

        :param attr: Attribute to get.

        :return: List of attribute.
        """
        if attr == "channel_group":
            return [getattr(a, attr) for a in self.anomalies]
        return np.array([getattr(a, attr) for a in self.anomalies])

    def minimal_output(self, position: LinePosition):
        """
        Interpolate anomaly properties for minimal output.

        :param position: Line locations and interpolation functions.
        """
        self.skew = np.mean(self.skew)
        for anom in self.anomalies:
            anom.start = position.interpolate_array(anom.start)
            anom.end = position.interpolate_array(anom.end)
            anom.inflect_up = position.interpolate_array(anom.inflect_up)
            anom.inflect_down = position.interpolate_array(anom.inflect_down)
            anom.peak = position.interpolate_array(anom.peak)

    def compute_dip_direction(
        self,
        azimuth: np.ndarray,
        cox: np.ndarray,
        cox_sort: np.ndarray,
    ) -> float:
        """
        Compute dip direction for an anomaly group.

        :param azimuth: Azimuth values for all anomalies.
        :param cox: Peak indices.
        :param cox_sort: Indices to sort cox.

        :return: Dip direction.
        """
        peak_values = self.get_list_attr("peak_values")
        dip_direction = azimuth[cox[0]]
        if peak_values[cox_sort][0] < peak_values[cox_sort][-1]:
            dip_direction = (dip_direction + 180) % 360.0
        return dip_direction

    def compute_skew(
        self,
        locs: np.ndarray,
        cox: np.ndarray,
        cox_sort: np.ndarray,
        azimuth_near: np.ndarray,
    ) -> float:
        """
        Compute skew factor for an anomaly group.

        :param locs: Resampled line vertices.
        :param cox: Center of oxidized coal; center of peak.
        :param cox_sort: Indices to sort cox.
        :param azimuth_near: Azimuth values for the group.

        :return: Skew.
        """
        inflect_up = self.get_list_attr("inflect_up")
        inflect_down = self.get_list_attr("inflect_down")

        skew = (locs[cox][cox_sort[0]] - locs[inflect_up][cox_sort]) / (
            locs[inflect_down][cox_sort] - locs[cox][cox_sort[0]] + 1e-8
        )
        skew[azimuth_near[cox_sort] > 180] = 1.0 / (
            skew[azimuth_near[cox_sort] > 180] + 1e-2
        )
        # Change skew factor from [-100, 1]
        flip_skew = skew < 1
        skew[flip_skew] = 1.0 / (skew[flip_skew] + 1e-2)
        skew = 1.0 - skew
        skew[flip_skew] *= -1

        return skew

    def compute_linear_fit(
        self,
        channels: dict,
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
            for i, channel in enumerate(channels.values())
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
