#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app project.
#
#  All rights reserved.
#

# pylint: disable=too-many-instance-attributes, too-many-arguments

from __future__ import annotations

import numpy as np
from geoh5py.groups import PropertyGroup

from peak_finder.anomaly import Anomaly
from peak_finder.line_position import LinePosition


class AnomalyGroup:  # pylint: disable=too-many-public-methods
    """
    Group of anomalies. Contains list with a subset of anomalies.
    """

    _position: LinePosition
    _anomalies: list[Anomaly]
    _property_group: PropertyGroup
    _full_azimuth: np.ndarray
    _channels: dict
    _full_peak_values: np.ndarray
    _linear_fit: list | None = None
    _skew: float | None = None
    _amplitude: float | None = None
    _migration: float | None = None
    _azimuth: float | None = None
    _group_center: np.ndarray | None = None
    _group_center_sort: np.ndarray | None = None
    _peak: np.ndarray | None = None
    _peaks: np.ndarray | None = None
    _start: int | None = None
    _end: int | None = None

    def __init__(
        self,
        position: LinePosition,
        anomalies: list[Anomaly],
        property_group: PropertyGroup,
        full_azimuth: np.ndarray,
        channels: dict,
        full_peak_values: np.ndarray,
        subgroups: set[AnomalyGroup],
    ):
        self.anomalies = anomalies
        self.channels = channels
        self.full_azimuth = full_azimuth
        self.full_peak_values = full_peak_values
        self.position = position
        self.property_group = property_group
        self.subgroups = subgroups

    @property
    def position(self) -> LinePosition:
        """
        Line position.
        """
        return self._position

    @position.setter
    def position(self, value: LinePosition):
        if not isinstance(value, LinePosition):
            raise TypeError("Attribute 'position' must be a LinePosition object.")

        self._position = value

    @property
    def anomalies(self) -> list[Anomaly]:
        """
        List of anomalies that are grouped together.
        """
        return self._anomalies

    @anomalies.setter
    def anomalies(self, value: list[Anomaly]):
        if not isinstance(value, list) and not all(
            isinstance(item, Anomaly) for item in value
        ):
            raise TypeError("Attribute 'anomalies` must be a list of Anomaly objects.")
        self._anomalies = value

    @property
    def group_center(self) -> np.ndarray | None:
        """
        Group center.
        """
        if (
            self._group_center is None
            and self.group_center_sort is not None
            and self.peaks is not None
        ):
            self._group_center = np.mean(
                self.position.interpolate_array(self.peaks[self.group_center_sort]),
                axis=0,
            )
        return self._group_center

    @property
    def group_center_sort(self) -> np.ndarray | None:
        """
        Group center sorting indices.
        """
        if self._group_center_sort is None:
            locs = self.position.locations_resampled
            self._group_center_sort = np.argsort(locs[self.peaks])
        return self._group_center_sort

    @property
    def migration(self) -> float | None:
        """
        Distance migrated from anomaly.
        """
        if (
            self._migration is None
            and self.group_center_sort is not None
            and self.peaks is not None
        ):
            locs = self.position.locations_resampled
            self._migration = np.abs(
                locs[self.peaks[self.group_center_sort[-1]]]
                - locs[self.peaks[self.group_center_sort[0]]]
            )
        return self._migration

    @property
    def amplitude(self) -> float | None:
        """
        Amplitude of anomalies.
        """
        if self._amplitude is None and self.anomalies is not None:
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

    @property
    def property_group(self) -> PropertyGroup:
        """
        Channel group.
        """
        return self._property_group

    @property_group.setter
    def property_group(self, value):
        self._property_group = value

    @property
    def subgroups(self) -> set[AnomalyGroup]:
        """
        Groups merged into this group.
        """
        if len(self._subgroups) == 0:
            return {self}
        return self._subgroups

    @subgroups.setter
    def subgroups(self, value):
        self._subgroups = value

    @property
    def channels(self) -> dict:
        """
        Dict of active channels and values.
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

    @property
    def full_azimuth(self) -> np.ndarray | None:
        """
        Full azimuth values for line.
        """
        return self._full_azimuth

    @full_azimuth.setter
    def full_azimuth(self, value):
        self._full_azimuth = value

    @property
    def full_peak_values(self) -> np.ndarray | None:
        """
        Full peak values for group.
        """
        return self._full_peak_values

    @full_peak_values.setter
    def full_peak_values(self, value):
        self._full_peak_values = value

    @property
    def peaks(self) -> np.ndarray | None:
        """
        List of peaks from all anomalies in group.
        """
        if self._peaks is None:
            self._peaks = self.get_list_attr("peak")
        return self._peaks

    @property
    def start(self) -> int | None:
        """
        Start position of the anomaly group.
        """
        if self._start is None and self.peaks is not None:
            self._start = np.min(self.get_list_attr("start"))
        return self._start

    @property
    def end(self) -> int | None:
        """
        End position of the anomaly group.
        """
        if self._end is None and self.peaks is not None:
            self._end = np.max(self.get_list_attr("end"))
        return self._end

    def get_list_attr(self, attr: str) -> list | np.ndarray:
        """
        Get list of attribute from anomalies.

        :param attr: Attribute to get.

        :return: List of attribute.
        """
        return np.array([getattr(a, attr) for a in self.anomalies])

    def compute_dip_direction(
        self,
    ) -> float | None:
        """
        Compute dip direction for an anomaly group.

        :return: Dip direction.
        """
        if (
            self.group_center is None
            or self.group_center_sort is None
            or self.full_azimuth is None
            or self.peaks is None
            or self.full_peak_values is None
        ):
            return None

        dip_direction = self.full_azimuth[self.peaks[0]]

        if (
            self.full_peak_values[self.group_center_sort][0]
            < self.full_peak_values[self.group_center_sort][-1]
        ):
            dip_direction = (dip_direction + 180) % 360.0

        return dip_direction

    def compute_skew(
        self,
    ) -> float | None:
        """
        Compute skew factor for an anomaly group.

        :return: Skew.
        """
        if (
            self.group_center is None
            or self.group_center_sort is None
            or self.full_azimuth is None
            or self.peaks is None
        ):
            return None

        locs = self.position.locations_resampled
        azimuth_near = self.full_azimuth[self.peaks]

        inflect_up = self.get_list_attr("inflect_up")
        inflect_down = self.get_list_attr("inflect_down")

        skew = (
            locs[self.peaks][self.group_center_sort[0]]
            - locs[inflect_up][self.group_center_sort]
        ) / (
            locs[inflect_down][self.group_center_sort]
            - locs[self.peaks][self.group_center_sort[0]]
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

        return np.mean(skew)

    def compute_linear_fit(
        self,
    ) -> list[float] | None:
        """
        Compute linear fit for the anomaly group.

        :return: List of intercept, slope for the linear fit.
        """
        if (
            self.channels is None
            or self.anomalies is None
            or self.full_peak_values is None
        ):
            return None

        gates = np.array([a.parent.data_entity for a in self.anomalies])

        times = [
            channel["time"]
            for i, channel in enumerate(self.channels.values())
            if (i in list(gates) and "time" in channel)
        ]

        linear_fit = None
        if len(times) > 2 and len(self.anomalies) > 0:
            times = np.hstack(times)[self.full_peak_values > 0]
            if len(times) > 2:
                # Compute linear trend
                slope, intercept = np.polyfit(
                    times, np.log(self.full_peak_values[self.full_peak_values > 0]), 1
                )
                linear_fit = [intercept, slope]

        return linear_fit
