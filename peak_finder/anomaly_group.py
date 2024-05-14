#  Copyright (c) 2024 Mira Geoscience Ltd.
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


class AnomalyGroup:
    """
    Group of anomalies. Contains list with a subset of anomalies.
    """

    def __init__(
        self,
        anomalies: list[Anomaly],
        property_group: PropertyGroup,
        subgroups: set[AnomalyGroup],
    ):
        self._amplitude: float | None = None
        self._group_center: np.ndarray | None = None
        self._group_center_sort: np.ndarray | None = None
        self._peaks: np.ndarray | None = None
        self._start: int | None = None
        self._end: int | None = None

        self.anomalies = anomalies
        self.property_group = property_group
        self.subgroups = subgroups

    @property
    def amplitude(self) -> float | None:
        """
        Amplitude of anomalies.
        """
        if self._amplitude is None and self.anomalies is not None:
            self._amplitude = np.sum([anom.amplitude for anom in self.anomalies])
        return self._amplitude

    @property
    def position(self) -> LinePosition:
        """
        Line position.
        """
        return self.anomalies[0].parent.position

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
    def peaks(self) -> np.ndarray:
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
            self._start = np.median(self.get_list_attr("start"))
        return self._start

    @property
    def end(self) -> int | None:
        """
        End position of the anomaly group.
        """
        if self._end is None and self.peaks is not None:
            self._end = np.median(self.get_list_attr("end"))
        return self._end

    def get_list_attr(self, attr: str) -> np.ndarray:
        """
        Get list of attribute from anomalies.

        :param attr: Attribute to get.

        :return: List of attribute.
        """
        return np.asarray([getattr(a, attr) for a in self.anomalies])
