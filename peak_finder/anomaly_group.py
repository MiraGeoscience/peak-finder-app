#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app project.
#
#  All rights reserved.
#

# pylint: disable=too-many-instance-attributes, too-many-arguments

from __future__ import annotations

import numpy as np

from peak_finder.anomaly import Anomaly


class AnomalyGroup:
    """
    Group of anomalies. Contains list with a subset of anomalies.

    :param anomalies: List of anomalies.
    :param property_group: Channel group.
    :param subgroups: Groups merged into this group.

    :ivar amplitude: Sum of anomalies amplitudes.
    :ivar center: Center of the group.
    :ivar center_sort: Center of the group sorted.
    :ivar peaks: Peaks of the group.
    :ivar start: Start of the group.
    :ivar end: End of the group.
    """

    def __init__(
        self,
        anomalies: list[Anomaly],
        property_group: str,
        subgroups: set[AnomalyGroup],
    ):
        self.anomalies = anomalies
        self.property_group = property_group
        self.subgroups = subgroups

        self.amplitude: float
        self.center: np.ndarray
        self.center_sort: np.ndarray
        self.peaks: np.ndarray
        self.start: int
        self.end: int

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
        self.position = self.anomalies[0].parent.position
        self._compute_metrics()

    def _compute_metrics(self):
        """
        Compute metrics for the group.
        """
        self.amplitude = np.sum([anom.amplitude for anom in self.anomalies])
        self.peaks = self.get_list_attr("peak")
        self.start = np.median(self.get_list_attr("start"))
        self.end = np.median(self.get_list_attr("end"))
        self.center_sort = np.argsort(self.position.locations_resampled[self.peaks])

        interp_locs = self.position.interpolate_array(self.peaks[self.center_sort])
        self.center = np.mean(interp_locs, axis=0)

    @property
    def property_group(self) -> str:
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

    def get_list_attr(self, attr: str) -> np.ndarray:
        """
        Get list of attribute from anomalies.

        :param attr: Attribute to get.

        :return: List of attribute.
        """
        return np.asarray([getattr(a, attr) for a in self.anomalies])
