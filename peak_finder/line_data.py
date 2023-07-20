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
from peak_finder.line_position import LinePosition


class LineData:
    """
    Contains full list of Anomaly objects.
    """

    def __init__(  # pylint: disable=R0913
        self,
        channel: int,
        uid: UUID,
        position: LinePosition,
        min_amplitude: int,
        min_width: float,
        max_migration: float,
    ):
        self._channel = channel
        self._uid = uid
        self._position = position
        self._min_amplitude = min_amplitude
        self._min_width = min_width
        self._max_migration = max_migration

        self._anomalies: list[Anomaly] = []

    @property
    def channel(self) -> int:
        """ """
        return self._channel

    @channel.setter
    def channel(self, value):
        self._channel = value

    @property
    def uid(self) -> UUID:
        """ """
        return self._uid

    @uid.setter
    def uid(self, value):
        self._uid = value

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

    def add_anomalies(  # pylint: disable=R0913, R0914
        self,
        values,
        channel_groups: dict,
        peaks_inds: np.ndarray,
        lows_inds: np.ndarray,
        inflect_up_inds: np.ndarray,
        inflect_down_inds: np.ndarray,
        locs: np.ndarray,
    ):
        """
        Iterate over peaks and add to anomalies.

        :param channel_groups: Channel groups.
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
                    channel=self.channel,
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
                        if self.uid in channel_group["properties"]
                    ],
                )
                self.anomalies.append(new_anomaly)  # pylint: disable=no-member
