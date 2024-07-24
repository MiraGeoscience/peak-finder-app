#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app project.
#
#  All rights reserved.
#

from __future__ import annotations

import numpy as np
from geoh5py.data import FloatData

from peak_finder.anomaly import Anomaly
from peak_finder.line_position import LinePosition


class LineData:  # pylint: disable=too-many-instance-attributes
    """
    Contains full list of Anomaly objects and line data values.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        data: FloatData,
        position: LinePosition,
        min_amplitude: int,
        min_width: float,
        max_migration: float,
        min_value: float = -np.inf,
    ):
        self.data_entity = data
        self.position: LinePosition = position
        self.min_amplitude = min_amplitude
        self.min_width = min_width
        self.max_migration = max_migration
        self.min_value = min_value
        self.min_value = min_value
        self._values: np.ndarray | None = None
        self._peaks: np.ndarray | None = None
        self._lows: np.ndarray | None = None
        self._inflect_up: np.ndarray | None = None
        self._inflect_down: np.ndarray | None = None
        self._values_resampled: np.ndarray | None = None
        self._anomalies: list[Anomaly] | None = None

    @property
    def values(self) -> np.ndarray | None:
        """
        Original values sorted along line.
        """
        if self._values is None:
            if self.data_entity is not None and self.position.sorting is not None:
                self._values = self.data_entity.values[  # type: ignore
                    self.position.sorting
                ]

            if self._values is not None and len(self._values) != len(
                self.position.locations
            ):
                raise ValueError(
                    f"Number of values ({len(self._values)}) does not match "
                    f"number of locations ({self.position.locations})"
                )

        return self._values

    @property
    def data_entity(self) -> FloatData:
        """
        Data entity.
        """
        return self._data_entity

    @data_entity.setter
    def data_entity(self, data):
        """
        Data entity.
        """
        if getattr(self, "_data_entity", None) is not None:
            raise ValueError("Data entity is already set.")

        if not isinstance(data, FloatData):
            raise TypeError("Data entity must be of type geoh5py.data.FloatData.")

        self._data_entity = data

    @property
    def values_resampled(self) -> np.ndarray:
        """
        Values re-sampled on a regular interval.
        """
        if self._values_resampled is None:
            (
                self._values_resampled,
                _,
            ) = self.position.resample_values(self.values)
        return self._values_resampled

    @values_resampled.setter
    def values_resampled(self, values):
        self._values_resampled = values

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
    def min_value(self) -> float:
        """
        Min data value for anomaly.
        """
        return self._min_value

    @min_value.setter
    def min_value(self, value):
        self._min_value = value

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
        if self._anomalies is None:
            self._anomalies = self.compute()
        return self._anomalies

    @property
    def peaks(self) -> np.ndarray | None:
        """
        Find peak indices.
        """
        if self._peaks is None:
            values = self.values_resampled

            dx = self.derivative(order=1)  # pylint: disable=C0103
            ddx = self.derivative(order=2)

            self._peaks = np.where(
                (np.diff(np.sign(dx)) != 0)
                & (ddx[1:] < 0)
                & (values[:-1] > self.min_value)  # pylint: disable=unsubscriptable-object
            )[0]
        return self._peaks

    @property
    def lows(self) -> np.ndarray | None:
        """
        Find lows indices.
        """
        if self._lows is None:
            values = self.values_resampled

            dx = self.derivative(order=1)  # pylint: disable=C0103
            ddx = self.derivative(order=2)

            lows = np.where(
                (np.diff(np.sign(dx)) != 0)
                & (ddx[1:] > 0)
                & (values[:-1] >= self.min_value)  # pylint: disable=unsubscriptable-object
            )[0]
            self._lows = np.r_[0, lows, self.position.locations_resampled.shape[0] - 1]
        return self._lows

    @property
    def inflect_up(self) -> np.ndarray | None:
        """
        Find upward inflection indices.
        """
        if self._inflect_up is None:
            values = self.values_resampled

            dx = self.derivative(order=1)  # pylint: disable=C0103
            ddx = self.derivative(order=2)

            self._inflect_up = np.where(
                (np.diff(np.sign(ddx)) != 0)
                & (dx[1:] > 0)
                & (values[:-1] >= self.min_value)  # pylint: disable=unsubscriptable-object
            )[0]
        return self._inflect_up

    @property
    def inflect_down(self) -> np.ndarray | None:
        """
        Find downward inflection indices.
        """
        if self._inflect_down is None:
            values = self.values_resampled

            dx = self.derivative(order=1)  # pylint: disable=C0103
            ddx = self.derivative(order=2)

            self._inflect_down = np.where(
                (np.diff(np.sign(ddx)) != 0)
                & (dx[1:] < 0)
                & (values[:-1] >= self.min_value)  # pylint: disable=unsubscriptable-object
            )[0]
        return self._inflect_down

    def get_list_attr(self, attr: str) -> list | np.ndarray:
        """
        Get list of anomaly attributes.

        :param attr: Attribute name.

        :return: List or np.ndarray of attribute values.
        """
        return np.array([getattr(a, attr) for a in self.anomalies])

    def get_amplitude_and_width(self, anomaly: Anomaly) -> tuple[float, float, float]:
        """
        Get amplitude and width of anomaly.

        :param anomaly: Anomaly.

        :return: Amplitude and width of anomaly.
        """
        # Amplitude threshold
        locs = self.position.locations_resampled
        values = self.values_resampled
        delta_amp = (
            np.abs(
                np.min(
                    [
                        values[anomaly.peak]  # pylint: disable=unsubscriptable-object
                        - values[anomaly.start],  # pylint: disable=unsubscriptable-object
                        values[anomaly.peak]  # pylint: disable=unsubscriptable-object
                        - values[anomaly.end],  # pylint: disable=unsubscriptable-object
                    ]
                )
            )
            / (np.std(values) + 2e-32)
        ) * 100.0

        # Width threshold
        delta_x = locs[anomaly.end] - locs[anomaly.start]

        # Amplitude
        amplitude = (
            np.sum(np.abs(values[anomaly.start : anomaly.end]))  # pylint: disable=unsubscriptable-object
            * self.position.sampling
        )

        return delta_amp, delta_x, amplitude

    def derivative(self, order: int = 1) -> np.ndarray:
        """
        Compute and return the first order derivative.

        :param order: Order of derivative.

        :return: Derivative of values_resampled.
        """
        deriv = self.values_resampled
        for _ in range(order):
            deriv = (
                deriv[1:] - deriv[:-1]  # pylint: disable=unsubscriptable-object
            ) / self.position.sampling
            deriv = np.r_[
                2 * deriv[0] - deriv[1], deriv  # pylint: disable=unsubscriptable-object
            ]

        return deriv

    def get_peak_bounds(
        self,
        peak: int,
        inds: np.ndarray,
        shift: int,
    ) -> np.ndarray:
        """
        Get indices for critical points.

        :param peak: Index of peak of anomaly.
        :param inds: Indices to index locs.
        :param shift: Shift value.

        :return: Indices of critical points.
        """
        return np.median(
            [
                0,
                inds.shape[0] - 1,
                np.searchsorted(inds, peak) - shift,
            ]
        ).astype(int)

    def compute(self) -> list[Anomaly]:
        """
        Iterate over peaks and add to anomalies.

        :return: List of anomalies.
        """
        if (  # pylint: disable=R0916
            self.peaks is None
            or len(self.peaks) == 0
            or self.lows is None
            or len(self.lows) == 0
            or self.inflect_up is None
            or len(self.inflect_up) == 0
            or self.inflect_down is None
            or len(self.inflect_down) == 0
        ):
            return []

        anomalies = []
        locs = self.position.locations_resampled

        for peak in self.peaks:
            # Get start of peak
            ind = self.get_peak_bounds(peak, self.lows, 1)
            start = self.lows[ind]

            # Get end of peak
            ind = self.get_peak_bounds(peak, self.lows, 0)
            end = np.min([locs.shape[0] - 1, self.lows[ind]])

            # Inflection points
            ind = self.get_peak_bounds(peak, self.inflect_up, 1)
            inflect_up = self.inflect_up[ind]

            ind = self.get_peak_bounds(peak, self.inflect_down, 0)
            inflect_down = np.min([locs.shape[0] - 1, self.inflect_down[ind] + 1])

            new_anomaly = Anomaly(
                self,
                start,
                end,
                inflect_up,
                inflect_down,
                peak,
            )
            # Check amplitude and width thresholds
            delta_amp, delta_x, amplitude = self.get_amplitude_and_width(new_anomaly)
            if (delta_amp > self.min_amplitude) & (np.abs(delta_x) > self.min_width):
                new_anomaly.amplitude = amplitude
                anomalies.append(new_anomaly)  # pylint: disable=no-member

        return anomalies
