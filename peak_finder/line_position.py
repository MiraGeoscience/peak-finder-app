#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app project.
#
#  All rights reserved.
#

from __future__ import annotations

import numpy as np
from geoapps_utils.utils.numerical import running_mean
from scipy.interpolate import interp1d


class LinePosition:  # pylint: disable=R0902
    """
    Compute and store the derivatives of inline data values. The values are re-sampled at a constant
    interval, padded then transformed to the Fourier domain using the :obj:`numpy.fft` package.

    :param locations: An array of data locations, either as distance along line or 3D coordinates.
        For 3D coordinates, the locations are automatically converted and sorted as distance from
        the origin.
    :param values: Data values used to compute derivatives over, shape(locations.shape[0],).
    :param smoothing: Number of neighbours used by the :obj:`geoapps.utils.running_mean` routine.
    :param residual: Use the residual between the values and the running mean to compute
        derivatives.
    :param sampling: Sampling interval length (m) used in the FFT. Defaults to the mean data
        separation.
    """

    def __init__(  # pylint: disable=R0913
        self,
        *,
        locations: np.ndarray,
        line_indices: np.ndarray,
        line_start: np.ndarray,
        sorting: np.ndarray,
        smoothing: int = 0,
        residual: bool = False,
        **kwargs,
    ):
        self._locations: np.ndarray
        self._locations_resampled: np.ndarray
        self._sampling: float
        self._sampling_width: int
        self._map_locations = None
        self.line_indices = line_indices
        self.line_start = line_start
        self.sorting = sorting
        self.locations = locations
        self._smoothing = smoothing
        self._residual = residual
        self._x_interp: interp1d | None = None
        self._y_interp: interp1d | None = None
        self._z_interp: interp1d | None = None

        for key, value in kwargs.items():
            if getattr(self, key, None) is not None:
                setattr(self, key, value)

    @property
    def locations(self) -> np.ndarray:
        """
        Position of values along line.
        """
        return self._locations

    @locations.setter
    def locations(self, locations: np.ndarray):
        self.y_locations = None
        self.z_locations = None

        if locations is None or len(locations) < 2:
            raise ValueError("Locations must be an array of at least 2 points.")

        if locations.ndim > 1:
            self.x_locations = locations[self.sorting, 0]
            self.y_locations = locations[self.sorting, 1]

            if locations.shape[1] == 3:
                self.z_locations = locations[self.sorting, 2]

            xy_locations = self.line_start[None, :2] - locations[self.sorting, :2]

            distances = np.linalg.norm(xy_locations, axis=1)
        else:
            self.x_locations = locations
            distances = locations[self.sorting]

        self._locations = distances

        if self._locations[0] == self._locations[-1]:
            raise ValueError("Locations must be unique.")

        dx = np.mean(  # pylint: disable=C0103
            np.abs(self.locations[1:] - self.locations[:-1])
        )
        self._sampling_width = np.abs(
            np.ceil((self._locations[-1] - self._locations[0]) / dx).astype(int)
        )
        self._locations_resampled = np.linspace(
            distances[0], distances[-1], self._sampling_width
        )
        self._sampling = np.mean(
            np.abs(self.locations_resampled[1:] - self.locations_resampled[:-1])
        )

    @property
    def line_indices(self) -> np.ndarray:
        """
        Indices for current line
        """
        return self._line_indices

    @line_indices.setter
    def line_indices(self, value):
        self._line_indices = value

    @property
    def line_start(self) -> np.ndarray:
        """
        Start index for current line
        """
        return self._line_start

    @line_start.setter
    def line_start(self, value: np.ndarray):
        if len(value) < 2:
            raise ValueError("Line start must be an array of at least two values.")

        self._line_start = np.asarray(value)

    @property
    def sorting(self) -> np.ndarray:
        """
        Locations sorting order.
        """
        return self._sorting

    @sorting.setter
    def sorting(self, value):
        if not isinstance(value, np.ndarray):
            raise ValueError("Sorting must be a numpy array.")

        self._sorting = value

    @property
    def map_locations(self) -> np.ndarray:
        """
        A list where the indices are the resampled locations indices and the values are the
        original locations indices.
        """
        if self._map_locations is None:
            locs = self.locations
            locs_resampled = self.locations_resampled
            indices = abs(locs_resampled[:, None] - locs).argmin(axis=1)
            self._map_locations = indices
        return self._map_locations

    @property
    def locations_resampled(self) -> np.ndarray:
        """
        Position of values resampled on a fix interval.
        """
        return self._locations_resampled

    @property
    def sampling(self) -> float:
        """
        Discrete interval length (m)
        """
        return self._sampling

    @property
    def residual(self) -> bool:
        """
        Use the residual of the smoothing data.
        """
        return self._residual

    @residual.setter
    def residual(self, value):
        assert isinstance(value, bool), "Residual must be a bool"

        self._residual = value

    @property
    def smoothing(self) -> int:
        """
        Smoothing factor in terms of number of nearest neighbours used
        in a running mean averaging of the signal.
        """
        return self._smoothing

    @smoothing.setter
    def smoothing(self, value):
        assert (
            isinstance(value, int) and value >= 0
        ), "Smoothing parameter must be an integer >0"
        if value != self._smoothing:
            self._smoothing = value

    def resample_values(self, values) -> tuple[np.ndarray, np.ndarray]:
        """
        Values re-sampled on a regular interval.
        """
        interp = interp1d(self.locations, values, fill_value="extrapolate")
        values_resampled = interp(self._locations_resampled)

        if self._smoothing > 0:
            mean_values = running_mean(
                values_resampled,
                width=self._smoothing,
                method="centered",
            )

            if self.residual:
                return values_resampled - mean_values, values_resampled

            return mean_values, values_resampled

        return values_resampled, values_resampled

    def interp_x(self, distance: np.ndarray) -> np.ndarray:
        """
        Get the x-coordinate from the inline distance.

        :param distance: Inline distance.

        :return: x-coordinate.
        """
        if self._x_interp is None:
            self._x_interp = interp1d(
                self.locations,
                self.x_locations,
                bounds_error=False,
                fill_value="extrapolate",
            )
        return self._x_interp(distance)

    def interp_y(self, distance: np.ndarray) -> np.ndarray | None:
        """
        Get the y-coordinate from the inline distance.

        :param distance: Inline distance.

        :return: y-coordinate.
        """
        if self._y_interp is None and self.y_locations is not None:
            self._y_interp = interp1d(
                self.locations,
                self.y_locations,
                bounds_error=False,
                fill_value="extrapolate",
            )

        if self._y_interp is None:
            return None

        return self._y_interp(distance)

    def interp_z(self, distance: float) -> float | None:
        """
        Get the z-coordinate from the inline distance.

        :param distance: Inline distance.

        :return: z-coordinate.
        """
        if self._z_interp is None and self.z_locations is not None:
            self._z_interp = interp1d(
                self.locations,
                self.z_locations,
                bounds_error=False,
                fill_value="extrapolate",
            )

        if self._z_interp is None:
            return None

        return self._z_interp(distance)

    def interpolate_array(self, inds: np.ndarray) -> np.ndarray:
        """
        Interpolate the locations of the line profile at the given indices.

        :param inds: Indices of locations to interpolate.

        :return: Interpolated locations.
        """
        return np.c_[
            self.interp_x(self.locations_resampled[inds]),
            self.interp_y(self.locations_resampled[inds]),
            self.interp_z(self.locations_resampled[inds]),
        ]

    def compute_azimuth(self) -> np.ndarray:
        """
        Compute azimuth of line profile.
        """
        locs = self.locations_resampled
        mat = np.c_[self.interp_x(locs), self.interp_y(locs)]
        angles = np.arctan2(mat[1:, 1] - mat[:-1, 1], mat[1:, 0] - mat[:-1, 0])
        angles = np.r_[angles[0], angles].tolist()
        azimuth = (450.0 - np.rad2deg(running_mean(angles, width=5))) % 360.0
        return azimuth
