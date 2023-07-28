#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app project.
#
#  All rights reserved.
#

from __future__ import annotations

import numpy as np
from geoapps_utils.numerical import running_mean, traveling_salesman
from scipy.interpolate import interp1d


class LinePosition:  # pylint: disable=R0902
    """
    Compute and store the derivatives of inline data values. The values are re-sampled at a constant
    interval, padded then transformed to the Fourier domain using the :obj:`numpy.fft` package.

    :param locations: An array of data locations, either as distance along line or 3D coordinates.
        For 3D coordinates, the locations are automatically converted and sorted as distance from
        the origin.
    :param values: Data values used to compute derivatives over, shape(locations.shape[0],).
    :param epsilon: Adjustable constant used in :obj:`scipy.interpolate.Rbf`. Defaults to 20x the
        average sampling
    :param interpolation: Type on interpolation accepted by the :obj:`scipy.interpolate.Rbf`
        routine: 'multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate'
    :param smoothing: Number of neighbours used by the :obj:`geoapps.utils.running_mean` routine.
    :param residual: Use the residual between the values and the running mean to compute
        derivatives.
    :param sampling: Sampling interval length (m) used in the FFT. Defaults to the mean data
        separation.
    """

    def __init__(  # pylint: disable=R0913
        self,
        locations: np.ndarray | None = None,
        epsilon: float | None = None,
        interpolation: str = "gaussian",
        smoothing: int = 0,
        residual: bool = False,
        sampling: float | None = None,
        **kwargs,
    ):
        self._locations_resampled = None
        self._epsilon = epsilon
        self.x_locations = None
        self.y_locations = None
        self.z_locations = None
        self.locations = locations
        self._interpolation = interpolation
        self._smoothing = smoothing
        self._residual = residual
        self._sampling = sampling
        self.Fx = None  # pylint: disable=C0103
        self.Fy = None  # pylint: disable=C0103
        self.Fz = None  # pylint: disable=C0103

        for key, value in kwargs.items():
            if getattr(self, key, None) is not None:
                setattr(self, key, value)

    @property
    def epsilon(self) -> float | None:
        """
        Adjustable constant used by :obj:`scipy.interpolate.Rbf`
        """
        if getattr(self, "_epsilon", None) is None and self.locations is not None:
            width = self.locations[-1] - self.locations[0]
            self._epsilon = width / 5.0

        return self._epsilon

    @property
    def sampling_width(self) -> int:
        """
        Number of padding cells added for the FFT
        """
        if (
            getattr(self, "_sampling_width", None) is None
            and self.locations_resampled is not None
        ):
            self._sampling_width = int(np.floor(len(self.locations_resampled)))

        return self._sampling_width

    @property
    def locations(self) -> np.ndarray:
        """
        Position of values along line.
        """
        return self._locations

    @locations.setter
    def locations(self, locations):
        self._locations = None
        self.x_locations = None
        self.y_locations = None
        self.z_locations = None
        self.sorting = None
        # self.values_resampled = None
        self._locations_resampled = None

        if locations is not None:
            self.sorting = traveling_salesman(locations)
            if np.all(np.diff(self.sorting) < 0):
                self.sorting = np.flip(self.sorting)
            if locations.ndim > 1:
                if np.std(locations[:, 1]) > np.std(locations[:, 0]):
                    start = np.argmin(locations[:, 1])
                else:
                    start = np.argmin(locations[:, 0])
                self.x_locations = locations[self.sorting, 0]
                self.y_locations = locations[self.sorting, 1]

                if locations.shape[1] == 3:
                    self.z_locations = locations[self.sorting, 2]

                distances = np.linalg.norm(
                    np.c_[
                        locations[start, 0] - locations[self.sorting, 0],
                        locations[start, 1] - locations[self.sorting, 1],
                    ],
                    axis=1,
                )

            else:
                self.x_locations = locations
                distances = locations[self.sorting]

            self._locations = distances

            if self._locations[0] == self._locations[-1]:
                return

            dx = np.mean(  # pylint: disable=C0103
                np.abs(self.locations[1:] - self.locations[:-1])
            )
            self._sampling_width = np.ceil(
                (self._locations[-1] - self._locations[0]) / dx
            ).astype(int)
            self._locations_resampled = np.linspace(
                self._locations[0], self._locations[-1], self.sampling_width
            )

    @property
    def locations_resampled(self) -> np.ndarray:
        """
        Position of values resampled on a fix interval.
        """
        return self._locations_resampled

    @property
    def sampling(self) -> float | None:
        """
        Discrete interval length (m)
        """
        if (
            getattr(self, "_sampling", None) is None
            and self.locations_resampled is not None
        ):
            self._sampling = np.mean(
                np.abs(self.locations_resampled[1:] - self.locations_resampled[:-1])
            )
        return self._sampling

    @property
    def interpolation(self) -> str:
        """
        Method of interpolation: 'linear', 'nearest', 'slinear', 'quadratic' or 'cubic'
        """
        return self._interpolation

    @interpolation.setter
    def interpolation(self, method):
        methods = ["linear", "nearest", "slinear", "quadratic", "cubic"]
        assert method in methods, f"Method on interpolation must be one of {methods}"

    @property
    def residual(self) -> bool:
        """
        Use the residual of the smoothing data.
        """
        return self._residual

    @residual.setter
    def residual(self, value):
        assert isinstance(value, bool), "Residual must be a bool"
        if value != self._residual:
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

    def resample_values(self, values) -> np.ndarray:
        """
        Values re-sampled on a regular interval.
        """
        values_resampled = None
        values_resampled_raw = None
        if self.locations is not None:
            interp = interp1d(self.locations, values, fill_value="extrapolate")
            values_resampled = interp(self._locations_resampled)
            if values_resampled is not None:
                values_resampled_raw = values_resampled.copy()
            if self._smoothing > 0:
                mean_values = running_mean(
                    values_resampled,
                    width=self._smoothing,
                    method="centered",
                )

                if self.residual:
                    values_resampled = values_resampled - mean_values
                else:
                    values_resampled = mean_values

        return values_resampled, values_resampled_raw

    def interp_x(self, distance: float) -> float:
        """
        Get the x-coordinate from the inline distance.

        :param distance: Inline distance.

        :return: x-coordinate.
        """
        if getattr(self, "Fx", None) is None:
            self.Fx = interp1d(
                self.locations,
                self.x_locations,
                bounds_error=False,
                fill_value="extrapolate",
            )
        return self.Fx(distance)  # type: ignore

    def interp_y(self, distance: float) -> float:
        """
        Get the y-coordinate from the inline distance.

        :param distance: Inline distance.

        :return: y-coordinate.
        """
        if getattr(self, "Fy", None) is None:
            self.Fy = interp1d(
                self.locations,
                self.y_locations,
                bounds_error=False,
                fill_value="extrapolate",
            )
        return self.Fy(distance)  # type: ignore

    def interp_z(self, distance: float) -> float:
        """
        Get the z-coordinate from the inline distance.

        :param distance: Inline distance.

        :return: z-coordinate.
        """
        if getattr(self, "Fz", None) is None:
            self.Fz = interp1d(
                self.locations,
                self.z_locations,
                bounds_error=False,
                fill_value="extrapolate",
            )
        return self.Fz(distance)  # type: ignore

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
