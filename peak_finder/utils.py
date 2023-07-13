#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app project.
#
#  All rights reserved.
#

from __future__ import annotations

from uuid import UUID

import numpy as np
from geoh5py.groups import PropertyGroup
from scipy.interpolate import interp1d

from peak_finder.base.utils import running_mean


def default_groups_from_property_group(
    property_group: PropertyGroup,
    start_index: int = 0,
) -> dict:  # pylint: disable=R0914
    """
    Create default channel groups from a property group.

    :param property_group: Property group to create channel groups from.
    :param start_index: Starting index of the groups.

    :return: Dictionary of channel groups.
    """
    _default_channel_groups = {
        "early": {"label": ["early"], "color": "#0000FF", "channels": []},
        "middle": {"label": ["middle"], "color": "#FFFF00", "channels": []},
        "late": {"label": ["late"], "color": "#FF0000", "channels": []},
        "early + middle": {
            "label": ["early", "middle"],
            "color": "#00FFFF",
            "channels": [],
        },
        "early + middle + late": {
            "label": ["early", "middle", "late"],
            "color": "#008000",
            "channels": [],
        },
        "middle + late": {
            "label": ["middle", "late"],
            "color": "#FFA500",
            "channels": [],
        },
    }

    parent = property_group.parent

    data_list = [
        parent.workspace.get_entity(uid)[0] for uid in property_group.properties
    ]

    start = start_index
    end = len(data_list)
    block = int((end - start) / 3)
    ranges = {
        "early": np.arange(start, start + block).tolist(),
        "middle": np.arange(start + block, start + 2 * block).tolist(),
        "late": np.arange(start + 2 * block, end).tolist(),
    }

    channel_groups = {}
    for i, (key, default) in enumerate(_default_channel_groups.items()):
        prop_group = parent.find_or_create_property_group(name=key)
        prop_group.properties = []

        for val in default["label"]:
            for ind in ranges[val]:
                prop_group.properties += [data_list[ind].uid]

        channel_groups[prop_group.name] = {
            "data": prop_group.uid,
            "color": default["color"],
            "label": [i + 1],
            "properties": prop_group.properties,
        }

    return channel_groups


class LineDataDerivatives:  # pylint: disable=R0902
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
        values: np.array | None = None,
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
        self.values = values
        self._interpolation = interpolation
        self._smoothing = smoothing
        self._residual = residual
        self._sampling = sampling
        self._values_resampled_raw = None
        self._values_resampled = None
        self.Fx = None  # pylint: disable=C0103
        self.Fy = None  # pylint: disable=C0103
        self.Fz = None  # pylint: disable=C0103

        # if values is not None:
        #     self._values = values[self.sorting]

        for key, value in kwargs.items():
            if getattr(self, key, None) is not None:
                setattr(self, key, value)

    @property
    def epsilon(self):
        """
        Adjustable constant used by :obj:`scipy.interpolate.Rbf`
        """
        if getattr(self, "_epsilon", None) is None:
            width = self.locations[-1] - self.locations[0]
            self._epsilon = width / 5.0

        return self._epsilon

    @property
    def sampling_width(self):
        """
        Number of padding cells added for the FFT
        """
        if getattr(self, "_sampling_width", None) is None:
            self._sampling_width = int(np.floor(len(self.values_resampled)))

        return self._sampling_width

    @property
    def locations(self):
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
        self.values_resampled = None
        self._locations_resampled = None

        if locations is not None:
            if locations.ndim > 1:
                if np.std(locations[:, 1]) > np.std(locations[:, 0]):
                    start = np.argmin(locations[:, 1])
                    self.sorting = np.argsort(locations[:, 1])
                else:
                    start = np.argmin(locations[:, 0])
                    self.sorting = np.argsort(locations[:, 0])

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
                self.sorting = np.argsort(locations)
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
    def locations_resampled(self):
        """
        Position of values resampled on a fix interval
        """
        return self._locations_resampled

    @property
    def values(self):
        """
        Original values sorted along line.
        """
        return self._values

    @values.setter
    def values(self, values):
        self.values_resampled = None
        self._values = None
        if (values is not None) and (self.sorting is not None):
            self._values = values[self.sorting]

    @property
    def sampling(self):
        """
        Discrete interval length (m)
        """
        if getattr(self, "_sampling", None) is None:
            self._sampling = np.mean(
                np.abs(self.locations_resampled[1:] - self.locations_resampled[:-1])
            )
        return self._sampling

    @property
    def values_resampled(self):
        """
        Values re-sampled on a regular interval
        """
        if getattr(self, "_values_resampled", None) is None:
            # self._values_resampled = self.values_padded[self.sampling_width: -self.sampling_width]
            interp = interp1d(self.locations, self.values, fill_value="extrapolate")
            self._values_resampled = interp(self._locations_resampled)
            self._values_resampled_raw = self._values_resampled.copy()
            if self._smoothing > 0:
                mean_values = running_mean(
                    self._values_resampled, width=self._smoothing, method="centered"
                )

                if self.residual:
                    self._values_resampled = self._values_resampled - mean_values
                else:
                    self._values_resampled = mean_values

        return self._values_resampled

    @values_resampled.setter
    def values_resampled(self, values):
        self._values_resampled = values
        self._values_resampled_raw = None

    @property
    def values_resampled_raw(self):
        """
        Resampled values prior to smoothing
        """
        return self._values_resampled_raw

    @property
    def interpolation(self):
        """
        Method of interpolation: ['linear'], 'nearest', 'slinear', 'quadratic' or 'cubic'
        """
        return self._interpolation

    @interpolation.setter
    def interpolation(self, method):
        methods = ["linear", "nearest", "slinear", "quadratic", "cubic"]
        assert method in methods, f"Method on interpolation must be one of {methods}"

    @property
    def residual(self):
        """
        Use the residual of the smoothing data
        """
        return self._residual

    @residual.setter
    def residual(self, value):
        assert isinstance(value, bool), "Residual must be a bool"
        if value != self._residual:
            self._residual = value
            self.values_resampled = None

    @property
    def smoothing(self):
        """
        Smoothing factor in terms of number of nearest neighbours used
        in a running mean averaging of the signal
        """
        return self._smoothing

    @smoothing.setter
    def smoothing(self, value):
        assert (
            isinstance(value, int) and value >= 0
        ), "Smoothing parameter must be an integer >0"
        if value != self._smoothing:
            self._smoothing = value
            self.values_resampled = None

    def interp_x(self, distance):
        """
        Get the x-coordinate from the inline distance.
        """
        if getattr(self, "Fx", None) is None and self.x_locations is not None:
            self.Fx = interp1d(
                self.locations,
                self.x_locations,
                bounds_error=False,
                fill_value="extrapolate",
            )
        return self.Fx(distance)

    def interp_y(self, distance):
        """
        Get the y-coordinate from the inline distance.
        """
        if getattr(self, "Fy", None) is None and self.y_locations is not None:
            self.Fy = interp1d(
                self.locations,
                self.y_locations,
                bounds_error=False,
                fill_value="extrapolate",
            )
        return self.Fy(distance)

    def interp_z(self, distance):
        """
        Get the z-coordinate from the inline distance.
        """
        if getattr(self, "Fz", None) is None and self.z_locations is not None:
            self.Fz = interp1d(
                self.locations,
                self.z_locations,
                bounds_error=False,
                fill_value="extrapolate",
            )
        return self.Fz(distance)

    def derivative(self, order=1) -> np.ndarray:
        """
        Compute and return the first order derivative.
        """
        deriv = self.values_resampled
        for _ in range(order):
            deriv = (
                deriv[1:] - deriv[:-1]  # pylint: disable=unsubscriptable-object
            ) / self.sampling
            deriv = np.r_[
                2 * deriv[0] - deriv[1], deriv  # pylint: disable=unsubscriptable-object
            ]

        return deriv

    def interpolate_array(self, inp_array) -> np.ndarray:
        return np.c_[
            self.interp_x(self.locations_resampled[inp_array]),
            self.interp_y(self.locations_resampled[inp_array]),
            self.interp_z(self.locations_resampled[inp_array]),
        ]

    def get_peak_indices(
        self,
        min_value: float,
    ):
        """
        Get maxima and minima for a line profile.

        :param min_value: Minimum value for data.

        :return: Indices of maxima.
        :return: Indices of minima.
        :return: Indices of upward inflection points.
        :return: Indices of downward inflection points.
        """
        values = self.values_resampled
        dx = self.derivative(order=1)  # pylint: disable=C0103
        ddx = self.derivative(order=2)

        # Find maxima and minima
        peaks = np.where(
            (np.diff(np.sign(dx)) != 0)
            & (ddx[1:] < 0)
            & (values[:-1] > min_value)  # pylint: disable=unsubscriptable-object
        )[0]
        lows = np.where(
            (np.diff(np.sign(dx)) != 0)
            & (ddx[1:] > 0)
            & (values[:-1] > min_value)  # pylint: disable=unsubscriptable-object
        )[0]
        lows = np.r_[0, lows, self.locations_resampled.shape[0] - 1]
        # Find inflection points
        inflect_up = np.where(
            (np.diff(np.sign(ddx)) != 0)
            & (dx[1:] > 0)
            & (values[:-1] > min_value)  # pylint: disable=unsubscriptable-object
        )[0]
        inflect_down = np.where(
            (np.diff(np.sign(ddx)) != 0)
            & (dx[1:] < 0)
            & (values[:-1] > min_value)  # pylint: disable=unsubscriptable-object
        )[0]

        return peaks, lows, inflect_up, inflect_down


class Anomalies:
    def __init__(self):
        self._channel = []
        self._start = []
        self._end = []
        self._inflect_up = []
        self._inflect_down = []
        self._peak = []
        self._peak_values = []
        self._amplitude = []
        self._group = []
        self._channel_group = []

    @property
    def channel(self):
        """ """
        return self._channel

    @channel.setter
    def channel(self, value):
        self._channel = value

    @property
    def start(self):
        """ """
        return self._start

    @start.setter
    def start(self, value):
        self._start = value

    @property
    def end(self):
        """ """
        return self._end

    @end.setter
    def end(self, value):
        self._end = value

    @property
    def inflect_up(self):
        """ """
        return self._inflect_up

    @inflect_up.setter
    def inflect_up(self, value):
        self._inflect_up = value

    @property
    def inflect_down(self):
        """ """
        return self._inflect_down

    @inflect_down.setter
    def inflect_down(self, value):
        self._inflect_down = value

    @property
    def peak(self):
        """ """
        return self._peak

    @peak.setter
    def peak(self, value):
        self._peak = value

    @property
    def peak_values(self):
        """ """
        return self._peak_values

    @peak_values.setter
    def peak_values(self, value):
        self._peak_values = value

    @property
    def amplitude(self):
        """ """
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value):
        self._amplitude = value

    @property
    def group(self):
        """ """
        return self._group

    @group.setter
    def group(self, value):
        self._group = value

    @property
    def channel_group(self):
        """ """
        return self._channel_group

    @channel_group.setter
    def channel_group(self, value):
        self._channel_group = value

    def reformat_lists(self):
        self.channel = np.hstack(self.channel)
        self.start = np.hstack(self.start)
        self.inflect_up = np.hstack(self.inflect_up)
        self.peak = np.hstack(self.peak)
        self.peak_values = np.hstack(self.peak_values)
        self.inflect_down = np.hstack(self.inflect_down)
        self.amplitude = np.hstack(self.amplitude)
        self.end = np.hstack(self.end)
        self.group = np.hstack(self.group)

    def iterate_over_peaks(
        self,
        profile: LineDataDerivatives,
        channel: int,
        channel_groups: dict,
        uid: UUID,
        min_amplitude: int,
        min_width: float,
        peaks_inds: np.ndarray,
        lows_inds: np.ndarray,
        inflect_up_inds: np.ndarray,
        inflect_down_inds: np.ndarray,
        locs: np.ndarray,
    ):
        """
        Iterate over peaks and add to anomalies.

        :param profile: Line profile.
        :param channel: Channel.
        :param channel_groups: Channel groups.
        :param uid: Channel uid.
        :param min_amplitude: Minimum amplitude.
        :param min_width: Minimum width.
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
            ind = np.median(
                [
                    0,
                    lows_inds.shape[0] - 1,
                    np.searchsorted(locs[lows_inds], locs[peak]) - 1,
                ]
            ).astype(int)
            start = lows_inds[ind]

            # Get end of peak
            ind = np.median(
                [
                    0,
                    lows_inds.shape[0] - 1,
                    np.searchsorted(locs[lows_inds], locs[peak]),
                ]
            ).astype(int)
            end = np.min([locs.shape[0] - 1, lows_inds[ind]])

            # Inflection points
            ind = np.median(
                [
                    0,
                    inflect_up_inds.shape[0] - 1,
                    np.searchsorted(locs[inflect_up_inds], locs[peak]) - 1,
                ]
            ).astype(int)
            inflect_up = inflect_up_inds[ind]

            ind = np.median(
                [
                    0,
                    inflect_down_inds.shape[0] - 1,
                    np.searchsorted(locs[inflect_down_inds], locs[peak]),
                ]
            ).astype(int)
            inflect_down = np.min([locs.shape[0] - 1, inflect_down_inds[ind] + 1])

            # Check amplitude and width thresholds
            values = profile.values_resampled
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
            delta_x = locs[end] - locs[start]
            amplitude = (
                np.sum(
                    np.abs(values[start:end])  # pylint: disable=unsubscriptable-object
                )
                * profile.sampling
            )

            if (delta_amp > min_amplitude) & (delta_x > min_width):
                self.channel.append(channel)
                self.start.append(start)
                self.end.append(end)
                self.inflect_up.append(inflect_up)
                self.inflect_down.append(inflect_down)
                self.peak.append(peak)
                self.peak_values.append(
                    values[peak]  # pylint: disable=unsubscriptable-object
                )
                self.amplitude.append(amplitude)
                self.group.append(-1)
                self.channel_group.append(
                    [
                        key
                        for key, channel_group in enumerate(channel_groups.values())
                        if uid in channel_group["properties"]
                    ]
                )


class AnomalyGroup:
    _skew = None

    def __init__(
        self,
        channels,
        start,
        end,
        inflect_up,
        inflect_down,
        peak,
        cox,
        azimuth,
        migration,
        amplitude,
        channel_group,
        linear_fit,
    ):
        self._channels = channels
        self._start = start
        self._end = end
        self._inflect_up = inflect_up
        self._inflect_down = inflect_down
        self._peak = peak
        self._cox = cox
        self._azimuth = azimuth
        self._migration = migration
        self._amplitude = amplitude
        self._channel_group = channel_group
        self._linear_fit = linear_fit

    @property
    def channels(self):
        """ """
        return self._channels

    @channels.setter
    def channels(self, value):
        self._channels = value

    @property
    def start(self):
        """ """
        return self._start

    @start.setter
    def start(self, value):
        self._start = value

    @property
    def end(self):
        """ """
        return self._end

    @end.setter
    def end(self, value):
        self._end = value

    @property
    def inflect_up(self):
        """ """
        return self._inflect_up

    @inflect_up.setter
    def inflect_up(self, value):
        self._inflect_up = value

    @property
    def inflect_down(self):
        """ """
        return self._inflect_down

    @inflect_down.setter
    def inflect_down(self, value):
        self._inflect_down = value

    @property
    def peak(self):
        """ """
        return self._peak

    @peak.setter
    def peak(self, value):
        self._peak = value

    @property
    def cox(self):
        """ """
        return self._cox

    @cox.setter
    def cox(self, value):
        self._cox = value

    @property
    def azimuth(self):
        """ """
        return self._azimuth

    @azimuth.setter
    def azimuth(self, value):
        self._azimuth = value

    @property
    def migration(self):
        """ """
        return self._migration

    @migration.setter
    def migration(self, value):
        self._migration = value

    @property
    def amplitude(self):
        """ """
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value):
        self._amplitude = value

    @property
    def channel_group(self):
        """ """
        return self._channel_group

    @channel_group.setter
    def channel_group(self, value):
        self._channel_group = value

    @property
    def linear_fit(self):
        """ """
        return self._linear_fit

    @linear_fit.setter
    def linear_fit(self, value):
        self._linear_fit = value

    @property
    def skew(self):
        """ """
        return self._skew

    @skew.setter
    def skew(self, value):
        self._skew = value

    @staticmethod
    def compute_skew(
        locs: np.ndarray,
        cox: np.ndarray,
        cox_sort: np.ndarray,
        inflect_up: np.ndarray,
        inflect_down: np.ndarray,
        azimuth_near: np.ndarray,
    ):
        """
        Compute skew factor for an anomaly group.

        :param loc: Resampled line vertices.
        :param cox: Center of oxidized coal; center of peak.
        :param cox_sort: Indices to sort cox.
        :param inflect_up: Upwards inflection points for the group.
        :param inflect_down: Downwards inflection points for the group.
        :param azimuth_near: Azimuth values for the group.

        :return: Skew.
        """
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

    @staticmethod
    def compute_linear_fit(
        channels: np.ndarray,
        gates: np.ndarray,
        cox: np.ndarray,
        values: np.ndarray,
    ) -> list[float]:
        """
        Compute linear fit for the anomaly group.

        :param channels: Channels.
        :param gates: Values for anomalies channel in the anomaly group.
        :param cox: Center of oxidized coal; center of peak.
        :param values: Normalized peak values of the anomaly group.

        :return: List of intercept, slope for the linear fit.
        """
        times = [
            channel["time"]
            for i, channel in enumerate(channels.values())
            if (i in list(gates) and "time" in channel)
        ]

        linear_fit = None
        if len(times) > 2 and len(cox) > 0:
            times = np.hstack(times)[values > 0]
            if len(times) > 2:
                # Compute linear trend
                slope, intercept = np.polyfit(times, np.log(values[values > 0]), 1)
                linear_fit = [intercept, slope]

        return linear_fit


def get_near_peaks(
    ind: int,
    peaks_position: np.ndarray,
    max_migration: float,
    anomalies: Anomalies,
):
    """
    Get indices of peaks within the migration distance.

    :param ind: Index of the peak.
    :param peaks_position: Array of peak positions.
    :param max_migration: Maximum migration distance.
    :param anomalies: Anomalies object.

    :return: Array of indices of peaks within the migration distance.
    """
    dist = np.abs(peaks_position[ind] - peaks_position)
    # Find anomalies across channels within horizontal range
    near = np.where((dist < max_migration) & (anomalies.group == -1))[0]
    # Reject from group if channel gap > 1
    u_gates, u_count = np.unique(anomalies.channel[near], return_counts=True)
    if len(u_gates) > 1 and np.any((u_gates[1:] - u_gates[:-1]) > 2):
        cutoff = u_gates[np.where((u_gates[1:] - u_gates[:-1]) > 2)[0][0]]
        near = near[anomalies.channel[near] <= cutoff]  # Remove after cutoff
    # Check for multiple nearest peaks on single channel
    # and keep the nearest
    u_gates, u_count = np.unique(anomalies.channel[near], return_counts=True)
    for gate in u_gates[np.where(u_count > 1)]:
        mask = np.ones_like(near, dtype="bool")
        sub_ind = anomalies.channel[near] == gate
        sub_ind[np.where(sub_ind)[0][np.argmin(dist[near][sub_ind])]] = False
        mask[sub_ind] = False
        near = near[mask]
    return near


def group_anomalies(
    anomalies: Anomalies,
    profile: LineDataDerivatives,
    max_migration: float,
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

    :param anomalies: Anomalies object.
    :param max_migration: Maximum migration distance between anomalies.
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
    locs = profile.locations_resampled
    peaks_position = locs[anomalies.peak]
    groups = []
    group_id = -1

    for i in range(peaks_position.shape[0]):
        # Skip if already labeled
        if anomalies.group[i] != -1:
            continue
        group_id += 1

        # Find nearest peaks
        near = get_near_peaks(
            i,
            peaks_position,
            max_migration,
            anomalies,
        )

        score = np.zeros(len(channel_groups))
        for ids in near:
            score[anomalies.channel_group[ids]] += 1

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
            data_uid[anomalies.channel[id]] in channel_group["properties"]
            for id in near
        ]
        near = near[mask, ...]
        if len(near) == 0:
            continue

        anomalies.group[near] = group_id
        gates = anomalies.channel[near]
        cox = anomalies.peak[near]

        inflect_down = anomalies.inflect_down[near]
        inflect_up = anomalies.inflect_up[near]
        cox_sort = np.argsort(locs[cox])
        azimuth_near = azimuth[cox]
        dip_direction = azimuth[cox[0]]

        if (
            anomalies.peak_values[near][cox_sort][0]
            < anomalies.peak_values[near][cox_sort][-1]
        ):
            dip_direction = (dip_direction + 180) % 360.0

        migration = np.abs(locs[cox[cox_sort[-1]]] - locs[cox[cox_sort[0]]])

        skew = AnomalyGroup.compute_skew(
            locs,
            cox,
            cox_sort,
            inflect_up,
            inflect_down,
            azimuth_near,
        )

        values = anomalies.peak_values[near] * np.prod(data_normalization)
        amplitude = np.sum(anomalies.amplitude[near])

        linear_fit = AnomalyGroup.compute_linear_fit(channels, gates, cox, values)

        group = AnomalyGroup(
            channels=gates,
            start=anomalies.start[near],
            end=anomalies.end[near],
            inflect_up=anomalies.inflect_up[near],
            inflect_down=anomalies.inflect_down[near],
            peak=cox,
            cox=np.mean(
                profile.interpolate_array(cox[cox_sort[0]]),
                axis=0,
            ),
            azimuth=dip_direction,
            migration=migration,
            amplitude=amplitude,
            channel_group=channel_group,
            linear_fit=linear_fit,
        )

        if minimal_output:
            group.skew = np.mean(skew)
            group.inflect_down = profile.interpolate_array(inflect_down)
            group.inflect_up = profile.interpolate_array(inflect_up)
            group.start = profile.interpolate_array(anomalies.start[near])
            group.end = profile.interpolate_array(anomalies.end[near])
            group.peaks = profile.interpolate_array(cox)
        else:
            group.peak_values = values

        groups += [group]

    return groups


def find_anomalies(  # pylint: disable=R0912, R0913, R0914, R0915 # noqa: C901
    locations: np.ndarray,
    line_indices: np.ndarray,
    channels: dict,
    channel_groups: dict,
    smoothing: int,
    data_normalization: list | str,
    min_amplitude: int,
    min_value: float,
    min_width: float,
    max_migration: float,
    min_channels: int,
    use_residual: bool = False,
    minimal_output: bool = False,
    return_profile: bool = False,
) -> (list[AnomalyGroup] | None, LineDataDerivatives | None):
    """
    Find all anomalies along a line profile of data.
    Anomalies are detected based on the lows, inflection points and peaks.
    Neighbouring anomalies are then grouped and assigned a channel_group label.

    :param locations: Survey vertices.
    :param line_indices: Indices of vertices for line profile.
    :param channels: Channels.
    :param channel_groups: Property groups to use for grouping anomalies.
    :param smoothing: Smoothing factor.
    :param data_normalization: Value(s) to normalize data by.
    :param min_amplitude: Minimum amplitude of anomaly as percent.
    :param min_value: Minimum data value of anomaly.
    :param min_width: Minimum width of anomaly in meters.
    :param max_migration: Maximum peak migration.
    :param min_channels: Minimum number of channels in anomaly.
    :param use_residual: Whether to use the residual of the smoothing data.
    :param minimal_output: Whether to return minimal output.
    :param return_profile: Whether to return the line profile.

    :return: List of groups and line profile.
    """

    profile = LineDataDerivatives(
        locations=locations[line_indices], smoothing=smoothing, residual=use_residual
    )
    locs = profile.locations_resampled

    if data_normalization == "ppm":  # pp2t
        data_normalization = [1e-6]

    if locs is None:
        return None

    # Compute azimuth
    mat = np.c_[profile.interp_x(locs), profile.interp_y(locs)]
    angles = np.arctan2(mat[1:, 1] - mat[:-1, 1], mat[1:, 0] - mat[:-1, 0])
    angles = np.r_[angles[0], angles].tolist()
    azimuth = (450.0 - np.rad2deg(running_mean(angles, width=5))) % 360.0

    anomalies = Anomalies()
    data_uid = list(channels)
    property_groups = list(channel_groups.values())
    group_prop_size = np.r_[[len(grp["properties"]) for grp in channel_groups.values()]]
    # Iterate over channels and add to anomalies
    for chan, (uid, params) in enumerate(channels.items()):
        if "values" not in list(params):
            continue

        values = params["values"][line_indices].copy()
        profile.values = values  # Update profile with current line values

        # Get indices for peaks and inflection points for line
        peaks, lows, inflect_up, inflect_down = profile.get_peak_indices(min_value)
        # Iterate over peaks and add to anomalies
        anomalies.iterate_over_peaks(
            profile,
            chan,
            channel_groups,
            uid,
            min_amplitude,
            min_width,
            peaks,
            lows,
            inflect_up,
            inflect_down,
            locs,
        )

    if len(anomalies.peak) == 0:
        groups = None
    else:
        anomalies.reformat_lists()
        # Group anomalies
        groups = group_anomalies(
            anomalies,
            profile,
            max_migration,
            channel_groups,
            group_prop_size,
            min_channels,
            property_groups,
            data_uid,
            azimuth,
            data_normalization,
            channels,
            minimal_output,
        )

    if return_profile:
        return groups, profile
    return groups
