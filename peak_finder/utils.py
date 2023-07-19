#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app project.
#
#  All rights reserved.
#
# pylint: disable=C0302

from __future__ import annotations

from uuid import UUID

import numpy as np
from geoh5py.groups import PropertyGroup
from geoh5py.objects import Curve
from scipy.interpolate import interp1d

from peak_finder.base.surveys import traveling_salesman
from peak_finder.base.utils import running_mean


def default_groups_from_property_group(  # pylint: disable=R0914
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
        values: np.ndarray | None = None,
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
            and self.values_resampled is not None
        ):
            self._sampling_width = int(np.floor(len(self.values_resampled)))

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
        self.values_resampled = None
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
    def values(self) -> np.ndarray:
        """
        Original values sorted along line.
        """
        return self._values

    @values.setter
    def values(self, values):
        self.values_resampled = None
        self._values = None
        if values is not None and self.sorting is not None:
            self._values = values[self.sorting]

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
    def values_resampled(self) -> np.ndarray:
        """
        Values re-sampled on a regular interval.
        """
        if self._values_resampled is None and self.locations is not None:
            interp = interp1d(self.locations, self.values, fill_value="extrapolate")
            self._values_resampled = interp(self._locations_resampled)
            if self._values_resampled is not None:
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
    def values_resampled_raw(self) -> np.ndarray:
        """
        Resampled values prior to smoothing.
        """
        return self._values_resampled_raw

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
            self.values_resampled = None

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
            self.values_resampled = None

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
            ) / self.sampling
            deriv = np.r_[
                2 * deriv[0] - deriv[1], deriv  # pylint: disable=unsubscriptable-object
            ]

        return deriv

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


class Anomaly:
    """
    Anomaly class. Contains indices of maxima, minima, inflection points.
    """

    def __init__(  # pylint: disable=R0913
        self,
        channel: int,
        start: int,
        end: int,
        inflect_up: int,
        inflect_down: int,
        peak: int,
        peak_values: list,
        amplitude: float,
        group: int,
        channel_group: list,
    ):
        self._channel = channel
        self._start = start
        self._end = end
        self._inflect_up = inflect_up
        self._inflect_down = inflect_down
        self._peak = peak
        self._peak_values = peak_values
        self._amplitude = amplitude
        self._group = group
        self._channel_group = channel_group

        self._azimuth: float | None = None

    @property
    def channel(self) -> int:
        """
        Channel.
        """
        return self._channel

    @channel.setter
    def channel(self, value):
        self._channel = value

    @property
    def start(self) -> int:
        """
        Index of start of anomaly.
        """
        return self._start

    @start.setter
    def start(self, value):
        self._start = value

    @property
    def end(self) -> int:
        """
        Index of end of anomaly.
        """
        return self._end

    @end.setter
    def end(self, value):
        self._end = value

    @property
    def inflect_up(self) -> int:
        """
        Index of upward inflection points of anomaly.
        """
        return self._inflect_up

    @inflect_up.setter
    def inflect_up(self, value):
        self._inflect_up = value

    @property
    def inflect_down(self) -> int:
        """
        Index of downward inflection points of anomaly.
        """
        return self._inflect_down

    @inflect_down.setter
    def inflect_down(self, value):
        self._inflect_down = value

    @property
    def peak(self) -> int:
        """
        Index of peak of anomaly.
        """
        return self._peak

    @peak.setter
    def peak(self, value):
        self._peak = value

    @property
    def peak_values(self) -> list:
        """
        Values of peak of anomaly.
        """
        return self._peak_values

    @peak_values.setter
    def peak_values(self, value):
        self._peak_values = value

    @property
    def amplitude(self) -> float:
        """
        Amplitude of anomaly.
        """
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value):
        self._amplitude = value

    @property
    def group(self) -> int:
        """
        Group id of anomaly.
        """
        return self._group

    @group.setter
    def group(self, value):
        self._group = value

    @property
    def channel_group(self) -> np.ndarray | list:
        """
        Channel groups.
        """
        return self._channel_group

    @channel_group.setter
    def channel_group(self, value):
        self._channel_group = value

    @property
    def azimuth(self) -> float | None:
        """
        Azimuth of anomaly.
        """
        return self._azimuth

    @azimuth.setter
    def azimuth(self, value):
        self._azimuth = value


class LineData:  # pylint: disable=R0902
    """
    Contains full list of Anomaly objects.
    """

    def __init__(
        self,
        position: LinePosition,
        min_amplitude: int,
        min_width: float,
        max_migration: float,
    ):
        self._position = position
        self._min_amplitude = min_amplitude
        self._min_width = min_width
        self._max_migration = max_migration

        self._anomalies: list[Anomaly] = []

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

    def iterate_over_peaks(  # pylint: disable=R0913, R0914
        self,
        channel: int,
        channel_groups: dict,
        uid: UUID,
        peaks_inds: np.ndarray,
        lows_inds: np.ndarray,
        inflect_up_inds: np.ndarray,
        inflect_down_inds: np.ndarray,
        locs: np.ndarray,
    ):
        """
        Iterate over peaks and add to anomalies.

        :param channel: Channel.
        :param channel_groups: Channel groups.
        :param uid: Channel uid.
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
            values = self.position.values_resampled
            delta_amp, delta_x, amplitude = self.get_amplitude_and_width(
                locs, values, peak, start, end
            )

            if (delta_amp > self.min_amplitude) & (delta_x > self.min_width):
                new_anomaly = Anomaly(
                    channel=channel,
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
                        if uid in channel_group["properties"]
                    ],
                )
                self.anomalies.append(new_anomaly)  # pylint: disable=no-member

    def get_near_peaks(
        self,
        anomaly: Anomaly,
        locs: np.ndarray,
        full_group: np.ndarray,
        full_channel: np.ndarray,
        peaks_position: np.ndarray,
    ) -> np.ndarray:
        """
        Get indices of peaks within the migration distance.

        :param anomaly: Anomaly.
        :param locs: Line vertices.
        :param full_group: Array of group ids.
        :param full_channel: Array of channels.
        :param peaks_position: Array of peak positions.

        :return: Array of indices of peaks within the migration distance.
        """
        dist = np.abs(locs[anomaly.peak] - peaks_position)
        # Find anomalies across channels within horizontal range
        near = np.where((dist < self.max_migration) & (full_group == -1))[0]

        # Reject from group if channel gap > 1
        u_gates, u_count = np.unique(full_channel[near], return_counts=True)
        if len(u_gates) > 1 and np.any((u_gates[1:] - u_gates[:-1]) > 2):
            cutoff = u_gates[np.where((u_gates[1:] - u_gates[:-1]) > 2)[0][0]]
            near = near[full_channel[near] <= cutoff]  # Remove after cutoff

        # Check for multiple nearest peaks on single channel
        # and keep the nearest
        u_gates, u_count = np.unique(full_channel[near], return_counts=True)
        for gate in u_gates[np.where(u_count > 1)]:
            mask = np.ones_like(near, dtype="bool")
            sub_ind = full_channel[near] == gate
            sub_ind[np.where(sub_ind)[0][np.argmin(dist[near][sub_ind])]] = False
            mask[sub_ind] = False
            near = near[mask]

        return near

    def group_anomalies(  # pylint: disable=R0913, R0914
        self,
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
        locs = self.position.locations_resampled

        peak_inds = self.get_list_attr("peak")
        peaks_position = locs[peak_inds]

        full_group = self.get_list_attr("group")
        full_channel = self.get_list_attr("channel")

        groups = []
        group_id = -1

        for anom in self.anomalies:
            # Skip if already labeled
            if anom.group != -1:
                continue
            group_id += 1

            # Find nearest peaks
            near = self.get_near_peaks(
                anom,
                locs,
                full_group,
                full_channel,
                peaks_position,
            )

            score = np.zeros(len(channel_groups))
            for ind in near:
                channel_groups_list = self.get_list_attr("channel_group")
                score[channel_groups_list[ind]] += 1

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
                data_uid[self.anomalies[ind].channel] in channel_group["properties"]
                for ind in near
            ]
            near = near[mask, ...]
            if len(near) == 0:
                continue

            # Update group_id on anomalies
            full_group[near] = group_id
            for ind in near:
                self.anomalies[ind].group = group_id

            # Make AnomalyGroup
            near_anomalies = np.array(self.anomalies)[near]
            group = AnomalyGroup(near_anomalies, channel_group)

            peaks = group.get_list_attr("peak")
            cox_sort = np.argsort(locs[peaks])
            azimuth_near = azimuth[peaks]
            group.azimuth = group.compute_dip_direction(azimuth, peaks, cox_sort)
            group.migration = np.abs(
                locs[peaks[cox_sort[-1]]] - locs[peaks[cox_sort[0]]]
            )
            group.skew = group.compute_skew(
                locs,
                peaks,
                cox_sort,
                azimuth_near,
            )

            # Normalize peak values
            for group_anom in group.anomalies:
                group_anom.peak_values *= np.prod(data_normalization)

            group.amplitude = np.sum([anom.amplitude for anom in group.anomalies])
            group.linear_fit = group.compute_linear_fit(channels)
            group.cox = (
                np.mean(
                    self.position.interpolate_array(peaks[cox_sort[0]]),
                    axis=0,
                ),
            )

            if minimal_output:
                group.minimal_output(self.position)

            groups += [group]

        return groups


class AnomalyGroup:  # pylint: disable=R0902
    """
    Group of anomalies. Contains list with a subset of anomalies.
    """

    def __init__(  # pylint: disable=R0913
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


class LineGroup:
    """
    Contains list of AnomalyGroup objects.
    """

    def __init__(self, dataset: LineData, groups: list[AnomalyGroup]):
        """
        :param dataset: LineData with list of all anomalies.
        :param groups: List of anomaly groups.
        """
        self._groups = groups
        self._dataset = dataset

    @property
    def groups(self) -> list[AnomalyGroup]:
        """
        List of anomaly groups.
        """
        return self._groups

    @groups.setter
    def groups(self, value):
        self._groups = value

    @property
    def dataset(self) -> LineData:
        """
        All anomalies.
        """
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value


class LineAnomaly:
    """
    Contains list of LineGroup objects.
    """

    def __init__(
        self,
        entity: Curve,
    ):
        """
        :param entity: Survey object.
        """
        self._entity = entity
        self._property_group: PropertyGroup | None = None
        self._position: LinePosition | None = None
        self._anomalies: list[LineGroup] = []

        self.locations = self.entity.vertices

    @property
    def entity(self) -> Curve:
        """
        Survey object.
        """
        return self._entity

    @entity.setter
    def entity(self, value):
        self._entity = value

    @property
    def locations(self) -> np.ndarray | None:
        """
        Survey vertices.
        """
        return self._locations

    @locations.setter
    def locations(self, value):
        self._locations = value

    @property
    def anomalies(self) -> list[LineGroup] | None:
        """
        List of line groups.
        """
        return self._anomalies

    @anomalies.setter
    def anomalies(self, value):
        self._anomalies = value

    @property
    def position(self) -> LinePosition | None:
        """
        Line position and interpolation.
        """
        return self._position

    @position.setter
    def position(self, value):
        self._position = value

    def find_anomalies(  # pylint: disable=R0912, R0913, R0914, R0915 # noqa: C901
        self,
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
    ) -> tuple[LineGroup | None, LinePosition] | LineGroup | None:
        """
        Find all anomalies along a line profile of data.
        Anomalies are detected based on the lows, inflection points and peaks.
        Neighbouring anomalies are then grouped and assigned a channel_group label.

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
        if self.locations is None:
            return None

        self.position = LinePosition(
            locations=self.locations[line_indices],
            smoothing=smoothing,
            residual=use_residual,
        )
        locs = self.position.locations_resampled
        if locs is None:
            return None

        if data_normalization == "ppm":
            data_normalization = [1e-6]

        # Compute azimuth
        mat = np.c_[self.position.interp_x(locs), self.position.interp_y(locs)]
        angles = np.arctan2(mat[1:, 1] - mat[:-1, 1], mat[1:, 0] - mat[:-1, 0])
        angles = np.r_[angles[0], angles].tolist()
        azimuth = (450.0 - np.rad2deg(running_mean(angles, width=5))) % 360.0

        line_data = LineData(self.position, min_amplitude, min_width, max_migration)

        # Used in group_anomalies
        data_uid = list(channels)
        property_groups = list(channel_groups.values())
        group_prop_size = np.r_[
            [len(grp["properties"]) for grp in channel_groups.values()]
        ]

        # Iterate over channels and add to anomalies
        for channel, (uid, params) in enumerate(channels.items()):
            if "values" not in list(params):
                continue
            # Update profile with current line values
            self.position.values = params["values"][line_indices].copy()

            # Get indices for peaks and inflection points for line
            peaks, lows, inflect_up, inflect_down = self.position.get_peak_indices(
                min_value
            )
            # Iterate over peaks and add to anomalies
            line_data.iterate_over_peaks(
                channel,
                channel_groups,
                uid,
                peaks,
                lows,
                inflect_up,
                inflect_down,
                locs,
            )
        line_data.anomalies = np.array(line_data.anomalies)
        if len(line_data.anomalies) == 0:
            line_group = None
        else:
            # Group anomalies
            groups = line_data.group_anomalies(
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
            line_group = LineGroup(line_data, groups)

        if return_profile:
            return line_group, self.position
        return line_group
