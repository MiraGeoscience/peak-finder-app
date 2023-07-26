#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app project.
#
#  All rights reserved.
#

from __future__ import annotations


class Anomaly:
    """
    Anomaly class. Contains indices of maxima, minima, inflection points.
    """

    def __init__(
        self,
        channel: int,
        start: int,
        end: int,
        inflect_up: int,
        inflect_down: int,
        peak: int,
        group: int,
    ):
        self._channel = channel
        self._start = start
        self._end = end
        self._inflect_up = inflect_up
        self._inflect_down = inflect_down
        self._peak = peak
        self._group = group

        self._amplitude: float | None = None

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
