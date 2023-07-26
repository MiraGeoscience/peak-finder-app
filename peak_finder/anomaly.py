#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app project.
#
#  All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass
class Anomaly:
    """
    Anomaly class.

    Contains indices of maxima, minima, inflection points.
    """
    start: int
    end: int
    inflect_up: int
    inflect_down: int
    peak: int

    def __post_init__(self):
        for attr in ["start", "end", "inflect_up", "inflect_down", "peak"]:
            if not isinstance(getattr(self, attr), np.integer):
                raise TypeError(f"Atribute '{attr}' must be an integer.")
