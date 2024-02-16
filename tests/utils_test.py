#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app project.
#
#  All rights reserved.
#
from pathlib import Path

import numpy as np
from geoh5py import Workspace
from geoh5py.objects import Curve

from peak_finder.utils import get_ordered_survey_lines


def test_get_ordered_survey_lines(tmp_path: Path):
    # Create line field with non-adjacent labels
    h5file_path = tmp_path / r"testOrderingSurveyLines.geoh5"
    # Create temp workspace
    temp_ws = Workspace(h5file_path)

    x = np.c_[
        np.zeros(30), np.tile(np.arange(0, 10), 3), np.repeat(np.arange(0, 30, 10), 10)
    ]
    parts = np.concatenate([np.ones(10) * 1, np.ones(10) * 2, np.ones(10) * 3])

    curve = Curve.create(temp_ws, vertices=x, parts=parts)
    curve.add_data({"d0": {"values": np.ones(30)}})

    line_vals = np.concatenate([np.ones(10) * 5, np.ones(10) * 2, np.ones(10) * 8])

    line = curve.add_data(
        {
            "line_id": {
                "values": line_vals,
                "value_map": {2: "123", 5: "56", 8: "33"},
                "type": "referenced",
            }
        }
    )
    curve.add_data_to_group(line, property_group="Line")

    ordered_lines = get_ordered_survey_lines(curve, line)

    assert ordered_lines == {5: "56", 2: "123", 8: "33"}
