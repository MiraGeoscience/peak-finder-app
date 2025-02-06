# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2025 Mira Geoscience Ltd.                                     '
#                                                                                   '
#  This file is part of peak-finder-app package.                                    '
#                                                                                   '
#  peak-finder-app is distributed under the terms and conditions of the MIT License '
#  (see LICENSE file at the root of this source code package).                      '
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from pathlib import Path

import numpy as np
from geoh5py import Workspace
from geoh5py.objects import Curve

from peak_finder.line_group import LineGroup
from peak_finder.utils import get_ordered_survey_lines


def test_get_ordered_survey_lines(tmp_path: Path):
    # Create line field with non-adjacent labels
    h5file_path = tmp_path / r"testOrderingSurveyLines.geoh5"
    # Create temp workspace
    temp_ws = Workspace.create(h5file_path)

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


def test_accumulate_groups():
    neighbourhood = np.vstack(
        [
            [0, 2],
            [0, 1],
            [1, 3],
            [1, 4],
            [2, 3],
            [3, 4],
        ]
    )

    paths = LineGroup.accumulate_groups(
        neighbourhood[0][np.newaxis, :], neighbourhood, 2
    )
    assert len(paths) == 1

    paths = LineGroup.accumulate_groups(
        neighbourhood[0][np.newaxis, :], neighbourhood, 3
    )
    assert len(paths) == 1
    assert len(np.unique(paths)) == 3

    paths = LineGroup.accumulate_groups(
        neighbourhood[1][np.newaxis, :], neighbourhood, 3
    )
    assert len(paths) == 2
    assert len(np.unique(paths[0])) == len(np.unique(paths[1])) == 3

    paths = LineGroup.accumulate_groups(
        neighbourhood[1][np.newaxis, :], neighbourhood, 4
    )
    assert len(paths) == 2
    assert len(np.unique(paths[0])) == 4 and len(np.unique(paths[1])) == 3
