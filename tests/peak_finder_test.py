#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app project.
#
#  All rights reserved.
#

from __future__ import annotations

import string
from pathlib import Path

import numpy as np
from geoh5py.objects import Curve
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace

from peak_finder.application import PeakFinder, PeakFinderDriver
from peak_finder.params import PeakFinderParams


def test_peak_finder_app(tmp_path: Path):  # pylint: disable=too-many-locals
    h5file_path = tmp_path / r"testPeakFinder.geoh5"
    # Create temp workspace
    temp_ws = Workspace(h5file_path)

    params = PeakFinderParams(geoh5=str(h5file_path))
    app = PeakFinder(params=params, ui_json_data={})
    app.workspace = temp_ws

    x = np.arange(-2 * np.pi + np.pi / 4, 2 * np.pi, np.pi / 32)

    curve = Curve.create(temp_ws, vertices=np.c_[x, np.zeros((x.shape[0], 2))])

    for ind in range(5):
        data = curve.add_data(
            {f"d{ind}": {"values": np.sin(x + np.pi / 8.0 * ind) - 0.1 * ind}}
        )
        curve.add_data_to_group(data, property_group="obs")

    line = curve.add_data(
        {
            "line_id": {
                "values": np.ones_like(x),
                "value_map": {1: "1", 2: "2", 3: "3"},
                "type": "referenced",
            }
        }
    )
    curve.add_data_to_group(line, property_group="Line")

    data_map = {d.name: d.uid for d in curve.children}

    early = curve.find_or_create_property_group(
        name="early", properties=[data_map["d0"]]
    )
    middle = curve.find_or_create_property_group(
        name="middle", properties=[data_map["d1"]]
    )
    late = curve.find_or_create_property_group(
        name="late", properties=[data_map["d2"], data_map["d3"], data_map["d4"]]
    )
    early_middle = curve.find_or_create_property_group(
        name="early + middle", properties=[data_map["d0"], data_map["d1"]]
    )
    early_middle_late = curve.find_or_create_property_group(
        name="early + middle + late",
        properties=[
            data_map["d0"],
            data_map["d1"],
            data_map["d2"],
            data_map["d3"],
            data_map["d4"],
        ],
    )
    middle_late = curve.find_or_create_property_group(
        name="middle + late",
        properties=[data_map["d1"], data_map["d2"], data_map["d3"], data_map["d4"]],
    )

    param_names = string.ascii_lowercase[:6]
    property_groups = {}
    for ind, group in enumerate(
        [early, middle, late, early_middle, early_middle_late, middle_late]
    ):
        property_groups[group.name] = {
            "param": param_names[ind],
            "data": str(group.uid),
            "color": "#000000",
            "label": [ind + 1],
            "properties": [str(p) for p in group.properties],
        }

    objects = "{" + str(curve.uid) + "}"
    smoothing = 6
    max_migration = 1.0
    min_channels = 1
    min_amplitude = 0
    min_value = -1.4
    min_width = 1.0
    line_field = "{" + str(line.uid) + "}"

    app.trigger_click(
        n_clicks=0,
        objects=objects,
        flip_sign=[],
        line_field=line_field,
        smoothing=smoothing,
        min_amplitude=min_amplitude,
        min_value=min_value,
        min_width=min_width,
        max_migration=max_migration,
        min_channels=min_channels,
        line_id=1,
        property_groups=property_groups,
        ga_group_name="peak_finder",
        live_link=[],
        monitoring_directory=str(tmp_path),
    )

    filename = next(tmp_path.glob("peak_finder*.geoh5"))
    with Workspace(filename) as out_ws:
        anomalies_obj = out_ws.get_entity("PointMarkers")[0]
        amplitudes = anomalies_obj.get_data("amplitude")[0].values
        assert len(amplitudes) == 13, f"Expected 13 groups. Found {len(amplitudes)}"
        channel_groups = anomalies_obj.get_data("channel_group")[0].values
        grouping = []
        for group in np.arange(1, 7):
            grouping.append(np.sum([g == group for g in channel_groups]))
        assert grouping == [
            2,
            2,
            1,
            2,
            3,
            3,
        ], f"Expected 1 group of each type. Found {grouping}"
        skew = anomalies_obj.get_data("skew")[0].values
        assert np.sum([skew > 0]) == 9
        assert np.sum([skew < 0]) == 4


def test_peak_finder_driver(tmp_path: Path):
    uijson_path = tmp_path.parent / "test_peak_finder_app0"
    json_file = next(uijson_path.glob("*.ui.json"))
    driver = PeakFinderDriver.start(str(uijson_path / json_file))

    with driver.params.geoh5.open(mode="r"):
        results = driver.params.geoh5.get_entity("PointMarkers")
        compare_entities(results[0], results[1], ignore=["_uid"])
