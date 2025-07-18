# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2025 Mira Geoscience Ltd.                                     '
#                                                                                   '
#  This file is part of peak-finder-app package.                                    '
#                                                                                   '
#  peak-finder-app is distributed under the terms and conditions of the MIT License '
#  (see LICENSE file at the root of this source code package).                      '
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

import string
from pathlib import Path

import numpy as np
from dash._callback_context import context_value
from dash._utils import AttributeDict
from geoh5py.data import ReferencedData
from geoh5py.objects import Curve, Points
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace
from scipy import stats

from peak_finder.application import PeakFinder, PeakFinderDriver
from peak_finder.params import PeakFinderParams


# pylint: disable=R0801


def get_template_anomalies():
    """
    Add gaussian anomalies in series.
    """
    dist1 = 10 * stats.norm.pdf(np.arange(0, 200, 0.1), 100, 1.5)
    dist2 = 3 * stats.norm.pdf(np.arange(200, 450, 0.1), 300, 25)
    dist3 = 5 * stats.norm.pdf(np.arange(450, 650, 0.1), 600, 1)
    dist4 = 7 * stats.norm.pdf(np.arange(650, 725, 0.1), 700, 1.75)
    dist5 = 1000 * stats.norm.pdf(np.arange(725, 875, 0.1), 800, 3)
    dist6 = 7 * stats.norm.pdf(np.arange(875, 1000, 0.1), 900, 3)

    return np.concatenate((dist1, dist2, dist3, dist4, dist5, dist6))


def test_peak_finder_app(tmp_path: Path):  # pylint: disable=too-many-locals
    h5file_path = tmp_path / r"testPeakFinder.geoh5"
    # Create temp workspace
    temp_ws = Workspace(h5file_path)

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
                "value_map": {0: "Unknown", 1: "1", 2: "2", 3: "3"},
                "type": "referenced",
            }
        }
    )

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
    temp_ws.close()

    param_names = string.ascii_lowercase[:6]
    property_groups = {}
    colors = [
        "#0000FF",
        "#FFFF00",
        "#FF0000",
        "#00FFFF",
        "#008000",
        "#FFA500",
    ]
    for ind, group in enumerate(
        [early, middle, late, early_middle, early_middle_late, middle_late]
    ):
        property_groups[group.name] = {
            "param": param_names[ind],
            "data": str(group.uid),
            "color": colors[ind],
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

    params = PeakFinderParams(geoh5=str(h5file_path))
    app = PeakFinder(
        params,
        ui_json_data={
            "objects": objects,
            "line_field": line_field,
        },
    )
    app.property_groups = property_groups
    app.workspace = temp_ws

    app.trigger_click(
        n_clicks=0,
        flip_sign=[],
        trend_lines=[],
        masking_data=None,
        smoothing=smoothing,
        min_amplitude=min_amplitude,
        min_value=min_value,
        min_width=min_width,
        max_migration=max_migration,
        min_channels=min_channels,
        n_groups=1,
        max_separation=100.0,
        selected_line=1,
        ga_group_name="peak_finder",
    )

    with Workspace(h5file_path) as out_ws:
        anomalies_obj = out_ws.get_entity("Anomaly Groups")[0]
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


def test_peak_finder_driver(tmp_path: Path):
    uijson_path = tmp_path.parent / "test_peak_finder_app0"
    json_file = next(uijson_path.glob("*.ui.json"))
    driver = PeakFinderDriver.start(str(uijson_path / json_file))

    with driver.params.geoh5.open(mode="r"):
        results = driver.params.geoh5.get_entity("Anomaly Groups")
        compare_entities(results[0], results[1], ignore=["_uid"])


def test_merging_peaks(tmp_path: Path):  # pylint: disable=too-many-locals
    h5file_path = tmp_path / r"testPeakFinder.geoh5"
    # Create temp workspace
    temp_ws = Workspace(h5file_path)

    x = np.arange(0, 1000, 0.1)

    curve = Curve.create(temp_ws, vertices=np.c_[x, np.zeros((x.shape[0], 2))])
    dist = get_template_anomalies()
    data = curve.add_data({"data": {"values": dist}})
    curve.add_data_to_group(data, property_group="obs")

    line = curve.add_data(
        {
            "line_id": {
                "values": np.ones_like(x),
                "value_map": {0: "Unknown", 1: "1", 2: "2", 3: "3"},
                "type": "referenced",
            }
        }
    )

    prop_group = curve.find_or_create_property_group(
        name="prop group", properties=[data.uid]
    )

    property_groups = {
        "obs": {
            "param": "a",
            "data": str(prop_group.uid),
            "color": "#000000",
            "label": [0],
            "properties": [str(p) for p in prop_group.properties],
        }
    }

    objects = "{" + str(curve.uid) + "}"
    smoothing = 6
    max_migration = 1.0
    min_channels = 1
    min_amplitude = 0
    min_value = -1.4
    min_width = 1.0
    line_field = "{" + str(line.uid) + "}"

    params = PeakFinderParams(geoh5=str(h5file_path))
    app = PeakFinder(
        params,
        ui_json_data={
            "objects": objects,
            "line_field": line_field,
        },
    )
    app.property_groups = property_groups
    app.workspace = temp_ws
    temp_ws.close()
    # Test merging peaks
    n_groups_list = [2, 2, 2, 3, 2]
    max_separation_list = [1, 55, 65, 65, 90]

    expected_peaks = [
        [],
        [850],
        [750, 850],
        [800],
        [200, 650, 750, 850],
    ]
    for ind in range(5):
        app.trigger_click(
            n_clicks=0,
            flip_sign=[],
            trend_lines=[],
            masking_data=None,
            smoothing=smoothing,
            min_amplitude=min_amplitude,
            min_value=min_value,
            min_width=min_width,
            max_migration=max_migration,
            min_channels=min_channels,
            n_groups=n_groups_list[ind],
            max_separation=max_separation_list[ind],
            selected_line=1,
            ga_group_name="peak_finder_" + str(ind),
        )

        with Workspace(h5file_path) as out_ws:
            group = out_ws.get_entity("peak_finder_" + str(ind))[0]
            anomalies_obj = [
                obj for obj in group.children if obj.name == "Anomaly Groups"
            ]
            if len(expected_peaks[ind]) == 0:  # type: ignore
                assert len(anomalies_obj) == 0
                continue
            amplitudes = anomalies_obj[0].get_data("amplitude")[0].values
            assert len(amplitudes) == len(expected_peaks[ind])  # type: ignore
            assert np.all(
                np.isclose(
                    np.sort(anomalies_obj[0].vertices[:, 0]),
                    expected_peaks[ind],
                    rtol=0.05,
                )
            )


def test_masking_peaks(tmp_path: Path):  # pylint: disable=too-many-locals
    h5file_path = tmp_path / r"testPeakFinder.geoh5"
    # Create temp workspace
    temp_ws = Workspace(h5file_path)

    x = np.arange(0, 1000, 0.1)

    curve = Curve.create(temp_ws, vertices=np.c_[x, np.zeros((x.shape[0], 2))])

    peak1 = 5 * stats.norm.pdf(np.arange(0, 650, 0.1), 600, 1)
    peak2 = 7 * stats.norm.pdf(np.arange(650, 725, 0.1), 700, 1.75)
    peak3 = 1000 * stats.norm.pdf(np.arange(725, 875, 0.1), 800, 3)
    peak4 = 7 * stats.norm.pdf(np.arange(875, 1000, 0.1), 900, 3)

    dist = np.concatenate((peak1, peak2, peak3, peak4))

    data = curve.add_data({"data": {"values": dist}})
    curve.add_data_to_group(data, property_group="obs")

    line = curve.add_data(
        {
            "line_id": {
                "values": np.ones_like(x),
                "value_map": {0: "Unknown", 1: "1", 2: "2", 3: "3"},
                "type": "referenced",
            }
        }
    )

    masking_array = np.ones_like(x)
    masking_array[(x > 650) & (x < 850)] = 0
    masking_data = curve.add_data(
        {
            "masking": {
                "values": np.array(masking_array, dtype=bool),
                "type": "boolean",
            }
        }
    )

    prop_group = curve.find_or_create_property_group(
        name="prop group", properties=[data.uid]
    )
    temp_ws.close()

    property_groups = {
        "obs": {
            "param": "a",
            "data": str(prop_group.uid),
            "color": "#000000",
            "label": [0],
            "properties": [str(p) for p in prop_group.properties],
        }
    }

    objects = "{" + str(curve.uid) + "}"
    smoothing = 6
    max_migration = 1.0
    min_channels = 1
    min_amplitude = 0
    min_value = -1.4
    min_width = 1.0
    line_field = "{" + str(line.uid) + "}"

    params = PeakFinderParams(geoh5=str(h5file_path))
    app = PeakFinder(
        params,
        ui_json_data={
            "objects": objects,
            "line_field": line_field,
        },
    )
    app.property_groups = property_groups

    # Test masking
    app.trigger_click(
        n_clicks=0,
        flip_sign=[],
        trend_lines=[],
        masking_data=str(masking_data.uid),
        smoothing=smoothing,
        min_amplitude=min_amplitude,
        min_value=min_value,
        min_width=min_width,
        max_migration=max_migration,
        min_channels=min_channels,
        n_groups=1,
        max_separation=350,
        selected_line=1,
        ga_group_name="peak_finder_masking",
    )

    with Workspace(h5file_path) as out_ws:
        anomalies_obj = out_ws.get_entity("Anomaly Groups")[0]
        vertices = anomalies_obj.vertices[:, 0]
        assert np.all((vertices < 650) | (vertices > 850))
        assert len(vertices) == 2


def test_map_locations(tmp_path: Path):  # pylint: disable=too-many-locals
    h5file_path = tmp_path / r"testPeakFinder.geoh5"
    # Create temp workspace
    temp_ws = Workspace(h5file_path)

    x = np.linspace(0, 1000, 10000)
    noise = np.random.uniform(low=-5, high=5, size=len(x))
    x = np.sort(x + noise)

    curve = Curve.create(temp_ws, vertices=np.c_[x, np.zeros((x.shape[0], 2))])

    dist = get_template_anomalies()

    expected_peaks = [100, 300, 600, 700, 800, 900]

    data = curve.add_data({"data": {"values": dist}})
    curve.add_data_to_group(data, property_group="obs")

    line = curve.add_data(
        {
            "line_id": {
                "values": np.ones_like(x),
                "value_map": {0: "Unknown", 1: "1", 2: "2", 3: "3"},
                "type": "referenced",
            }
        }
    )

    prop_group = curve.find_or_create_property_group(
        name="prop group", properties=[data.uid]
    )

    property_groups = {
        "obs": {
            "param": "a",
            "data": str(prop_group.uid),
            "color": "#000000",
            "label": [0],
            "properties": [str(p) for p in prop_group.properties],
        }
    }

    survey = "{" + str(curve.uid) + "}"
    smoothing = 6
    max_migration = 1.0
    min_channels = 1
    min_amplitude = 0
    min_value = -1.4
    min_width = 1.0
    line_field = "{" + str(line.uid) + "}"

    params = PeakFinderParams(geoh5=str(h5file_path))
    app = PeakFinder(
        params,
        ui_json_data={
            "objects": survey,
            "line_field": line_field,
        },
    )
    app.property_groups = property_groups
    app.workspace = temp_ws

    # Test merging peaks
    n_groups = 1
    max_separation = 100

    context_value.set(
        AttributeDict(
            **{
                "triggered_inputs": [
                    {"prop_id": "line_indices_trigger.data", "data": 0}
                ]
            }
        )
    )

    app.update_line_indices(0, 0, 1, 1)

    context_value.set(
        AttributeDict(
            **{
                "triggered_inputs": [
                    {"prop_id": "line_indices_trigger.data", "data": 1}  # type: ignore
                ]
            }
        )
    )

    app.compute_lines(
        lines_computation_trigger=0,
        line_indices_trigger=0,
        survey_trigger=0,
        selected_line=1,
        n_lines=1,
        smoothing=smoothing,
        max_migration=max_migration,
        min_channels=min_channels,
        min_amplitude=min_amplitude,
        min_value=min_value,
        min_width=min_width,
        n_groups=n_groups,
        max_separation=max_separation,
    )

    if app.computed_lines is not None:
        for value in app.computed_lines.values():
            positions = value["position"]
            anomaly_groups = value["anomalies"]

            for pos_ind, anomaly_group in enumerate(anomaly_groups):
                map_locs = positions[pos_ind].map_locations

                og_inds = [anom.peaks for anom in anomaly_group]
                inds = map_locs[og_inds]

                locs = positions[pos_ind].locations[inds]
                og_locs = positions[pos_ind].locations_resampled[og_inds]

                assert len(np.setdiff1d(locs, og_locs)) == len(expected_peaks)
                assert np.isclose(locs, expected_peaks, atol=5).any()
                assert np.isclose(og_locs, expected_peaks, atol=5).any()


def test_trend_line(tmp_path: Path):  # pylint: disable=too-many-locals
    h5file_path = tmp_path / r"testPeakFinder.geoh5"
    # Create temp workspace
    temp_ws = Workspace(h5file_path)

    x_locs, y_locs = [], []
    line_id, data = [], []
    for ind in range(5):
        x = np.linspace(0, 1000, 10000)
        noise = np.random.uniform(low=-5, high=5, size=len(x))
        x = np.sort(x + noise)

        y = np.linspace(0, 1000, 10000) + ind * 300
        noise = np.random.uniform(low=-10, high=10, size=len(x))
        y = np.sort(y + noise)

        x_locs.append(x)
        y_locs.append(y)
        line_id.append(np.ones_like(x) * (ind + 1))
        data.append(get_template_anomalies())

    x_locs = np.concatenate(x_locs)
    y_locs = np.concatenate(y_locs)
    curve = Curve.create(temp_ws, vertices=np.c_[x_locs, y_locs, np.zeros_like(x_locs)])

    data = curve.add_data({"data": {"values": np.concatenate(data)}})
    prop_group = curve.add_data_to_group(data, property_group="obs")
    value_map = {0: "Unknown"}
    value_map.update({ind + 1: f"{ind + 1}" for ind in range(len(line_id))})

    line = curve.add_data(
        {
            "line_id": {
                "values": np.concatenate(line_id),
                "value_map": value_map,
                "type": "referenced",
            }
        }
    )

    temp_ws.close()

    params = PeakFinderParams(
        geoh5=temp_ws,
        objects=curve,
        line_field=line,
        group_a_data=prop_group,
        trend_lines=True,
    )

    params.input_file.write_ui_json("test_peak_trend", tmp_path)
    PeakFinderDriver(params).run()

    with temp_ws.open():
        trend_lines = temp_ws.get_entity("Trend Lines")[0]
        anomalies = temp_ws.get_entity("Anomaly Groups")[0]

        assert isinstance(anomalies, Points)
        assert isinstance(trend_lines, Curve)

        anom_group = anomalies.get_data("channel_group")[0]
        trend_group = trend_lines.get_data("channel_group")[0]

        assert isinstance(trend_group, ReferencedData)
        assert trend_group.entity_type.uid == anom_group.entity_type.uid
