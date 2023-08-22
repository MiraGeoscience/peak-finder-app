#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app project.
#
#  All rights reserved.
#

from __future__ import annotations

from copy import deepcopy

from geoh5py.ui_json.constants import default_ui_json as base_ui_json

import peak_finder

defaults = {
    "version": peak_finder.__version__,
    "title": "Peak Finder Parameters",
    "geoh5": None,
    "objects": None,
    "data": None,
    "flip_sign": False,
    "line_field": None,
    "smoothing": 6,
    "min_amplitude": 1.0,
    "min_value": None,
    "min_width": 100.0,
    "max_migration": 25.0,
    "min_channels": 1,
    "ga_group_name": "peak_finder",
    "structural_markers": False,
    "line_id": None,
    "center": None,
    "width": None,
    "group_a_data": None,
    "group_a_color": None,
    "group_b_data": None,
    "group_b_color": None,
    "group_c_data": None,
    "group_c_color": None,
    "group_d_data": None,
    "group_d_color": None,
    "group_e_data": None,
    "group_e_color": None,
    "group_f_data": None,
    "group_f_color": None,
    "run_command": "peak_finder.application",
    "monitoring_directory": None,
    "workspace_geoh5": None,
    "conda_environment": "geoapps",
    "conda_environment_boolean": False,
}

default_ui_json = deepcopy(base_ui_json)
default_ui_json.update(
    {
        "version": peak_finder.__version__,
        "title": "Peak Finder Parameters",
        "launch_dash": {
            "main": True,
            "label": "Run interactive app",
            "value": True,
        },
        "objects": {
            "main": False,
            "group": "Data",
            "label": "Object",
            "meshType": [
                "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
            ],
            "value": None,
        },
        "data": {
            "association": "Vertex",
            "dataType": "Float",
            "group": "Data",
            "main": False,
            "dataGroupType": "Multi-element",
            "label": "Channels",
            "parent": "objects",
            "value": None,
        },
        "flip_sign": {
            "main": False,
            "group": "Data",
            "label": "Flip sign",
            "value": False,
        },
        "line_field": {
            "association": "Vertex",
            "dataType": "Referenced",
            "group": "Data",
            "main": False,
            "label": "Line Field",
            "parent": "objects",
            "value": None,
        },
        "smoothing": {
            "group": "Detection Parameters",
            "label": "Smoothing window",
            "main": False,
            "value": 6,
        },
        "min_amplitude": {
            "group": "Detection Parameters",
            "label": "Minimum Amplitude (%)",
            "value": 1.0,
            "main": False,
        },
        "min_value": {
            "group": "Detection Parameters",
            "label": "Minimum Value",
            "value": 0.0,
            "main": False,
        },
        "min_width": {
            "group": "Detection Parameters",
            "label": "Minimum Width (m)",
            "value": 100.0,
            "main": False,
        },
        "max_migration": {
            "group": "Detection Parameters",
            "label": "Maximum Peak Migration (m)",
            "value": 25.0,
            "main": False,
        },
        "min_channels": {
            "group": "Detection Parameters",
            "label": "Minimum # Channels",
            "value": 1,
            "main": False,
        },
        "ga_group_name": {
            "enabled": True,
            "main": False,
            "group": "Python run preferences",
            "label": "Save As",
            "value": "peak_finder",
        },
        "structural_markers": {
            "main": False,
            "group": "Python run preferences",
            "label": "Export all markers",
            "value": False,
        },
        "line_id": None,
        "group_a_data": {
            "main": False,
            "association": "Vertex",
            "group": "Group A",
            "dataGroupType": "Multi-element",
            "label": "Property Group",
            "parent": "objects",
            "value": None,
        },
        "group_a_color": {
            "main": False,
            "dataType": "Text",
            "group": "Group A",
            "label": "Color",
            "value": None,
        },
        "group_b_data": {
            "main": False,
            "association": "Vertex",
            "group": "Group B",
            "dataGroupType": "Multi-element",
            "label": "Property Group",
            "parent": "objects",
            "optional": True,
            "enabled": False,
            "value": None,
        },
        "group_b_color": {
            "main": False,
            "dataType": "Text",
            "group": "Group B",
            "label": "Color",
            "dependency": "group_b_data",
            "dependencyType": "enabled",
            "value": None,
        },
        "group_c_data": {
            "main": False,
            "association": "Vertex",
            "group": "Group C",
            "dataGroupType": "Multi-element",
            "label": "Property Group",
            "parent": "objects",
            "optional": True,
            "enabled": False,
            "value": None,
        },
        "group_c_color": {
            "main": False,
            "dataType": "Text",
            "group": "Group C",
            "label": "Color",
            "dependency": "group_c_data",
            "dependencyType": "enabled",
            "value": None,
        },
        "group_d_data": {
            "main": False,
            "association": "Vertex",
            "group": "Group D",
            "dataGroupType": "Multi-element",
            "label": "Property Group",
            "parent": "objects",
            "optional": True,
            "enabled": False,
            "value": None,
        },
        "group_d_color": {
            "main": False,
            "dataType": "Text",
            "group": "Group D",
            "label": "Color",
            "dependency": "group_d_data",
            "dependencyType": "enabled",
            "value": None,
        },
        "group_e_data": {
            "main": False,
            "association": "Vertex",
            "group": "Group E",
            "dataGroupType": "Multi-element",
            "label": "Property Group",
            "parent": "objects",
            "optional": True,
            "enabled": False,
            "value": None,
        },
        "group_e_color": {
            "main": False,
            "dataType": "Text",
            "group": "Group E",
            "label": "Color",
            "dependency": "group_e_data",
            "dependencyType": "enabled",
            "value": None,
        },
        "group_f_data": {
            "main": False,
            "association": "Vertex",
            "group": "Group F",
            "dataGroupType": "Multi-element",
            "label": "Property Group",
            "parent": "objects",
            "optional": True,
            "enabled": False,
            "value": None,
        },
        "group_f_color": {
            "main": False,
            "dataType": "Text",
            "group": "Group F",
            "label": "Color",
            "dependency": "group_f_data",
            "dependencyType": "enabled",
            "value": None,
        },
        "conda_environment": "geoapps",
        "run_command": "geoapps.peak_finder.driver",
    }
)


# Over-write validations for jupyter app parameters
validations = {
    "line_id": {"types": [int, type(None)]},
    "center": {"types": [float, type(None)]},
    "width": {"types": [float, type(None)]},
}

app_initializer: dict = {}
