#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app project.
#
#  All rights reserved.
#

from __future__ import annotations

import json

import peak_finder

defaults = {
    "version": peak_finder.__version__,
    "title": "Peak Finder Parameters",
    "geoh5": None,
    "objects": None,
    "flip_sign": False,
    "line_field": None,
    "trend_lines": None,
    "masking_data": None,
    "smoothing": 6,
    "min_amplitude": 1.0,
    "min_value": None,
    "min_width": 100.0,
    "max_migration": 25.0,
    "min_channels": 1,
    "n_groups": 1,
    "max_separation": 100.0,
    "ga_group_name": "peak_finder",
    "structural_markers": False,
    "line_id": None,
    "center": None,
    "width": None,
    "group_a_data": None,
    "group_a_color": "#0000FF",
    "group_b_data": None,
    "group_b_color": "#FFFF00",
    "group_c_data": None,
    "group_c_color": "#FF0000",
    "group_d_data": None,
    "group_d_color": "#00FFFF",
    "group_e_data": None,
    "group_e_color": "#008000",
    "group_f_data": None,
    "group_f_color": "#FFA500",
    "run_command": "peak_finder.application",
    "monitoring_directory": None,
    "workspace_geoh5": None,
    "conda_environment": "peak-finder-app",
    "conda_environment_boolean": False,
}

file = peak_finder.assets_path() / "uijson/peak_finder.ui.json"

with open(file, encoding="utf-8") as f:
    default_ui_json = json.load(f)

default_ui_json["version"] = peak_finder.__version__

# Over-write validations for jupyter app parameters
validations = {
    "line_id": {"types": [int, type(None)]},
    "center": {"types": [float, type(None)]},
    "width": {"types": [float, type(None)]},
}

app_initializer: dict = {}
