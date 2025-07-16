# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2025 Mira Geoscience Ltd.                                     '
#                                                                                   '
#  This file is part of peak-finder-app package.                                    '
#                                                                                   '
#  peak-finder-app is distributed under the terms and conditions of the MIT License '
#  (see LICENSE file at the root of this source code package).                      '
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#
#  This file is part of peak-finder-app.
#
#  All rights reserved.

from __future__ import annotations

import re
from pathlib import Path

import tomli as toml
import yaml
from jinja2 import Template
from packaging.version import Version

import peak_finder


def get_pyproject_version():
    path = Path(__file__).resolve().parents[1] / "pyproject.toml"

    with open(str(path), encoding="utf-8") as file:
        pyproject = toml.loads(file.read())

    return pyproject["project"]["version"]


def get_conda_recipe_version():
    path = Path(__file__).resolve().parents[1] / "recipe.yaml"

    with open(str(path), encoding="utf-8") as file:
        content = file.read()

    template = Template(content)
    rendered_yaml = template.render()

    recipe = yaml.safe_load(rendered_yaml)

    return recipe["context"]["version"]


def test_version_is_consistent():
    assert peak_finder.__version__ == get_pyproject_version()
    normalized_conda_version = Version(get_conda_recipe_version())
    normalized_version = Version(peak_finder.__version__)
    assert normalized_conda_version == normalized_version


def test_conda_version_is_pep440():
    version = Version(get_conda_recipe_version())
    assert version is not None


def validate_version(version_str):
    try:
        version = Version(version_str)
        return (version.major, version.minor, version.micro, version.pre, version.post)
    except InvalidVersion:
        return None


def test_version_is_valid():
    assert validate_version(peak_finder.__version__) is not None
