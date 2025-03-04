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

    return pyproject["tool"]["poetry"]["version"]


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


def test_version_is_semver():
    semver_re = (
        r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
        r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
        r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
        r"(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
    )
    assert re.search(semver_re, peak_finder.__version__) is not None
