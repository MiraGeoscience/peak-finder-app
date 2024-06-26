#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app.
#
#  All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest
import tomli as toml


@pytest.fixture
def pyproject() -> dict:
    """Return the pyproject.toml as a dictionary."""

    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with open(pyproject_path, "rb") as pyproject_file:
        return toml.load(pyproject_file)
