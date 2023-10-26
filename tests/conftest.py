#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app.
#
#  All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest
import toml


@pytest.fixture
def pyproject() -> dict[str]:
    """Return the pyproject.toml as a dictionary."""

    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    return toml.load(pyproject_path)
