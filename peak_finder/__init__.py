#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app project.
#
#  All rights reserved.
#

__version__ = "0.2.0-alpha.1"

import os
import warnings
from pathlib import Path


def assets_path() -> Path:
    """Return the path to the assets folder."""

    assets_dir_env_var = "PEAK_FINDER_ASSETS_DIR"
    assets_dirname = os.environ.get(assets_dir_env_var, None)
    if assets_dirname:
        assets_folder = Path(assets_dirname)
        if not assets_folder.is_dir():
            warnings.warn(
                f"Custom assets folder not found: {assets_dir_env_var}={assets_dirname}"
            )
        else:
            return assets_folder

    parent = Path(__file__).parent
    folder_name = f"{parent.name}-assets"
    assets_folder = parent.parent / folder_name
    if not assets_folder.is_dir():
        raise RuntimeError(f"Assets folder not found: {assets_folder}")

    return assets_folder
