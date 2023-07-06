#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import argparse
from pathlib import Path

from peak_finder.base import assets_path
from peak_finder.base.params import BaseParams


def write_default_uijson(
        path: str | Path,
        params_class: BaseParams,
        app_initializer: dict,
        filename: str,
        use_initializers=False
):
    app_initializer["geoh5"] = str(assets_path() / "FlinFlon.geoh5")
    app_initializer = app_initializer if use_initializers else {}

    params = params_class(validate=False, **app_initializer)

    validation_options = {
        "update_enabled": (True if params.geoh5 is not None else False)
    }
    params.input_file.validation_options = validation_options
    params.write_input_file(name=filename + ".ui.json", path=path, validate=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write defaulted ui.json files.")
    parser.add_argument(
        "path",
        type=Path,
        help="Path to folder where default ui.json files will be written.",
    )
    parser.add_argument(
        "--use_initializers",
        help="Write files initialized with FlinFlon values.",
        action="store_true",
    )
    args = parser.parse_args()
    write_default_uijson(args.path, args.use_initializers)
