#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app project.
#
#  All rights reserved.
#

from __future__ import annotations

from pathlib import Path

from geoh5py.ui_json import InputFile

from peak_finder.base.write_default_uijson import write_default_uijson
from peak_finder.constants import app_initializer
from peak_finder.params import PeakFinderParams


def test_write_default_uijson(tmp_path: Path):
    write_default_uijson(
        path=tmp_path,
        params_class=PeakFinderParams,
        app_initializer=app_initializer,
        filename="peak_finder",
        use_initializers=True
    )

    filepath = tmp_path / "peak_finder.ui.json"
    assert filepath.is_file()
    ifile = InputFile.read_ui_json(filepath, validate=False)
    params = PeakFinderParams(input_file=ifile, validate=False)
    assert params.line_id == 13
