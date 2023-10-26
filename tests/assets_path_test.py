#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app.
#
#  All rights reserved.

from __future__ import annotations

from pathlib import Path

from peak_finder import assets_path


def test_assets_directory_exist():
    assert assets_path().is_dir()


def test_assets_directory_from_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PEAK_FINDER_ASSETS_DIR", str(tmp_path.absolute()))

    assert assets_path().is_dir()
    assert tmp_path == assets_path()


def test_assets_directory_from_wrong_env(tmp_path: Path, monkeypatch):
    non_existing_path = tmp_path / "wrong"
    monkeypatch.setenv("PEAK_FINDER_ASSETS_DIR", str(non_existing_path.absolute()))

    assert non_existing_path.is_dir() is False
    assert assets_path().is_dir() is True
    assert non_existing_path != assets_path()


def test_uijson_files_exists():
    assert (assets_path() / "uijson").is_dir()
    assert list((assets_path() / "uijson").iterdir())[0].is_file()
