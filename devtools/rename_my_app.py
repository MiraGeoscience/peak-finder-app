#!/usr/bin/env python3

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of peak-finder-app package.                               '
#                                                                              '
#  All rights reserved.                                                        '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

import argparse
import logging
import mimetypes
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

THIS_FILE = Path(__file__).resolve()
IGNORE_EXTENSIONS = [
    ".bmp",
    ".geoh5",
    ".gif",
    ".h5",
    ".ico",
    ".jpeg",
    ".jpg",
    ".png",
    ".pt",
    ".pynb",
]


def dash_to_underscore(name: str) -> str:
    return name.replace("-", "_")


def underscore_to_dash(name: str) -> str:
    return name.replace("_", "-")


class MyAppRenamer:
    """
    Replaces 'my-app' and 'my_app' with the new name in all files
    (excluding the '.git' and assets folder, as well as some known binary file extensions).

    Also calls git mv on 'my_app' and 'my_app-assets'.
    """

    def __init__(self, newname: str):
        """:param newname: The new name to replace 'my-app' and 'my_app' with."""

        self.root = Path(__file__).parents[1]
        self.newname = underscore_to_dash(newname)
        self.newname_underscore = dash_to_underscore(newname)

        logger.info(f"new name={self.newname}")
        logger.info(f"new name with underscores={self.newname_underscore}")

    def replace_in_file(self, filepath: Path) -> None:
        logger.debug(f"reading {filepath} ...")
        with open(filepath, encoding="utf-8") as file:
            content: str = file.read()
            patched_content = re.sub(r"\bmy-app\b", self.newname, content)
            patched_content = re.sub(
                r"\bmy_app\b", self.newname_underscore, patched_content
            )
        if patched_content != content:
            logger.info(f"patching {filepath}")
            with open(filepath, "w", encoding="utf-8") as file:
                file.write(patched_content)

    def get_non_image_files(self) -> list[Path]:
        """Return the list of files tracked by git that are not images, according to mimetypes."""

        tracked_files = subprocess.run(
            ["git", "ls-files"], cwd=self.root, capture_output=True, encoding="utf-8"
        ).stdout.splitlines()

        def is_image(p: Path) -> bool:
            mimetype, _ = mimetypes.guess_type(p, strict=False)
            return mimetype is not None and mimetype.startswith("image")

        paths = [Path(file) for file in tracked_files]
        return [p.resolve() for p in paths if p.is_file() and not is_image(p)]

    def run(self):
        """Applies the renaming across files and directories."""

        if (self.root / self.newname_underscore).exists():
            logger.error(
                f"Cannot rename to {self.newname_underscore} because this folder already exists."
            )
            exit(1)

        new_assets_folder_name = f"{self.newname_underscore}-assets"
        if (self.root / new_assets_folder_name).exists():
            logger.error(
                f"Cannot rename to {new_assets_folder_name} because this folder already exists."
            )
            exit(1)

        status_lines = subprocess.run(
            ["git", "status", "--short", "."],
            cwd=self.root,
            capture_output=True,
            encoding="utf-8",
        ).stdout.splitlines()
        # exclude untracked files
        status_lines = [line for line in status_lines if not line.startswith("?")]

        has_uncommitted_changes = len(status_lines) > 0
        if has_uncommitted_changes:
            logger.warning("There are uncommitted changes")
            print("Uncommitted changes:\n" + "\n".join(status_lines))
            print(
                "Preferably commit or stash them before running this script.\n"
                "Do you want to continue nonetheless? (y/n)"
            )
            answer = input()
            if answer.lower() != "y":
                logger.info("Aborting.")
                exit(1)

        for f in self.get_non_image_files():
            if not self.should_skip(f):
                self.replace_in_file(f)

        subprocess.run(["git", "add", "-u", "."], cwd=self.root)

        subprocess.run(["git", "mv", "my_app", self.newname_underscore], cwd=self.root)
        subprocess.run(
            ["git", "mv", "my_app-assets", new_assets_folder_name], cwd=self.root
        )

        logger.info("Changes applied. Please review them before committing.")

    def should_skip(self, file: Path) -> bool:
        """Check if the given path is a file that should be skipped."""

        return file == THIS_FILE or file.suffix.lower() in IGNORE_EXTENSIONS


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Rename my-app and my_app with a new name."
    )
    parser.add_argument(
        "newname", type=str, help="The new name to replace my-app and my_app with."
    )
    args = parser.parse_args()

    renamer = MyAppRenamer(args.newname)
    renamer.run()
