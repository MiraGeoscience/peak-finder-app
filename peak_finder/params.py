#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app project.
#
#  All rights reserved.
#

from __future__ import annotations

import string
from copy import deepcopy

from geoapps_utils.driver.params import BaseParams
from geoh5py.data import Data
from geoh5py.groups import PropertyGroup
from geoh5py.objects import ObjectBase
from geoh5py.ui_json import InputFile

from peak_finder.constants import default_ui_json, defaults, validations


class PeakFinderParams(BaseParams):  # pylint: disable=R0902, R0904
    """
    Parameter class for peak finder application.
    """

    def __init__(self, input_file: InputFile | None = None, **kwargs):
        self._default_ui_json: dict | None = deepcopy(default_ui_json)
        self._defaults: dict | None = deepcopy(defaults)
        self._free_parameter_keys: list = ["data", "color"]
        self._free_parameter_identifier: str = "group"
        self._validations: dict | None = validations
        self._objects: ObjectBase | None = None
        self._data: Data | None = None
        self._flip_sign: bool | None = None
        self._line_field: Data | None = None
        self._smoothing: int | None = None
        self._min_amplitude: int | None = None
        self._min_value: float | None = None
        self._min_width: float | None = None
        self._max_migration: float | None = None
        self._min_channels: int | None = None
        self._ga_group_name: str | None = None
        self._structural_markers: bool | None = None
        self._line_id: int | None = None
        self._center: float | None = None
        self._width: float | None = None
        self._group_a_data: PropertyGroup | None = None
        self._group_a_color: str | None = None
        self._group_b_data: PropertyGroup | None = None
        self._group_b_color: str | None = None
        self._group_c_data: PropertyGroup | None = None
        self._group_c_color: str | None = None
        self._group_d_data: PropertyGroup | None = None
        self._group_d_color: str | None = None
        self._group_e_data: PropertyGroup | None = None
        self._group_e_color: str | None = None
        self._group_f_data: PropertyGroup | None = None
        self._group_f_color: str | None = None
        self._property_groups: dict | None = None
        self._template_data: Data | None = None
        self._template_color: str | None = None
        self._plot_result: bool = True
        self._title: str | None = None

        if input_file is None:
            ui_json = deepcopy(self._default_ui_json)
            input_file = InputFile(
                ui_json=ui_json,
                validations=self.validations,
                validate=False,
            )
        super().__init__(input_file=input_file, **kwargs)

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, val):
        self.setter_validator("center", val)

    @property
    def conda_environment(self):
        return self._conda_environment

    @conda_environment.setter
    def conda_environment(self, val):
        self.setter_validator("conda_environment", val)

    @property
    def conda_environment_boolean(self):
        return self._conda_environment_boolean

    @conda_environment_boolean.setter
    def conda_environment_boolean(self, val):
        self.setter_validator("conda_environment_boolean", val)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        self.setter_validator("data", val, fun=self._uuid_promoter)

    @property
    def flip_sign(self):
        return self._flip_sign

    @flip_sign.setter
    def flip_sign(self, val):
        self.setter_validator("flip_sign", val)

    @property
    def ga_group_name(self):
        return self._ga_group_name

    @ga_group_name.setter
    def ga_group_name(self, val):
        self.setter_validator("ga_group_name", val)

    @property
    def line_field(self):
        return self._line_field

    @line_field.setter
    def line_field(self, val):
        self.setter_validator("line_field", val, fun=self._uuid_promoter)

    @property
    def line_id(self):
        return self._line_id

    @line_id.setter
    def line_id(self, val):
        self.setter_validator("line_id", val)

    @property
    def max_migration(self):
        return self._max_migration

    @max_migration.setter
    def max_migration(self, val):
        self.setter_validator("max_migration", val)

    @property
    def min_amplitude(self):
        return self._min_amplitude

    @min_amplitude.setter
    def min_amplitude(self, val):
        self.setter_validator("min_amplitude", val)

    @property
    def min_channels(self):
        return self._min_channels

    @min_channels.setter
    def min_channels(self, val):
        self.setter_validator("min_channels", val)

    @property
    def min_value(self):
        return self._min_value

    @min_value.setter
    def min_value(self, val):
        self.setter_validator("min_value", val)

    @property
    def min_width(self):
        return self._min_width

    @min_width.setter
    def min_width(self, val):
        self.setter_validator("min_width", val)

    @property
    def monitoring_directory(self):
        return self._monitoring_directory

    @monitoring_directory.setter
    def monitoring_directory(self, val):
        self.setter_validator("monitoring_directory", val)

    @property
    def objects(self):
        return self._objects

    @objects.setter
    def objects(self, val):
        self.setter_validator("objects", val, fun=self._uuid_promoter)

    @property
    def plot_result(self):
        return self._plot_result

    @plot_result.setter
    def plot_result(self, val):
        self._plot_result = val

    @property
    def smoothing(self):
        return self._smoothing

    @smoothing.setter
    def smoothing(self, val):
        self.setter_validator("smoothing", val)

    @property
    def structural_markers(self):
        return self._structural_markers

    @structural_markers.setter
    def structural_markers(self, val):
        self.setter_validator("structural_markers", val)

    @property
    def template_data(self):
        return self._template_data

    @template_data.setter
    def template_data(self, val):
        self.setter_validator("template_data", val)

    @property
    def template_color(self):
        return self._template_color

    @template_color.setter
    def template_color(self, val):
        self.setter_validator("template_color", val)

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, val):
        self.setter_validator("title", val)

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, val):
        self.setter_validator("width", val)

    @property
    def group_a_data(self):
        return self._group_a_data

    @group_a_data.setter
    def group_a_data(self, val):
        self.setter_validator("group_a_data", val)

    @property
    def group_a_color(self):
        return self._group_a_color

    @group_a_color.setter
    def group_a_color(self, val):
        self.setter_validator("group_a_color", val)

    @property
    def group_b_data(self):
        return self._group_b_data

    @group_b_data.setter
    def group_b_data(self, val):
        self.setter_validator("group_b_data", val)

    @property
    def group_b_color(self):
        return self._group_b_color

    @group_b_color.setter
    def group_b_color(self, val):
        self.setter_validator("group_b_color", val)

    @property
    def group_c_data(self):
        return self._group_c_data

    @group_c_data.setter
    def group_c_data(self, val):
        self.setter_validator("group_c_data", val)

    @property
    def group_c_color(self):
        return self._group_c_color

    @group_c_color.setter
    def group_c_color(self, val):
        self.setter_validator("group_c_color", val)

    @property
    def group_d_data(self):
        return self._group_d_data

    @group_d_data.setter
    def group_d_data(self, val):
        self.setter_validator("group_d_data", val)

    @property
    def group_d_color(self):
        return self._group_d_color

    @group_d_color.setter
    def group_d_color(self, val):
        self.setter_validator("group_d_color", val)

    @property
    def group_e_data(self):
        return self._group_e_data

    @group_e_data.setter
    def group_e_data(self, val):
        self.setter_validator("group_e_data", val)

    @property
    def group_e_color(self):
        return self._group_e_color

    @group_e_color.setter
    def group_e_color(self, val):
        self.setter_validator("group_e_color", val)

    @property
    def group_f_data(self):
        return self._group_f_data

    @group_f_data.setter
    def group_f_data(self, val):
        self.setter_validator("group_f_data", val)

    @property
    def group_f_color(self):
        return self._group_f_color

    @group_f_color.setter
    def group_f_color(self, val):
        self.setter_validator("group_f_color", val)

    def get_property_groups(self):
        """
        Generate a dictionary of groups with associate properties from params.
        """
        count = 0
        property_groups = {}
        for name in string.ascii_lowercase[:6]:
            prop_group = getattr(self, f"group_{name}_data", None)
            if prop_group is not None:
                count += 1
                property_groups[prop_group.name] = {
                    "param": name,
                    "data": prop_group.uid,
                    "color": getattr(self, f"group_{name}_color", None),
                    "label": [count],
                    "properties": prop_group.properties,
                }
        return property_groups
