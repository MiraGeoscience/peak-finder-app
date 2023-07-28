#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app project.
#
#  All rights reserved.
#
# pylint: disable=C0302

from __future__ import annotations

import numpy as np
from geoh5py.groups import PropertyGroup


def default_groups_from_property_group(  # pylint: disable=R0914
    property_group: PropertyGroup,
    start_index: int = 0,
) -> dict:
    """
    Create default channel groups from a property group.

    :param property_group: Property group to create channel groups from.
    :param start_index: Starting index of the groups.

    :return: Dictionary of channel groups.
    """
    _default_channel_groups = {
        "early": {"label": ["early"], "color": "#0000FF", "channels": []},
        "middle": {"label": ["middle"], "color": "#FFFF00", "channels": []},
        "late": {"label": ["late"], "color": "#FF0000", "channels": []},
        "early + middle": {
            "label": ["early", "middle"],
            "color": "#00FFFF",
            "channels": [],
        },
        "early + middle + late": {
            "label": ["early", "middle", "late"],
            "color": "#008000",
            "channels": [],
        },
        "middle + late": {
            "label": ["middle", "late"],
            "color": "#FFA500",
            "channels": [],
        },
    }

    parent = property_group.parent

    data_list = [
        parent.workspace.get_entity(uid)[0] for uid in property_group.properties
    ]

    start = start_index
    end = len(data_list)
    block = int((end - start) / 3)
    ranges = {
        "early": np.arange(start, start + block).tolist(),
        "middle": np.arange(start + block, start + 2 * block).tolist(),
        "late": np.arange(start + 2 * block, end).tolist(),
    }

    channel_groups = {}
    for i, (key, default) in enumerate(_default_channel_groups.items()):
        prop_group = parent.find_or_create_property_group(name=key)
        prop_group.properties = []

        for val in default["label"]:
            for ind in ranges[val]:
                prop_group.properties += [data_list[ind].uid]

        channel_groups[prop_group.name] = {
            "data": prop_group.uid,
            "color": default["color"],
            "label": [i + 1],
            "properties": prop_group.properties,
        }

    return channel_groups
