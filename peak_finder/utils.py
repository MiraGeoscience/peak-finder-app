#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of peak-finder-app project.
#
#  All rights reserved.
#

import numpy as np
from geoapps_utils.utils.numerical import traveling_salesman
from geoh5py.data import Data
from geoh5py.objects import Curve


def get_ordered_survey_lines(survey: Curve, line_field: Data) -> dict:
    """
    Order of survey lines.

    :param survey: Survey object.
    :type line_field: Survey line labels.
    """
    if survey.vertices is None:
        return {}

    line_ids = []
    line_labels = []
    locs = []
    value_map = line_field.value_map.map  # type: ignore

    for line_id in np.unique(line_field.values):
        line_indices = np.where(line_field.values == line_id)[0]
        mean_locs = np.mean(survey.vertices[line_indices], axis=0)
        line_ids.append(line_id)
        line_labels.append(value_map[line_id])
        locs.append(mean_locs)

    order = traveling_salesman(np.array(locs))
    ordered_survey_lines = {line_ids[ind]: line_labels[ind] for ind in order}

    return ordered_survey_lines
