.. _parameters:

Parameters
==========

In this section we will look at all the controls in the Peak Finder user
interface.  The parameters exposed in the interface are divided into three
sections: Visual Parameters, Data Selection, and Detection Parameters.
Each parameter will expose the docstring of the underlying parameter where
applicable and include a description of the parameters effect on the
visualization and peak finding process.

Visual Parameters
~~~~~~~~~~~~~~~~~

.. figure:: /images/parameters/visual_parameters.png
   :align: left

**X-axis Label**

Updates the label on the data section view x-axis.

.. figure:: /images/parameters/visualization/section_xlabel.png

   The x-axis label is updated to reflect the selection.

**Y-axis Scaling**

Updates the scaling of the y-axis of the data section view

.. todo::

   Add a figure showing data with both linear and symlog scaling

**Linear threshold**

When Symlog is chosen for Y-axis scaling, this parameter will set the
region in which linear scaling is used.

.. figure:: /images/parameters/visualization/linthresh_compare.png
   :scale: 60%

   Comparing the data visualization with a symlog linear threshold set to
   10E-3.2 (left) and 10E-5.1 (right).

**Plot residuals**

Switches on and off the residual visualization that shows the difference
between the raw and smoothed data.  Go :ref:`here <smoothing>` to learn
about smoothing.

.. figure:: /images/parameters/visualization/residuals.png
   :scale: 40%

   The residual layer is used to show the effect of the smoothing factor.

**Plot markers**

Switches on and off the markers outlining the character of each anomaly

.. figure:: /images/parameters/visualization/markers.png
   :scale: 40%

   Markers are used to indicate the left and right edges, the center,
   and the inflection point in curvature of each anomaly.

Data Selection
~~~~~~~~~~~~~~
.. figure:: /images/parameters/data_selection_parameters.png
   :width: 80%
   :align: left

**Lines Field**

.. autoproperty:: peak_finder.params.PeakFinderParams.line_field

**Select Line**

.. autoproperty:: peak_finder.params.PeakFinderParams.line_id

.. todo::

   Add a figure showing the plan view line selection (black).

**Masking Data**

.. autoproperty:: peak_finder.params.PeakFinderParams.masking_data

.. todo::

   Update docstring and show a figure of a working masked result.

**N outward lines**

Includes N lines in plan view on either side of the selected line.

.. figure:: /images/parameters/data_selection/outward_line_compare.png
   :scale: 40%

   Comparing the plan view with 1 outward line (left) and 2 outward lines
   (right).

**Flip Y (-1x)**

.. autoproperty:: peak_finder.params.PeakFinderParams.flip_sign

.. todo::

   Update docstring and add figure showing the effect of flipping y.

**Select group colors**

.. todo::

   Add figure of color picker widget.  Move this ui to visualization group?

Detection Parameters
~~~~~~~~~~~~~~~~~~~~

.. figure:: /images/parameters/detection_parameters.png
   :align: left

**Smoothing**

.. _smoothing:

.. autoproperty:: peak_finder.params.PeakFinderParams.smoothing

.. todo::

   Update docstring and add reference figure shown for plot residuals.

**Minimum Amplitude (%)**

.. autoproperty:: peak_finder.params.PeakFinderParams.min_amplitude

.. todo::

    Update docstring and add figure showing the effect of anomaly identification

**Minimum Data Value**

.. autoproperty:: peak_finder.params.PeakFinderParams.min_value

.. todo::

    Update docstring and add figure showing the effect of anomaly identification

**Minimum Width (m)**

.. autoproperty:: peak_finder.params.PeakFinderParams.min_width

.. todo::

    Update docstring and add figure showing the effect of anomaly identification

**Max Peak Migration**

.. autoproperty:: peak_finder.params.PeakFinderParams.max_migration

.. todo::

    Update docstring and add figure showing the effect of anomaly identification

**Minimum # Channels**

.. autoproperty:: peak_finder.params.PeakFinderParams.min_channels

.. todo::

    Update docstring and add figure showing the effect of anomaly identification

**Merge N Peaks**

.. autoproperty:: peak_finder.params.PeakFinderParams.n_groups

.. todo::

    Update docstring and add figure showing the effect of anomaly identification

**Max Group Separation**

.. autoproperty:: peak_finder.params.PeakFinderParams.max_separation

.. todo::

    Update docstring and add figure showing the effect of anomaly identification

**Save as**

.. autoproperty:: peak_finder.params.PeakFinderParams.ga_group_name

.. todo::

   Update docstring and add figure showing resulting object saved in GA.

**Output Path**

Provide absolute path to save the output to.

**Geoscience ANALYST Pro - Live link**

If selected the output will be imported to the open GA sessions geoh5 file.

**EXPORT**

Saves the result


