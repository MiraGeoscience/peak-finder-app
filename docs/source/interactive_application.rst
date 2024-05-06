.. _interactive_application:

Interactive Application
=======================

If you are using **peak-finder** through the interactive application you will see
a window with visualizations and ui controls.  The visualization contains both a
section of data along a single line and a plan view to locate the chosen and
adjacent lines and anomaly picks in cartesian space.

.. figure:: /images/interactive/visualizations.png
    :scale: 40%

The ui controls section is divided into three subsections:

- `Visual Parameters`_
- `Data Selection`_
- `Detection Parameters`_

.. figure:: /images/interactive/ui_controls.png
    :scale: 40%



Visual Parameters
~~~~~~~~~~~~~~~~~

This section controls the appearance of the plotting area.

.. figure:: /images/parameters/visual_parameters.png
   :align: left

X-axis Label
____________

Updates the label on the data section view x-axis.

.. figure:: /images/parameters/visualization/section_xlabel.png

   The x-axis label is updated to reflect the selection.

Y-axis Scaling
______________

Updates the scaling of the y-axis of the data section view

.. todo::

   Add a figure showing data with both linear and symlog scaling

Linear threshold
________________

When Symlog is chosen for Y-axis scaling, this parameter will set the
region in which linear scaling is used.

.. figure:: /images/parameters/visualization/linthresh_compare.png
   :scale: 60%

   Comparing the data visualization with a symlog linear threshold set to
   10E-3.2 (left) and 10E-5.1 (right).

Plot residuals
______________

Switches on and off the residual visualization that shows the difference
between the raw and smoothed data.  See the :ref:`Smoothing` section for more details.

.. figure:: /images/parameters/visualization/residuals.png
   :scale: 40%

   The residual layer is used to show the effect of the smoothing factor.

Plot markers
____________

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

Lines Field
___________

.. autoproperty:: peak_finder.params.PeakFinderParams.line_field

Select Line
___________

.. autoproperty:: peak_finder.params.PeakFinderParams.line_id

.. todo::

   Add a figure showing the plan view line selection (black).

:ref:`Masking Data`
___________________

.. todo::

   Add a figure of a working masked result.

N outward lines
_______________

Includes N lines in plan view on either side of the selected line.

.. figure:: /images/parameters/data_selection/outward_line_compare.png
   :scale: 40%

   Comparing the plan view with 1 outward line (left) and 2 outward lines
   (right).

Flip Y (-1x)
____________

.. autoproperty:: peak_finder.params.PeakFinderParams.flip_sign

.. todo::

   Update docstring and add figure showing the effect of flipping y.

Select group colors
___________________

.. todo::

   Add figure of color picker widget.  Move this ui to visualization group?

Detection Parameters
~~~~~~~~~~~~~~~~~~~~

The detection parameters are those that the peak-finder application uses to
tune the characterization and detection of anomalies within the data.  Most
of these are already described in the :ref:`Methodology` section.  Follow
the links for detailed descriptions of each parameter.

.. figure:: /images/parameters/detection_parameters.png
   :align: left


- :ref:`Smoothing window <Smoothing>`
- :ref:`Minimum Amplitude (%) <Minimum Amplitude>`
- :ref:`Minimum Value <Minimum Data Value>`
- :ref:`Minimum Width (m) <Minimum Width>`
- :ref:`Max Peak Migration (m) <Maximum Peak Migration>`
- :ref:`Minimum # Channels <Minimum number of channels>`
- :ref:`Merge # Peaks <Merge N Peaks>`
- :ref:`Max Separation <Max Group Separation>`


Save as
_______

.. autoproperty:: peak_finder.params.PeakFinderParams.ga_group_name

.. todo::

   Update docstring and add figure showing resulting object saved in GA.

Output Path
___________

Provide absolute path to save the output to.

Geoscience ANALYST Pro - Live link
__________________________________

If selected the output will be imported to the open GA sessions geoh5 file.

EXPORT
______

Saves the result
