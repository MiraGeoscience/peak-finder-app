.. _usage:

Basic usage
===========

ui.json interface
~~~~~~~~~~~~~~~~~

The ui.json interface has two sections:

**1. General Parameters**

The general parameters must be set before running the application.  They often
consist of data selectors for the geoh5 objects the application operates on.

.. figure:: /images/usage/general_parameters.png
    :scale: 40%

**2. Optional Parameters**

These parameters are not necessary to run the application.

.. figure:: /images/usage/optional_parameters.png
    :scale: 40%

The ui.json for the peak-finder application behaves different than other applications.
If the *run interactive app* checkbox is selected the optional parameters will be
used to initialize the application, but any changes made through the ui will override
the values set in the optional parameters tab.

Note that the group parameters have no equivalent in the ui, and must be set in the
optional parameters tab.

.. figure:: /images/usage/groups.png
    :scale: 40%

If you are using **peak-finder** through the interface you will see a window with
visualizations and ui controls.  The visualization contains botha section of data
along a single line and a plan view to locate the chosen and adjacent lines and
anomaly picks in cartesian space.

.. figure:: /images/usage/visualizations.png
    :scale: 40%

The ui controls section is divided into three subsections: visualization parameters,
data selection, and detection parameters.

.. figure:: /images/usage/ui_controls.png
    :scale: 40%

To learn more about the different ui controls and how they affect the visualization,
data selection and detection process see the :ref:`parameters` section.





