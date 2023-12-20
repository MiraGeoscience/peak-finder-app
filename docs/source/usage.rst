.. _usage:

Basic usage
===========

The ui.json for the peak-finder application can operate in two modes:

 - :ref:`Interactive <interactive_application>`
 - :ref:`Standalone <standalone_application>`

If the *run interactive app* checkbox is selected, the ui.json parameters will be
used to initialize an interactive application.  In this mode, any changes made
through the ui will override the values set in the ui.json.  Without the
*run interactive app* checkbox selected, the application will be run in a terminal
with the parameters set in the ui.json.

ui.json interface
~~~~~~~~~~~~~~~~~

The ui.json interface has two sections:

.. _General Parameters:

1. General Parameters
_____________________

The general parameters must be set before running the application, regardless of
the mode of operation.

.. figure:: /images/usage/general_parameters.png
    :scale: 40%

The *Object* parameter is required to provide the geoh5 object containing the data.
The object must contain a *Line Field* referenced data, and at least one
*Property Group* group that must be selected here.  The *Color* property of the
property group may be altered here, but a default value is provided that will
suffice in most cases.

2. Optional Parameters
______________________


These parameters are generally not necessary to run the application in interactive mode.

.. figure:: /images/usage/optional_parameters.png
    :scale: 40%

Note that the group parameters have no equivalent in the interactive application,
and must be set in the optional parameters tab.  If a user wishes to find peaks in more
than one group of time channels, say, they must add the groups here.

.. _grouping parameters:

.. figure:: /images/usage/groups.png
   :scale: 40%

   Groups can be added by checking the box to enable and selecting a property group and
   color representation.

To learn more about the algorithm and its parameters, see the :ref:`methodology <methodology>`
section.  To see how the parameters are exposed in each of the application modes visit the
:ref:`interactive <interactive_application>` and :ref:`standalone <standalone_application>`
pages
