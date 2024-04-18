|coverage| |maintainability| |precommit_ci| |docs| |style| |version| |status| |pyversions|


.. |docs| image:: https://readthedocs.org/projects/peak-finder-app/badge/
    :alt: Documentation Status
    :target: https://peak-finder-app.readthedocs.io/en/latest/?badge=latest

.. |coverage| image:: https://codecov.io/gh/MiraGeoscience/peak-finder-app/branch/develop/graph/badge.svg
    :alt: Code coverage
    :target: https://codecov.io/gh/MiraGeoscience/peak-finder-app

.. |style| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :alt: Coding style
    :target: https://github.com/pf/black

.. |version| image:: https://img.shields.io/pypi/v/peak-finder-app.svg
    :alt: version on PyPI
    :target: https://pypi.python.org/pypi/peak-finder-app/

.. |status| image:: https://img.shields.io/pypi/status/peak-finder-app.svg
    :alt: version status on PyPI
    :target: https://pypi.python.org/pypi/peak-finder-app/

.. |pyversions| image:: https://img.shields.io/pypi/pyversions/peak-finder-app.svg
    :alt: Python versions
    :target: https://pypi.python.org/pypi/peak-finder-app/

.. |precommit_ci| image:: https://results.pre-commit.ci/badge/github/MiraGeoscience/peak-finder-app/develop.svg
    :alt: pre-commit.ci status
    :target: https://results.pre-commit.ci/latest/github/MiraGeoscience/peak-finder-app/develop

.. |maintainability| image:: https://api.codeclimate.com/v1/badges/_token_/maintainability
   :target: https://codeclimate.com/github/MiraGeoscience/peak-finder-app/maintainability
   :alt: Maintainability


peak-finder-app: # TODO: SHORT DESCRIPTION
=========================================================================
**peak-finder-app** is a package for the detection and grouping of time-domain
electromagnetic (TEM) anomalies measured along flight lines.

.. contents:: Table of Contents
   :local:
   :depth: 3

Documentation
^^^^^^^^^^^^^
`Online documentation <https://mirageoscience-peak-finder-app.readthedocs-hosted.com/en/latest/>`_


Installation
^^^^^^^^^^^^
**peak-finder-app** is currently written for Python 3.10 or higher.

Install Conda
-------------

To install **peak-finder-app**, you need to install **Conda** first.

We recommend to install **Conda** using `miniforge`_.

.. _miniforge: https://github.com/conda-forge/miniforge

Within a conda environment
--------------------------

You can install (or update) a conda environment with all the requires packages to run **peak-finder-app**.
To do so you can directly run the **Install_or_Update.bat** file by double left clicking on it.

Install with conda
------------------

You can install the package using ``conda`` and the ``.lock`` files from a conda prompt:

.. code-block:: bash

  conda env create -n my-env -f environments/[the_desired_env].lock.yml

Install with PyPI
-----------------

You should not install the package from PyPI, as the app requires conda packages to run.
Still, you can install it in a conda environment without its dependencies (``--no-deps``).

From PyPI
~~~~~~~~~

To install the **peak-finder-app** package published on PyPI:

.. code-block:: bash

    pip install -U --no-deps peak-finder-app

From a Git tag or branch
~~~~~~~~~~~~~~~~~~~~~~~~
If the package is not on PiPY yet, you can install it from a Git tag:

.. code-block:: bash

    pip install -U --no-deps --force-reinstall https://github.com/MiraGeoscience/peak-finder-app/archive/refs/tags/TAG.zip

Or to install the latest changes available on a given Git branch:

.. code-block:: bash

    pip install -U --no-deps --force-reinstall https://github.com/MiraGeoscience/peak-finder-app/archive/refs/heads/BRANCH.zip

.. note::
    The ``--force-reinstall`` option is used to make sure the updated version
    of the sources is installed, and not the cached version, even if the version number
    did not change. The ``-U`` or ``--upgrade`` option is used to make sure to get the latest version,
    on not merely reinstall the same version. As the package is aimed to be in a **Conda environment**, the option ``--no-deps`` is used to avoid installing the dependencies with pip, as they will be installed with conda.

From a local copy of the sources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you have a git clone of the package sources locally,
you can install **peak-finder-app** from the local copy of the sources with:

.. code-block:: bash

    pip install -U --force-reinstall path/to/project_folder_with_pyproject_toml

Or in **editable mode**, so that you can edit the sources and see the effect immediately at runtime:

.. code-block:: bash

    pip install -e -U --force-reinstall path/to/project_folder_with_pyproject_toml

Setup for development
^^^^^^^^^^^^^^^^^^^^^
To configure the development environment and tools, please see `README-dev.rst`_.

.. _README-dev.rst: README-dev.rst

License
^^^^^^^
# TODO: ADD LICENSE TERMS

Third Party Software
^^^^^^^^^^^^^^^^^^^^
The peak-finder-app Software may provide links to third party libraries or code (collectively “Third Party Software”)
to implement various functions. Third Party Software does not comprise part of the Software.
The use of Third Party Software is governed by the terms of such software license(s).
Third Party Software notices and/or additional terms and conditions are located in the
`THIRD_PARTY_SOFTWARE.rst`_ file.

.. _THIRD_PARTY_SOFTWARE.rst: THIRD_PARTY_SOFTWARE.rst

Copyright
^^^^^^^^^
Copyright (c) 2024 Mira Geoscience Ltd.
