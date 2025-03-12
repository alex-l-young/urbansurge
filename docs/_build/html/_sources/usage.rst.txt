Usage
=====

.. _installation:

Installation
------------

To use UrbanSurge, first install it using pip:

.. code-block:: console

   (.venv) $ pip install urbansurge

.. _epa_swmm:

EPA SWMM
--------

You can use the ``urbansurge.swmm_model.SWMM`` class:

.. py:function:: urbansurge.swmm_model.SWMM.get_component_names(self, section)
    :noindex:

    Returns the names of all components for a given section.

    :param section: Section name.
    :return: List of component names.