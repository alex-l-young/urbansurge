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

Load an EPA SWMM model with the ``urbansurge.swmm_model.SWMM`` class:

.. code-block:: python

   from urbansurge.swmm_model import SWMM

   cfg_path = './swmm_model_cfg.yml' # Path to configuration file.

   swmm = SWMM(cfg_path) # Create class instance.
   swmm.configure_model() # Configure model.
   swmm.run_simulation() # Run simulation.


.. _config_file:

Configuration File
--------

Structure of an UrbanSurge SWMM configuration file.

.. code-block:: yaml

   # =================================================================
   # Configuration file for model run.
   # =================================================================

   # Path to SWMM .inp file.
   inp_path: ~/Path/to/swmm_model.inp

   # Run from temporary .inp file. True or False.
   temp_inp: True

   # Verbosity. 0 or 1.
   verbose: 1