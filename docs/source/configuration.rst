Configuration 
===============

The configuration file is a `.yaml` file that makes up the majority of the settings. 
It is specified by the user, and if a field is not set, it falls back to a set of pre-defined defaults. 
It is split into sections to easily organize different parameters.

Specifying the Configuration 
-----------------------------

* Pipeline Mode 

To run diagnostics via the command line, pass the path to a yaml file to the diagnostic command. 
This will run the entire set of diagnostics according to the configuration file. 

.. code-block:: bash 

    diagnose --config path/to/your/config.yaml

* Standalone Mode 
The configuration file is not strictly required for running in standalone mode, 
but it can be specified to quickly access variables to avoid re-writing initialization parameters or ensure repeatability. 

.. code-block:: python 

    from DeepDiagnostics.utils.configuration import Config 


    Config("path/to/your/config.yaml")


Configuration Description 
-----------------------

.. attribute:: common

    :param out_dir: Folder where the results of program are saved. The path need not exist, it will be created if it does not.
    
    :param temp_config: Path to a yaml to store a temporary config. Used only if some arguments are specified outside the config (eg, if using both the --config and --model_path arguments)
    
    :param sim_location: Path to store settings for simulations. When using the register_simulator method, this is where the registered simulations are catalogued. 
    
    :param random_seed: Integer random seed to use. 

.. code-block:: yaml 

    common: 
        out_dir: "./DeepDiagnosticsResources/results/"
        temp_config: "./DeepDiagnosticsResources/temp/temp_config.yml"
        sim_location: "DeepDiagnosticsResources/simulators"
        random_seed: 42

.. attribute:: model

    :param model_path: Path to stored model. Required. 

    :param model_engine: Loading method to use. Choose from methods listed in :ref:`plots<plots>`

.. code-block:: yaml 

    model: 
        model_path: {No Default}
        model_engine: "SBIModel"

.. attribute:: data

    :param data_path: Path to stored data. Required.

    :param data_engine: Loading method to use. Choose from methods listed in  :ref:`plots<plots>`

    :param simulator: String name of the simulator. Must be pre-registered .

    :param prior: Prior distribution used in training. Used if "prior" is not included in the passed data. Choose from []
    
    :param prior_kwargs: kwargs to use with the initialization of the prior

    :param simulator_kwargs: kwargs to use with the initialization of the simulation

    :param simulator_dimensions: If the output of the simulation is 1D (non-image) or 2D (images.)

.. code-block:: yaml 

    data: 
        data_path: {No Default}
        data_engine: "H5Data"
        prior: "normal"
        prior_kwargs: {No Default}
        simulator_kwargs: {No Default}
        simulator_dimensions: 1

.. attribute:: plots_common

    :param axis_spines: Show axis ticks

    :param tight_layout:  Minimize the space between axes and labels

    :param default_colorway:  String colorway to use. Choose from `matplotlib's named colorways <https://matplotlib.org/stable/users/explain/colors/colormaps.html>`_.
    
    :param plot_style: Style sheet. Choose form `matplotlib's style sheets <https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html>`_.
    
    :param parameter_labels: Name of each theta parameter to use for titling and labels. Corresponding with the dim=1 axis of theta given by data. 
    
    :param parameter_colors: Colors to use for each theta parameters when representing the parameters on the same plot. 
    
    :param line_style_cycle: Line styles that can be used (besides for solid lines, which are always used.)
    
    :param figure_size: Default size for square figures. Will be adapted (slightly expanded) for multi-plot figures.

.. code-block:: yaml 

    plots_common: 
        axis_spines: False
        tight_layout: True
        default_colorway: "viridis"
        plot_style: "fast"
        parameter_labels: ["$m$", "$b$"]
        parameter_colors: ["#9C92A3", "#0F5257"]
        line_style_cycle: ["-", "-."]
        figure_size: [6, 6]

.. attribute:: metrics_common

    These parameters are used for every metric calculated, and for plots that require new inference to be run. 

    :param use_progress_bar: Show a progress bar when iteratively performing inference. 

    :param samples_per_inference: Number of samples used in a single iteration of inference. 

    :param percentiles: List of integer percentiles, for defining coverage regions. 

    :param number_simulations: Number of different simulations to run. Often, this means that the number of inferences performed for a metric is samples_per_inference*number_simulations

.. code-block:: yaml

    metrics_common:
        use_progress_bar: False
        samples_per_inference: 1000
        percentiles: [75, 85, 95]
        number_simulations: 50


.. attribute:: plots

    A dictionary of different plots to generate and their arguments. 
    Can be any of the implemented plots listed in :ref:`plots<plots>`
    If the plots are specified with an empty dictionary, defaults from the class are used.
    Defaults: ["CDFRanks", "Ranks", "CoverageFraction", "TARP", "LC2ST", "PPC"]

.. code-block:: yaml

    plots: 
        TARP: {} 
        

.. attribute:: metrics

    A dictionary of different metrics to generate and their arguments. 
    Can be any of the implemented plots listed in :ref:`metrics<metrics>`
    If the metrics are specified with an empty dictionary, defaults from the class are used.
    Defaults: ["AllSBC", "CoverageFraction", "LC2ST"]
    
.. code-block:: yaml

    metrics: 
        LC2ST: {} 
