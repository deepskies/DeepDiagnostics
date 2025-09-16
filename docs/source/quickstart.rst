Quickstart 
============

Notebook Example 
-----------------

`An example notebook can be found here for an interactive walkthrough <https://github.com/deepskies/DeepDiagnostics/blob/main/notebooks/example.ipynb>`_. 

Installation 
--------------

* From PyPi 
.. code-block:: bash

    pip install deepdiagnostics


* From Source 
.. code-block:: bash
    
    git clone https://github.com/deepskies/DeepDiagnostics/ 
    pip install poetry 
    poetry install 
    poetry run diagnose --help


Pre-requisites
-------------

DeepDiagnostics does not train models or generate data, they must be provided.
Possible model formats are listed in :ref:`models` and data formats in :ref:`data`.
If you are using a simulator, it must be registered by using `deepdiagnostics.utils.register.register_simulator`.
More information can be found in :ref:`custom_simulations`.

Output directories are automatically created, and if a run ID is not specified, one is generated. 
Only if a run ID is specified will previous runs be overwritten.

Configuration 
-------------

Description of the configuration file, including defaults, can be found in :ref:`configuration`. 
Below is a minimal example. 

...code-block:: yaml

    common: 
        out_dir: "./deepdiagnostics_results/"
        random_seed: 42
    data: 
        data_engine: "H5Data"
        data_path: "./data/my_data.h5"
        simulator: "MySimulator"
        simulator_kwargs: # Any augments used to initialize the simulator
            foo: bar
    model: 
        model_engine: "SBIModel"
        model_path: "./models/my_model.pkl"
    plots_common:   # Used across all plots
        parameter_labels:  # Can either be plain strings or rendered LaTeX strings
            - "My favorite parameter"
            - "My least favorite parameter"
            - "My most mid parameter"
        parameter_colors:   # Any color recognized by matplotlib
            - "#264a95"
            - "#ed9561"
            - "#89b7bb"
        line_style_cycle:  # Any line type recognized by matplotlib
            - solid
            - dashed
            - dotted
        figure_size:  # Approximate size, it can be scaled when adding additional subfigures
            - 6 # x length
            - 6 # y length
    metrics_common: # Used across all metrics (and plots if the plots have a calculation step)
        samples_per_inference: 1000
        number_simulations: 100
        percentiles: 
            - 68
            - 95
    plots:
        CoverageFraction:   # Arguments supplied to {plottype}.plot()
            include_coverage_std: True
            include_ideal_range: False
            reference_line_label: "Ideal Coverage"
        TARP: 
            coverage_sigma: 4
            title: "TARP of My Model"
    metrics:
        AllSBC
        Ranks:
            num_bins: 3


Pipeline 
---------

`DeepDiagnostics` includes a CLI tool for analysis. 
* To run the tool using a configuration file: 

.. code-block:: bash 

    diagnose --config {path to yaml}


* To use defaults with specific models and data: 

.. code-block:: bash

    diagnose --model_path {model pkl} --data_path {data pkl} [--simulator {sim name}]


Additional arguments can be found using ``diagnose -h``

Standalone 
----

`DeepDiagnostics` comes with the option to run different plots and metrics independently. 
This requires setting a configuration file ahead of time, and then running the plots. 

All plots and metrics can be found in :ref:`plots<plots>` and :ref:`metrics<metrics>`. 


.. code-block:: python 

    from deepdiagnostics.utils.configuration import Config 
    from deepdiagnostics.model import SBIModel 
    from deepdiagnostics.data import H5Data

    from deepdiagnostics.plots import LocalTwoSampleTest, Ranks

    Config({configuration_path})
    model = SBIModel({model_path})
    data = H5Data({data_path}, simulator={simulator name})

    LocalTwoSampleTest(data=data, model=model, show=True)(use_intensity_plot=False, n_alpha_samples=200)
    Ranks(data=data, model=model, show=True)(num_bins=3)


Custom Simulations
-------------------

To use generative model diagnostics, a simulator has to be included. 
This is done by `registering` your simulation with a name and a class associated. 

By doing this, the DeepDiagnostics can find your simulation at a later time and the simulation does not need to be loaded in memory at time of running the CLI pipeline or standalone modules.

.. code-block:: python 

    from deepdiagnostics.utils.register import register_simulator

    class MySimulation: 
        def __init__(...)
            ...
    

    register_simulator(simulator_name="MySimulation", simulator=MySimulation)


Simulations also require two different methods - `generate_context` (Which is used to either load or generate the non-theta input parameter for the simulation, also called `x`) and `simulate`. 
This is enforced by using the abstract class `deepdiagnostics.data.Simulator` as a parent class. 

.. code-block:: python 
    
    from deepdiagnostics.data import Simulator

    import numpy as np 


    class MySimulation(Simulator): 
        def generate_context(self, n_samples: int) -> np.ndarray:
            """Give a number of samples (int) and get a numpy array of random samples to be used for the simulation"""
            return np.random.uniform(0, 1)

        def simulate(self, theta: np.ndarray, context_samples: np.ndarray) -> np.ndarray:
            """Give the parameters of the simulation (theta), and x values (context_samples) and get a simulation sample.
            Theta and context should have the same shape for dimension 0, the number of samples."""
            simulation_results = np.zeros(theta.shape[0], 1)
            for index, context in enumerate(context_samples): 
                simulation_results[index] = theta[index][0]*context + theta[index][1]*context

            return simulation_results
