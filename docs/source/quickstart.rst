Quickstart 
============

Installation 
--------------

* From PyPi 
.. code-block:: console 
    pip install DeepDiagnostics


* From Source 
.. code-block:: console 
    git clone https://github.com/deepskies/DeepDiagnostics/ 
    pip install poetry 
    poetry shell 
    poetry install


Pipeline 
---------

`DeepDiagnostics` includes a CLI tool for analysis. 
* To run the tool using a configuration file: 

.. code-block:: console 
    diagnose --config {path to yaml}


* To use defaults with specific models and data: 

.. code-block:: console 
    diagnose --model_path {model pkl} --data_path {data pkl} [--simulator {sim name}]


Additional arguments can be found using ``diagnose -h``

Standalone 
----

`DeepDiagnostics` comes with the option to run different plots and metrics independently. 
This requires setting a configuration file ahead of time, and then running the plots. 

All plots and metrics can be found in :ref:`plots<plots>` and :ref:`metrics<metrics>`. 



.. code-block:: python 

    from DeepDiagnostics.utils.configuration import Config 
    from DeepDiagnostics.model import SBIModel 
    from DeepDiagnostics.data import H5Data

    from DeepDiagnostics.plots import ...

    Config({configuration_path})
    model = SBIModel({model_path})
    data = H5Data({data_path})

    {Plot of choice} 
