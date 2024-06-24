Client 
========

.. note:: 
    When running the client, you can supply **either** the configuration yaml file, or the CLI arguments. 
    You do not need to supply both. 

Use the command `diagnose -h` to view all usage of the CLI helper at any time.  
Specific argument descriptions and explanations can be found on the :ref:`configuration` page. 

.. code-block:: bash
    
    usage: diagnose [-h] [--config CONFIG] [--model_path MODEL_PATH] [--model_engine {SBIModel}] [--data_path DATA_PATH] [--data_engine {H5Data,PickleData}]
                    [--simulator SIMULATOR] [--out_dir OUT_DIR] [--metrics [{CoverageFraction,AllSBC,LC2ST}]]
                    [--plots [{CDFRanks,CoverageFraction,Ranks,TARP,LC2ST,PPC}]]

    options:
    -h, --help            show this help message and exit
    --config CONFIG, -c CONFIG
                            .yaml file with all arguments to run.
    --model_path MODEL_PATH, -m MODEL_PATH
                            String path to a model. Must be compatible with your model_engine choice.
    --model_engine {SBIModel}, -e {SBIModel}
                            Way to load your model. See each module's documentation page for requirements and specifications.
    --data_path DATA_PATH, -d DATA_PATH
                            String path to data. Must be compatible with data_engine choice.
    --data_engine {H5Data,PickleData}, -g {H5Data,PickleData}
                            Way to load your data. See each module's documentation page for requirements and specifications.
    --simulator SIMULATOR, -s SIMULATOR
                            String name of the simulator to use with generative metrics and plots. Must be pre-register with the `utils.register_simulator` method.
    --out_dir OUT_DIR     Where the results will be saved. Path need not exist, it will be created.
    --metrics [{CoverageFraction,AllSBC,LC2ST}]
                            List of metrics to run. To not run any, supply `--metrics `
    --plots [{CDFRanks,CoverageFraction,Ranks,TARP,LC2ST,PPC}]
                            List of plots to run. To not run any, supply `--plots `