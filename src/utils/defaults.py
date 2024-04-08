Defaults = {
    "common":{
        "out_dir":"./DeepDiagnosticsResources/results/", 
        "temp_config": "./DeepDiagnosticsResources/temp/temp_config.yml", 
        "sim_location": "DeepDiagnosticsResources_Simulators"
    }, 
    "model": {
        "model_engine": "SBIModel"
    }, 
    "data":{
        "data_engine": "H5Data"
    },
    "plots_common": {
        "axis_spines": False, 
        "tight_layout": True,
        "colorway": "virdids", 
        "plot_style": "fast"
    }, 
    "plots":{
        "type_of_plot":{"specific_kwargs"}
    }, 
    "metrics_common": {
        "use_progress_bar": False,
        "samples_per_inference":1000, 
        "percentiles":[75, 85, 95]

    },
    "metrics":{
        "AllSBC":{}, 
        "CoverageFraction": {}, 
    }
}