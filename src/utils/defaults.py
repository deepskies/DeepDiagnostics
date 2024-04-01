Defaults = {
    "common":{
        "out_dir":"./DeepDiagonisticsResources/results/", 
        "temp_config": "./DeepDiagonisticsResources/temp/temp_config.yml", 
        "sim_location": "DeepDiagonisticsResources_Simulators"
    }, 
    "model": {
        "model_engine": "SBIModel"
    }, 
    "data":{
        "data_engine": "H5Data"
    },
    "plot_common": {
        "axis_spines": False, 
        "tight_layout": True,
        "colorway": "virdids", 
        "plot_style": "fast"
    }, 
    "plots":{
        "type_of_plot":{"specific_kwargs"}
    }, 
    "metric_common": {

    },
    "metrics":{
        "type_of_metrics":{"specific_kwargs"}
    }
}