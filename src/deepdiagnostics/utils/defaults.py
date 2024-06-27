Defaults = {
    "common": {
        "out_dir": "./DeepDiagnosticsResources/results/",
        "temp_config": "./DeepDiagnosticsResources/temp/temp_config.yml",
        "sim_location": "./DeepDiagnosticsResources/simulators",
        "random_seed": 42,
    },
    "model": {"model_engine": "SBIModel"},
    "data": {
        "data_engine": "H5Data",
        "prior": "normal",
        "prior_kwargs": None,
        "simulator_kwargs": None,
        "simulator_dimensions": 1,
    },
    "plots_common": {
        "axis_spines": False,
        "tight_layout": True,
        "default_colorway": "viridis",
        "plot_style": "fast",
        "parameter_labels": ["$m$", "$b$"],
        "parameter_colors": ["#9C92A3", "#0F5257"],
        "line_style_cycle": ["-", "-."],
        "figure_size": [6, 6],
    },
    "plots": {
        "CDFRanks": {},
        "Ranks": {"num_bins": None},
        "CoverageFraction": {},
        "TARP": {
            "coverage_sigma": 3 
        },
        "LC2ST": {}, 
        "Parity":{},
        "PPC": {}, 
        "PriorPC":{}
    },
    "metrics_common": {
        "use_progress_bar": False,
        "samples_per_inference": 1000,
        "percentiles": [75, 85, 95],
        "number_simulations": 50,
    },
    "metrics": {
        "AllSBC": {},
        "CoverageFraction": {},
        "LC2ST":{}
    },
}
