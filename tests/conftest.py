import pytest 
import yaml 

from data import H5Data, Simulator
from models import SBIModel
from utils.register import register_simulator


class MockSimulator(Simulator): 
    def __init__(self): 
        pass 
    def __call__(self, thetas): 
        return thetas


@pytest.fixture
def model_path(): 
    return "resources/savedmodels/sbi/sbi_linear_from_data.pkl"

@pytest.fixture
def data_path(): 
    return "resources/saveddata/data_validation.h5"

@pytest.fixture 
def simulator_name():
    name = MockSimulator.__name__
    register_simulator(name, MockSimulator)
    return name

@pytest.fixture 
def mock_model(model_path): 
    return SBIModel(model_path)

@pytest.fixture
def mock_data(data_path, simulator_name): 
    return H5Data(data_path, simulator_name)

@pytest.fixture
def config_factory(): 
    def factory(
        out_dir=None, 
        model_path=None, 
        model_engine=None, 
        data_path=None, 
        data_engine=None, 
        simulator=None, 
        plot_settings=None, 
        metrics_settings=None, 
        plots=None, 
        metrics=None
):
        config = { "common": {}, "model": {}, "data":{}, "plot_common": {}, "plots":{}, "metric_common": {},"metrics":{}}
        
        # Single settings 
        if out_dir is not None: 
            config["common"]['out_dir'] = out_dir
        if model_path is not None: 
            config['model']['model_path'] = model_path
        if model_engine is not None: 
            config['model']['model_engine'] = model_engine
        if data_path is not None: 
            config['data']['data_path'] = data_path
        if data_engine is not None: 
            config['data']['data_engine'] = data_engine
        if simulator is not None: 
            config['data']['simulator'] = simulator

        # Dict settings
        if plot_settings is not None: 
            config['plot_common'] = plot_settings
        if metrics_settings is not None: 
            config['metric_common'] = metrics_settings

        if metrics is not None: 
            if isinstance(metrics, dict):
                config["metrics"] = metrics
            if isinstance(metrics, list): 
                config["metrics"] = {metric: {} for metric in metrics}

        if plots is not None: 
            if isinstance(plots, dict):
                config["plots"] = plots
            if isinstance(metrics, list): 
                config["plots"] = {plot: {} for plot in plots}

        temp_outpath = "./temp_config.yml"
        yaml.dump(config, open(temp_outpath, "w"))

        return temp_outpath
    
    return factory