import pytest 
import yaml 
import numpy as np 

from data import H5Data
from data.simulator import Simulator
from models import SBIModel
from utils.register import register_simulator


class MockSimulator(Simulator): 
    def __init__(self): 
        pass 

    def __call__(self, thetas, samples): 
        thetas = np.atleast_2d(thetas)
        # Check if the input has the correct shape
        if thetas.shape[1] != 2:
            raise ValueError("Input tensor must have shape (n, 2) where n is the number of parameter sets.")

        # Unpack the parameters
        if thetas.shape[0] == 1:
            # If there's only one set of parameters, extract them directly
            m, b = thetas[0, 0], thetas[0, 1]
        else:
            # If there are multiple sets of parameters, extract them for each row
            m, b = thetas[:, 0], thetas[:, 1]
        x = np.linspace(0, 100, samples)
        rs = np.random.RandomState()#2147483648)# 
        # I'm thinking sigma could actually be a function of x
        # if we want to get fancy down the road
        # Generate random noise (epsilon) based on a normal distribution with mean 0 and standard deviation sigma
        sigma = 1
        epsilon = rs.normal(loc=0, scale=sigma, size=(len(x), thetas.shape[0]))
        
        # Initialize an empty array to store the results for each set of parameters
        y = np.zeros((len(x), thetas.shape[0]))
        for i in range(thetas.shape[0]):
            m, b = thetas[i, 0], thetas[i, 1]
            y[:, i] = m * x + b + epsilon[:, i]
        return y.T


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
            config['plots_common'] = plot_settings
        if metrics_settings is not None: 
            config['metrics_common'] = metrics_settings

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