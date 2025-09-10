import os 
import shutil
import uuid
import pytest
import yaml
import numpy as np
import sys

from deepdiagnostics.data import H5Data, H5HierarchyData
from deepdiagnostics.data.simulator import Simulator
from deepdiagnostics.models import SBIModel, HierarchyModel
from deepdiagnostics.utils.config import get_item
from deepdiagnostics.utils.simulator_utils import register_simulator


class MockSimulator(Simulator):
    def generate_context(self, n_samples: int) -> np.ndarray:
        return np.linspace(0, 100, n_samples)

    def simulate(self, theta: np.ndarray, context_samples: np.ndarray) -> np.ndarray:
        thetas = np.atleast_2d(theta)
        if thetas.shape[1] != 2:
            raise ValueError(
                "Input tensor must have shape (n, 2) where n is the number of parameter sets."
            )

        if thetas.shape[0] == 1:
            # If there's only one set of parameters, extract them directly
            m, b = thetas[0, 0], thetas[0, 1]
        else:
            # If there are multiple sets of parameters, extract them for each row
            m, b = thetas[:, 0], thetas[:, 1]
        rs = np.random.RandomState()
        sigma = 1
        epsilon = rs.normal(
            loc=0, scale=sigma, size=(len(context_samples), thetas.shape[0])
        )

        # Initialize an empty array to store the results for each set of parameters
        y = np.zeros((len(context_samples), thetas.shape[0]))
        for i in range(thetas.shape[0]):
            m, b = thetas[i, 0], thetas[i, 1]
            y[:, i] = m * context_samples + b + epsilon[:, i]
        return y.T

class Mock2DSimulator(Simulator): 
    def __init__(self) -> None:
        "Create a 2D simulator that just produces noise"

    def generate_context(self, n_samples: int) -> np.ndarray: 
        return np.linspace(0, 28, n_samples)

    def simulate(self, theta, context_samples: np.ndarray): 

        generated_stars = []
        if len(theta.shape) == 1: 
            theta = [theta]
            
        for sample_index, t in enumerate(theta): 
            mock_data = np.random.normal(
                loc=t[0], scale=abs(t[1]), size=(len(context_samples), len(context_samples))
            )
            generated_stars.append(
                np.column_stack((context_samples, mock_data))
            )
        return np.array(generated_stars)

@pytest.fixture(autouse=True)
def setUp(result_output):
    register_simulator("MockSimulator", MockSimulator)
    register_simulator("Mock2DSimulator", Mock2DSimulator)

    yield 
    
    simulator_config_path = get_item("common", "sim_location", raise_exception=False)
    sim_paths = f"{simulator_config_path.strip('/')}/simulators.json"
    os.remove(sim_paths)

    # shutil.rmtree(result_output, ignore_errors=True)

@pytest.fixture
def model_path():
    return "resources/savedmodels/sbi/sbi_linear_from_data.pkl"

@pytest.fixture
def hierarchy_model_path():
    return "resources/savedmodels/trained_model_0909.pkl"

@pytest.fixture
def hierarchy_model_nn_path():
    return "resources/savedmodels/hnpe_src"

@pytest.fixture
def data_path():
    return "resources/saveddata/data_test.h5"

@pytest.fixture
def hierarchy_data_path():
    return "resources/saveddata/Hierarchicalpendulum_data_test_40.h5"

@pytest.fixture
def result_output(): 
    path = "./temp_results/"
    if not os.path.exists(path): 
        os.makedirs(path)
    return path

@pytest.fixture
def simulator_name():
    return MockSimulator.__name__


@pytest.fixture
def mock_model(model_path):
    return SBIModel(model_path)

@pytest.fixture
def mock_hierarchy_model(hierarchy_model_path, hierarchy_model_nn_path):
    return HierarchyModel(hierarchy_model_path, hierarchy_model_nn_path)

@pytest.fixture
def mock_data(data_path, simulator_name):
    return H5Data(data_path, simulator_name)

@pytest.fixture
def mock_hierarchy_data(hierarchy_data_path, simulator_name):
    return H5HierarchyData(hierarchy_data_path, simulator_name)

@pytest.fixture
def mock_run_id(): 
    return str(uuid.uuid4()).replace('-', '')[:10]

@pytest.fixture
def mock_2d_data(data_path): 
    return H5Data(data_path, "Mock2DSimulator", simulation_dimensions=2)

@pytest.fixture
def config_factory(result_output, model_path, data_path, hierarchy_model_path, hierarchy_data_path):
    def factory(
        use_hierarchy: bool = False,
        model_engine=None,
        data_engine=None,
        plot_2d=False,
        simulator=None,
        plot_settings=None,
        metrics_settings=None,
        plots=None,
        metrics=None,
    ):
        cfg = {
            "common": {"out_dir": result_output},
            "model": {},
            "data": {},
            "plots_common": {},
            "plots": {},
            "metrics_common": {},
            "metrics": {},
        }

        mp = hierarchy_model_path if use_hierarchy else model_path
        dp = hierarchy_data_path if use_hierarchy else data_path
        cfg["model"]["model_path"] = str(mp)
        cfg["data"]["data_path"] = str(dp)

        if model_engine is not None:
            cfg["model"]["model_engine"] = model_engine
        if data_engine is not None:
            cfg["data"]["data_engine"] = data_engine
        if simulator is not None:
            cfg["data"]["simulator"] = simulator
        if plot_2d:
            cfg["data"]["simulator_dimensions"] = 2

        if plot_settings:
            cfg["plots_common"].update(plot_settings)
        if metrics_settings:
            cfg["metrics_common"].update(metrics_settings)
        if metrics is not None:
            cfg["metrics"] = metrics if isinstance(metrics, dict) else {m: {} for m in metrics}
        if plots is not None:
            cfg["plots"] = plots if isinstance(plots, dict) else {p: {} for p in plots}

        temp_outpath = "./temp_config.yml"
        with open(temp_outpath, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return temp_outpath

    return factory
