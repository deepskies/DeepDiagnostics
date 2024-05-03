import os 
import pytest 

from utils.defaults import Defaults 
from utils.config import Config, get_item
from plots import (
    Plots, 
    CDFRanks, 
    Ranks, 
    CoverageFraction, 
    TARP
)

@pytest.fixture
def plot_config(config_factory): 
    out_dir = "./temp_results/"
    metrics_settings={"use_progress_bar":False, "samples_per_inference":10, "percentiles":[95]}
    config = config_factory(out_dir=out_dir, metrics_settings=metrics_settings)
    Config(config)


def test_all_plot_catalogued(): 
    '''Each metrics gets its own file, and each metric is included in the Metrics dictionary 
    so the client can use it. 
    This test verifies all metrics are cataloged'''

    all_files = os.listdir("src/plots/")
    files_ignore = ['plot.py', '__init__.py', '__pycache__'] # All files not containing a metric 
    num_files = len([file for file in all_files if file not in files_ignore])
    assert len(Plots) == num_files

def test_all_defaults(plot_config, mock_model, mock_data): 
    """
    Ensures each metric has a default set of parameters and is included in the defaults list
    Ensures each test can initialize, regardless of the veracity of the output 
    """
    for plot_name, plot_obj in Plots.items(): 
        assert plot_name in Defaults['plots']
        plot_obj(mock_model, mock_data,  save=True, show=False)

def test_plot_cdf(plot_config, mock_model, mock_data): 
    plot = CDFRanks(mock_model, mock_data, save=True, show=False)
    plot(**get_item("plots", "CDFRanks", raise_exception=False)) 
    assert os.path.exists(f"{plot.out_path}/{plot.plot_name}")

def test_plot_ranks(plot_config, mock_model, mock_data): 
    plot = Ranks(mock_model, mock_data, save=True, show=False)
    plot(**get_item("plots", "Ranks", raise_exception=False))
    assert os.path.exists(f"{plot.out_path}/{plot.plot_name}")

def test_plot_coverage(plot_config, mock_model, mock_data): 
    plot = CoverageFraction(mock_model, mock_data, save=True, show=False)
    plot(**get_item("plots", "CoverageFraction", raise_exception=False))
    assert os.path.exists(f"{plot.out_path}/{plot.plot_name}")

def test_plot_tarp(plot_config, mock_model, mock_data): 
    plot = TARP(mock_model, mock_data, save=True, show=False)
    plot(**get_item("plots", "TARP", raise_exception=False))