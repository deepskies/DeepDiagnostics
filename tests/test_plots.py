import os
import pytest

from utils.defaults import Defaults
from utils.config import Config, get_item

from plots import (
    Plots, 
    CDFRanks, 
    Ranks, 
    CoverageFraction, 
    TARP, 
    LocalTwoSampleTest, 
    PPC
)


@pytest.fixture
def plot_config(config_factory):
    metrics_settings = {
        "use_progress_bar": False,
        "samples_per_inference": 10,
        "percentiles": [95],
    }
    config = config_factory(metrics_settings=metrics_settings)
    return config

  
def test_all_defaults(plot_config, mock_model, mock_data): 
    """
    Ensures each metric has a default set of parameters and is included in the defaults list
    Ensures each test can initialize, regardless of the veracity of the output
    """
    Config(plot_config)
    for plot_name, plot_obj in Plots.items():
        assert plot_name in Defaults["plots"]
        plot_obj(mock_model, mock_data, save=True, show=False)

def test_plot_cdf(plot_config, mock_model, mock_data):
    Config(plot_config)
    plot = CDFRanks(mock_model, mock_data, save=True, show=False)
    plot(**get_item("plots", "CDFRanks", raise_exception=False))
    assert os.path.exists(f"{plot.out_dir}/{plot.plot_name}")

def test_plot_ranks(plot_config, mock_model, mock_data):
    Config(plot_config)
    plot = Ranks(mock_model, mock_data, save=True, show=False)
    plot(**get_item("plots", "Ranks", raise_exception=False))
    assert os.path.exists(f"{plot.out_dir}/{plot.plot_name}")

def test_plot_coverage(plot_config, mock_model, mock_data):
    Config(plot_config)
    plot = CoverageFraction(mock_model, mock_data, save=True, show=False)
    plot(**get_item("plots", "CoverageFraction", raise_exception=False))
    assert os.path.exists(f"{plot.out_dir}/{plot.plot_name}")

def test_plot_tarp(plot_config, mock_model, mock_data):
    Config(plot_config)
    plot = TARP(mock_model, mock_data, save=True, show=False)
    plot(**get_item("plots", "TARP", raise_exception=False))
    assert os.path.exists(f"{plot.out_dir}/{plot.plot_name}")

def test_lc2st(plot_config, mock_model, mock_data, mock_2d_data, result_output):
    Config(plot_config)
    plot = LocalTwoSampleTest(mock_model, mock_data, save=True, show=False)
    plot(**get_item("plots", "LC2ST", raise_exception=False))
    assert os.path.exists(f"{plot.out_dir}/{plot.plot_name}")

    plot = LocalTwoSampleTest(
        mock_model, mock_2d_data, save=True, show=False, 
        out_dir=f"{result_output.strip('/')}/mock_2d/")
    assert type(plot.data.simulator).__name__ == "Mock2DSimulator"
    plot(**get_item("plots", "LC2ST", raise_exception=False))
    assert os.path.exists(f"{plot.out_dir}/{plot.plot_name}")

def test_ppc(plot_config, mock_model, mock_data, mock_2d_data, result_output):
    Config(plot_config)
    plot = PPC(mock_model, mock_data, save=True, show=False)
    plot(**get_item("plots", "PPC", raise_exception=False))
    assert os.path.exists(f"{plot.out_dir}/{plot.plot_name}")

    plot = PPC(
        mock_model,
        mock_2d_data, save=True, show=False, 
        out_dir=f"{result_output.strip('/')}/mock_2d/")
    assert type(plot.data.simulator).__name__ == "Mock2DSimulator"
    plot(**get_item("plots", "PPC", raise_exception=False))
    assert os.path.exists(f"{plot.out_dir}/{plot.plot_name}")