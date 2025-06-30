import os
import pytest

from deepdiagnostics.utils.config import Config, get_item

from deepdiagnostics.plots import (
    CDFRanks, 
    Ranks, 
    CoverageFraction, 
    TARP, 
    LC2ST, 
    PPC,
    PriorPC,
    Parity
)


@pytest.fixture
def plot_config(config_factory):
    metrics_settings = {
        "use_progress_bar": False,
        "samples_per_inference": 10,
        "percentiles": [95, 75, 50],
    }
    config = config_factory(metrics_settings=metrics_settings)
    return config


def test_plot_cdf(plot_config, mock_model, mock_data, mock_run_id):
    Config(plot_config)
    plot = CDFRanks(mock_model, mock_data, mock_run_id, save=True, show=False)
    plot(**get_item("plots", "CDFRanks", raise_exception=False))
    assert os.path.exists(f"{plot.out_dir}/{mock_run_id}_{plot.plot_name}")

def test_plot_ranks(plot_config, mock_model, mock_data, mock_run_id):
    Config(plot_config)
    plot = Ranks(mock_model, mock_data, mock_run_id, save=True, show=False)
    plot(**get_item("plots", "Ranks", raise_exception=False))
    assert os.path.exists(f"{plot.out_dir}/{mock_run_id}_{plot.plot_name}")

def test_plot_coverage(plot_config, mock_model, mock_data, mock_run_id):
    Config(plot_config)
    plot = CoverageFraction(mock_model, mock_data, mock_run_id, save=True, show=False)
    plot(**get_item("plots", "CoverageFraction", raise_exception=False))
    assert os.path.exists(f"{plot.out_dir}/{mock_run_id}_{plot.plot_name}")

def test_plot_tarp(plot_config, mock_model, mock_data, mock_run_id):
    Config(plot_config)
    plot = TARP(mock_model, mock_data, mock_run_id, save=True, show=False)
    plot(**get_item("plots", "TARP", raise_exception=False))
    assert os.path.exists(f"{plot.out_dir}/{mock_run_id}_{plot.plot_name}")

def test_lc2st(plot_config, mock_model, mock_data, mock_2d_data, result_output, mock_run_id):
    Config(plot_config)
    plot = LC2ST(mock_model, mock_data, mock_run_id, save=True, show=False)
    plot(**get_item("plots", "LC2ST", raise_exception=False))
    assert os.path.exists(f"{plot.out_dir}/{mock_run_id}_{plot.plot_name}")

    plot = LC2ST(
        mock_model, mock_2d_data, mock_run_id, save=True, show=False, 
        out_dir=f"{result_output.strip('/')}/mock_2d/")
    assert type(plot.data.simulator).__name__ == "Mock2DSimulator"
    plot(**get_item("plots", "LC2ST", raise_exception=False))
    assert os.path.exists(f"{plot.out_dir}/{plot.plot_name}")

def test_ppc(plot_config, mock_model, mock_data, mock_2d_data, result_output, mock_run_id):
    Config(plot_config)
    plot = PPC(mock_model, mock_data, mock_run_id, save=True, show=False)
    plot(**get_item("plots", "PPC", raise_exception=False))
    assert os.path.exists(f"{plot.out_dir}/{mock_run_id}_{plot.plot_name}")

    plot = PPC(
        mock_model,
        mock_2d_data, mock_run_id, save=True, show=False, 
        out_dir=f"{result_output.strip('/')}/mock_2d/")
    assert type(plot.data.simulator).__name__ == "Mock2DSimulator"
    plot(**get_item("plots", "PPC", raise_exception=False))


def test_prior_pc(plot_config, mock_model, mock_2d_data, mock_data, mock_run_id, result_output):
    Config(plot_config)
    plot = PriorPC(mock_model, mock_data, mock_run_id, save=True, show=False)
    plot(**get_item("plots", "PriorPC", raise_exception=False))
    assert os.path.exists(f"{plot.out_dir}/{mock_run_id}_{plot.plot_name}")
    plot = PriorPC(
        mock_model,
        mock_2d_data, mock_run_id, save=True, show=False, 
        out_dir=f"{result_output.strip('/')}/mock_2d/")
    assert type(plot.data.simulator).__name__ == "Mock2DSimulator"
    plot(**get_item("plots", "PriorPC", raise_exception=False))
    assert os.path.exists(f"{plot.out_dir}/{mock_run_id}_{plot.plot_name}")

def test_parity(plot_config, mock_model, mock_data, mock_run_id):
    Config(plot_config)
    plot = Parity(mock_model, mock_data, mock_run_id, save=True, show=False)

    plot(include_difference= False, 
        include_residual = False, 
        include_percentage = False)

    assert os.path.exists(f"{plot.out_dir}/{mock_run_id}_{plot.plot_name}")
    assert os.path.exists(f"{plot.out_dir}/{mock_run_id}_{plot.plot_name}")

    plot(include_difference= True, 
        include_residual = False, 
        include_percentage = True)

    assert os.path.exists(f"{plot.out_dir}/{mock_run_id}_{plot.plot_name}")
    assert os.path.exists(f"{plot.out_dir}/{mock_run_id}_{plot.plot_name}")

    plot(include_difference= True, 
        include_residual = True, 
        include_percentage = True)

    assert os.path.exists(f"{plot.out_dir}/{mock_run_id}_{plot.plot_name}")


@pytest.mark.parametrize("plot_type", [CDFRanks, Ranks, CoverageFraction, TARP, LC2ST, PPC, PriorPC, Parity])
def test_rerun_plot(plot_type, plot_config, mock_model, mock_data, mock_run_id):

    Config(plot_config)
    plot = plot_type(mock_model, mock_data, mock_run_id, save=True, show=False)
    plot()
    assert os.path.exists(f"{plot.out_dir}{mock_run_id}_{plot.plot_name}")
    assert os.path.exists(f"{plot.out_dir}{mock_run_id}_diagnostic_metrics.h5")

    figure, subplots = plot_type(None, None, None, save=False, show=False).plot(
        data_display=f"{plot.out_dir}{mock_run_id}_diagnostic_metrics.h5",
        )
    assert subplots is not None
    figure.set_tight_layout(True)
    figure.savefig(f"{plot.out_dir}rerun_plot.png")

    assert os.path.exists(f"{plot.out_dir}rerun_plot.png")

