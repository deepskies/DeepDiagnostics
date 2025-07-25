import os
import pytest

from deepdiagnostics.utils.config import Config
from deepdiagnostics.metrics import (
    CoverageFraction, 
    AllSBC, 
    LC2ST
)

@pytest.fixture
def metric_config(config_factory):
    metrics_settings = {
        "use_progress_bar": False,
        "samples_per_inference": 10,
        "percentiles": [95],
    }
    return config_factory(metrics_settings=metrics_settings)

def test_coverage_fraction(metric_config, mock_model, mock_data, mock_run_id): 
    Config(metric_config)
    coverage_fraction = CoverageFraction(mock_model, mock_data, mock_run_id, save=True)
    _, (coverage_mean, coverage_std) = coverage_fraction.calculate()
    assert coverage_mean.all() is not None
    assert coverage_std.all() is not None

    assert coverage_mean.shape  ==  (1, 2) # One percentile over 2 dimensions of theta. 

    coverage_fraction = CoverageFraction(mock_model, mock_data, mock_run_id, save=True)
    coverage_fraction()
    assert os.path.exists(f"{coverage_fraction.out_dir}/{mock_run_id}_diagnostic_metrics.json")
    
def test_all_sbc(metric_config, mock_model, mock_data, mock_run_id):
    Config(metric_config)
    all_sbc = AllSBC(mock_model, mock_data, mock_run_id, save=True)
    all_sbc()
    assert all_sbc.output is not None
    assert os.path.exists(f"{all_sbc.out_dir}/{mock_run_id}_diagnostic_metrics.json")
    
def test_lc2st(metric_config, mock_model, mock_data, mock_run_id):
    Config(metric_config)
    lc2st = LC2ST(mock_model, mock_data, mock_run_id, save=True)
    lc2st()
    assert lc2st.output is not None
    assert os.path.exists(f"{lc2st.out_dir}/{mock_run_id}_diagnostic_metrics.json")

