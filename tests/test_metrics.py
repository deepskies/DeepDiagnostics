import os 
import pytest 

from utils.defaults import Defaults 
from utils.config import Config
from metrics import (
    Metrics, 
    CoverageFraction, 
    AllSBC, 
    LocalTwoSampleTest
)

@pytest.fixture
def metric_config(config_factory): 
    metrics_settings={"use_progress_bar":False, "samples_per_inference":10, "percentiles":[95]}
    config = config_factory(metrics_settings=metrics_settings)
    Config(config)

def test_all_defaults(metric_config, mock_model, mock_data): 
    """
    Ensures each metric has a default set of parameters and is included in the defaults list
    Ensures each test can initialize, regardless of the veracity of the output 
    """

    for metric_name, metric_obj in Metrics.items(): 
        assert metric_name in Defaults['metrics']
        metric_obj(mock_model, mock_data)

def test_coverage_fraction(metric_config, mock_model, mock_data): 
    coverage_fraction = CoverageFraction(mock_model, mock_data, save=True)
    _, coverage = coverage_fraction.calculate()
    assert coverage_fraction.output.all() is not None

    # TODO Shape of coverage 
    assert coverage.shape 

    coverage_fraction = CoverageFraction(mock_model, mock_data, save=True)
    coverage_fraction()
    assert os.path.exists(f"{coverage_fraction.out_dir}/diagnostic_metrics.json")
    
def test_all_sbc(metric_config, mock_model, mock_data): 
    all_sbc = AllSBC(mock_model, mock_data, save=True)
    all_sbc()
    assert all_sbc.output is not None
    assert os.path.exists(f"{all_sbc.out_dir}/diagnostic_metrics.json")
    
def test_lc2st(metric_config, mock_model, mock_data): 
    lc2st = LocalTwoSampleTest(mock_model, mock_data, save=True)
    lc2st()
    assert lc2st.output is not None
    assert os.path.exists(f"{lc2st.out_dir}/diagnostic_metrics.json")