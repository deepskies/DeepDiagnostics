import os
import pytest

from utils.defaults import Defaults
from utils.config import Config
from metrics import Metrics, CoverageFraction, AllSBC


@pytest.fixture
def metric_config(config_factory):
    metrics_settings = {
        "use_progress_bar": False,
        "samples_per_inference": 10,
        "percentiles": [95],
    }
    config = config_factory(metrics_settings=metrics_settings)
    Config(config)
    return config


def test_all_metrics_catalogued():
    """Each metrics gets its own file, and each metric is included in the Metrics dictionary
    so the client can use it.
    This test verifies all metrics are cataloged"""

    all_files = os.listdir("src/metrics/")
    files_ignore = [
        "metric.py",
        "__init__.py",
        "__pycache__",
    ]  # All files not containing a metric
    num_files = len([file for file in all_files if file not in files_ignore])
    assert len(Metrics) == num_files


def test_all_defaults(metric_config, mock_model, mock_data):
    """
    Ensures each metric has a default set of parameters and is included in the defaults list
    Ensures each test can initialize, regardless of the veracity of the output
    """
    Config(metric_config)

    for metric_name, metric_obj in Metrics.items():
        assert metric_name in Defaults["metrics"]
        metric_obj(mock_model, mock_data)


def test_coverage_fraction(metric_config, mock_model, mock_data):
    Config(metric_config)
    coverage_fraction = CoverageFraction(mock_model, mock_data)
    _, coverage = coverage_fraction.calculate()
    assert coverage_fraction.output.all() is not None

    # TODO Shape of coverage
    assert coverage.shape


def test_all_sbc(metric_config, mock_model, mock_data):
    Config(metric_config)
    all_sbc = AllSBC(mock_model, mock_data)
    all_sbc()
    # TODO What is this supposed to be
