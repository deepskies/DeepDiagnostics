import os
import uuid
import yaml
from argparse import ArgumentParser

from deepdiagnostics.utils.config import Config
from deepdiagnostics.utils.defaults import Defaults
from deepdiagnostics.data import DataModules
from deepdiagnostics.models import ModelModules
from deepdiagnostics.metrics import Metrics
from deepdiagnostics.plots import Plots
from deepdiagnostics.utils.simulator_utils import SimulatorMissingError


def parser():
    parser = ArgumentParser()
    parser.add_argument("--config", "-c", default=None, help=".yaml file with all arguments to run.")

    # Model
    parser.add_argument("--model_path", "-m", default=None, help="String path to a model. Must be compatible with your model_engine choice.")
    parser.add_argument(
        "--model_engine",
        "-e",
        default=Defaults["model"]["model_engine"],
        choices=ModelModules.keys(),
        help="Way to load your model. See each module's documentation page for requirements and specifications."
    )

    # Data
    parser.add_argument("--data_path", "-d", default=None, help="String path to data. Must be compatible with data_engine choice.")
    parser.add_argument(
        "--data_engine",
        "-g",
        default=Defaults["data"]["data_engine"],
        choices=DataModules.keys(),
        help="Way to load your data. See each module's documentation page for requirements and specifications."
    )
    parser.add_argument(
        "--simulator", "-s", 
        default=None, 
        help='String name of the simulator to use with generative metrics and plots. Must be pre-register with the `utils.register_simulator` method.')
    # Common
    parser.add_argument(
        "--out_dir", 
        default=Defaults["common"]["out_dir"], 
        help="Where the results will be saved. Path need not exist, it will be created."
    )

    # List of metrics (cannot supply specific kwargs)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[],
        choices=Metrics.keys(),
        help="List of metrics to run."
    )

    # List of plots
    parser.add_argument(
        "--plots",
        nargs="+",
        default=[],
        choices=Plots.keys(),
        help="List of plots to run."

    )

    args = parser.parse_args()
    if args.config is not None:
        config = Config(args.config)

    else:
        temp_config = Defaults["common"]["temp_config"]
        os.makedirs(os.path.dirname(temp_config), exist_ok=True)

        input_yaml = {
            "common": {"out_dir": args.out_dir},
            "model": {"model_path": args.model_path, "model_engine": args.model_engine},
            "data": {
                "data_path": args.data_path,
                "data_engine": args.data_engine,
                "simulator": args.simulator,
            },
            "plots": {key: {} for key in args.plots},
            "metrics": {key: {} for key in args.metrics},
        }

        yaml.dump(input_yaml, open(temp_config, "w"))
        config = Config(temp_config)

    return config


def main():
    config = parser()

    run_id = str(uuid.uuid4()).replace('-', '')[:10]

    model_path = config.get_item("model", "model_path")
    model_engine = config.get_item("model", "model_engine", raise_exception=False)
    model = ModelModules[model_engine](model_path)

    data_path = config.get_item("data", "data_path")
    data_engine = config.get_item("data", "data_engine", raise_exception=False)
    simulator_name = config.get_item("data", "simulator")
    data = DataModules[data_engine](data_path, simulator_name)

    out_dir = config.get_item("common", "out_dir", raise_exception=False)
    if not os.path.exists(os.path.dirname(out_dir)):
        os.makedirs(os.path.dirname(out_dir))

    metrics = config.get_section("metrics", raise_exception=False)
    plots = config.get_section("plots", raise_exception=False)

    if not config.get_item("common", "random_seed", raise_exception=False): 
        config["common"]["random_seed"] = int(run_id) % 10000

    for metrics_name, metrics_args in metrics.items():
        try: 
            Metrics[metrics_name](model, data, save=True)(**metrics_args)
        except SimulatorMissingError:
            print(f"Cannot run {metrics_name} - simulator missing.")

    for plot_name, plot_args in plots.items():
        try: 
            Plots[plot_name](model, data, run_id, save=True, show=False, out_dir=out_dir)(
                **plot_args
            )
        except SimulatorMissingError: 
            print(f"Cannot run {plot_name} - simulator missing.")