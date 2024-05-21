import os
import yaml
from argparse import ArgumentParser

from utils.config import Config
from utils.defaults import Defaults
from data import DataModules
from models import ModelModules
from metrics import Metrics
from plots import Plots


def parser():
    parser = ArgumentParser()
    parser.add_argument("--config", "-c", default=None)

    # Model
    parser.add_argument("--model_path", "-m", default=None)
    parser.add_argument(
        "--model_engine",
        "-e",
        default=Defaults["model"]["model_engine"],
        choices=ModelModules.keys(),
    )

    # Data
    parser.add_argument("--data_path", "-d", default=None)
    parser.add_argument(
        "--data_engine",
        "-g",
        default=Defaults["data"]["data_engine"],
        choices=DataModules.keys(),
    )
    parser.add_argument("--simulator", "-s", default=None)
    # Common
    parser.add_argument("--out_dir", default=Defaults["common"]["out_dir"])

    # List of metrics (cannot supply specific kwargs)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(Defaults["metrics"].keys()),
        choices=Metrics.keys(),
    )

    # List of plots
    parser.add_argument(
        "--plots",
        nargs="+",
        default=list(Defaults["plots"].keys()),
        choices=Plots.keys(),
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

    for metrics_name, metrics_args in metrics.items():
        try: 
            Metrics[metrics_name](model, data, **metrics_args)()
        except (NotImplementedError, RuntimeError) as error: 
            print(f"WARNING - skipping metric {metrics_name} due to error: {error}")

    for plot_name, plot_args in plots.items():
        try: 
            Plots[plot_name](model, data, save=True, show=False, out_dir=out_dir)(
                **plot_args
            )
        except (NotImplementedError, RuntimeError) as error: 
            print(f"WARNING - skipping plot {plot_name} due to error: {error}")
