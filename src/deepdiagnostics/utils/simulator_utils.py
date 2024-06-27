import os
import inspect
import importlib.util
import sys
import json 

from deepdiagnostics.utils.config import get_item

def register_simulator(simulator_name, simulator):

    simulator_config_path = get_item("common", "sim_location", raise_exception=False)
    sim_paths = f"{simulator_config_path.strip('/')}/simulators.json"
    simulator_location = os.path.abspath(inspect.getfile(simulator))

    if not os.path.exists(os.path.dirname(sim_paths)): 
        os.makedirs(os.path.dirname(sim_paths))

    if not os.path.exists(sim_paths):
        open(sim_paths, 'a').close()

    with open(sim_paths, "r+") as f: 
        try: 
            existing_sims = json.load(f)
        except json.decoder.JSONDecodeError: 
            existing_sims = {}
            
    existing_sims[simulator_name] = simulator_location
    with open(sim_paths, "w") as f: 
        existing_sims[simulator_name] = simulator_location
        json.dump(existing_sims, f)


def load_simulator(name, simulator_kwargs):
    simulator_config_path = get_item("common", "sim_location", raise_exception=False)
    sim_paths = f"{simulator_config_path.strip('/')}/simulators.json"
    if not os.path.exists(sim_paths): 
        raise RuntimeError(
                f"Simulator catalogue cannot be found at path {sim_paths}. Hint: have you registered your simulation with utils.register_simulator?"
            )

    with open(sim_paths, "r") as f: 
        paths = json.load(f)
        try: 
            simulator_path = paths[name]

        except KeyError as e:
            raise RuntimeError(
                f"Simulator cannot be found using name {e}. Hint: have you registered your simulation with utils.register_simulator?"
            )

    new_class = os.path.dirname(simulator_path)
    sys.path.insert(1, new_class)

    # TODO robust error checks
    module_name = os.path.basename(simulator_path.rstrip(".py"))
    m = importlib.import_module(module_name)

    simulator = getattr(m, name)

    simulator_kwargs = simulator_kwargs if simulator_kwargs is not None else get_item("data", "simulator_kwargs", raise_exception=False)
    simulator_kwargs = {} if simulator_kwargs is None else simulator_kwargs
    simulator_instance = simulator(**simulator_kwargs)

    if not hasattr(simulator_instance, "generate_context"):
        raise RuntimeError(
            "Simulator improperly formed - requires a generate_context method."
        )

    if not hasattr(simulator_instance, "simulate"):
        raise RuntimeError(
            "Simulator improperly formed - requires a simulate method."
        )

    return simulator_instance


class SimulatorMissingError(Exception): 
    pass