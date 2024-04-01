import os
import inspect

from utils.defaults import Defaults

def register_simulator(simulator_name, simulator): 
    simulator_prefix = Defaults['common']['sim_location']
    env_var_name = f"{simulator_prefix}:{simulator_name}"
    simulator_location = os.path.abspath(inspect.getfile(simulator))
    os.environ[env_var_name] = simulator_location