import importlib.util
import sys
import os

from utils.config import get_item
from utils.defaults import Defaults

class Data: 
    def __init__(self, path:str, simulator_name: str):
        self.data = self._load(path)
        self.simulator = self._load_simulator(simulator_name)

    def _load_simulator(self, name): 
        try: 
            simulator_path = os.environ[f"{Defaults['common']['sim_location']}:{name}"]
        except KeyError as e: 
            raise RuntimeError(f"Simulator cannot be found using env var {e}. Hint: have you registered your simulation with utils.register_simulator?")

        new_class = os.path.dirname(simulator_path)
        sys.path.insert(1, new_class)

        # TODO robust error checks 
        module_name = os.path.basename(simulator_path.rstrip('.py')) 
        m = importlib.import_module(module_name)
    
        simulator = getattr(m, name)
        return simulator()

    def _load(self, path:str): 
        raise NotImplementedError

    def x_true(self): 
        # From Data 
        raise NotImplementedError
    
    def y_true(self): 
        return self.simulator(self.theta_true(), self.x_true())
    
    def prior(self): 
        # From Data
        raise NotImplementedError
    
    def theta_true(self): 
        return get_item("data", "theta_true")
    
    def sigma_true(self): 
        return get_item("data", "sigma_true")
    
    def save(self, data, path:str): 
        raise NotImplementedError