import importlib.util
import sys
import os
import numpy as np 

from utils.config import get_item
from utils.defaults import Defaults

class Data: 
    def __init__(self, path:str, simulator_name: str, prior: str = "normal", prior_kwargs:dict=None):
        self.data = self._load(path)
        self.simulator = self._load_simulator(simulator_name)
        self.prior_dist = self.load_prior(prior, prior_kwargs)
        
        self.n_dims = self.theta_true().shape[1]

    def _load_simulator(self, name): 
        if name is not None: 
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
    
    def prior(self, n_samples:int): 
        return self.prior_dist(size=(n_samples, self.n_dims))
    
    def theta_true(self): 
        return get_item("data", "theta_true")
    
    def sigma_true(self): 
        return get_item("data", "sigma_true")
    
    def save(self, data, path:str): 
        raise NotImplementedError
    
    def load_prior(self, prior, prior_kwargs): 
        rng = np.random.default_rng(seed=42)
        choices = {
            "normal": rng.normal
        }

        if prior not in choices.keys(): 
            raise NotImplementedError(f"{prior} is not an option for a prior, choose from {list(choices.keys())}")
        if prior_kwargs is None: 
            prior_kwargs = {}
        return lambda size: choices[prior](**prior_kwargs, size=size)
