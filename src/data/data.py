from typing import Callable
from src.utils.config import get_item

class Data: 
    def __init__(self, simulator: Callable, path:str):
        self.data = self._load(path)
        self.simulator = simulator

    def _load(self, path:str): 
        raise NotImplementedError

    def x_true(self): 
        # From Data 
        raise NotImplementedError
    
    def y_true(self): 
        return self.simulator(self.theta_true(), self.x_true())
    
    def proir(self): 
        # From Data
        raise NotImplementedError
    
    def theta_true(self): 
        return get_item("??", "sigma_true")
    
    def sigma_true(self): 
        return get_item("??", "sigma_true")
    
    def save(self, data, path:str): 
        raise NotImplementedError