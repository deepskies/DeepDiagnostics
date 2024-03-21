from typing import Any
from src.data.data import Data 
from src.models.model import Model

class Metric: 
    def __init__(self, model:Model, data:Data) -> None:
        self.model = model 
        self.data = data 

    def _collect_data_params():
        raise NotImplementedError 
    
    def _run_model_inference(): 
        raise NotImplementedError
    
    def calculate(self): 
        raise NotImplementedError
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self._collect_data_params()
        self._run_model_inference()
        return self.calculate()

    