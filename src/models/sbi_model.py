import os
import pickle

from src.models.model import Model 
from src.data.data import Data

class SBIModel(Model): 
    def __init__(self, model_path:str): 
        super().__init__(model_path)

    def _load(self, path:str) -> None: 
        assert os.path.exists(path), f"Cannot find model file at location {path}"
        assert path.split(".")[-1] == 'pkl', "File extension must be 'pkl'"

        with open(path, "rb") as file:
            posterior = pickle.load(file)
        self.posterior = posterior

    def sample_posterior(self, n_samples:int, data:Data) -> "": # TODO typing
        return self.posterior.sample((n_samples,), x=data.y_true)

    def predict_posterior(self, data): 
        posterior_samples = self.sample_posterior(data.y_true)
        posterior_predictive_samples = data.simulator(posterior_samples)
        return posterior_predictive_samples