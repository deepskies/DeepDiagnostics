import pickle
from typing import Any, Callable, Optional
from deepdiagnostics.data.data import Data


class PickleData(Data):
    def __init__(self, 
        path: str, 
        simulator: Callable, 
        simulator_kwargs: dict = None,
        prior: str = None,
        prior_kwargs: dict = None,
        simulation_dimensions:Optional[int] = None,
    ):
        super().__init__(path, simulator, simulator_kwargs, prior, prior_kwargs, simulation_dimensions)

    def _load(self, path: str):
        assert path.split(".")[-1] == "pkl", "File extension must be 'pkl'"
        with open(path, "rb") as file:
            data = pickle.load(file)
        return data

    def save(self, data: Any, path: str):
        assert path.split(".")[-1] == "pkl", "File extension must be 'pkl'"
        with open(path, "wb") as file:
            pickle.dump(data, file)
