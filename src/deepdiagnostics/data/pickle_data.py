import pickle
from typing import Any
from deepdiagnostics.data.data import Data


class PickleData(Data):
    """
    Load data that is saved as a .pkl file. 
    """
    def __init__(self, 
        path, 
        simulator, 
        simulator_kwargs = None,
        prior = None,
        prior_kwargs = None,
        simulation_dimensions = None,
    ):
        super().__init__(path, simulator, simulator_kwargs, prior, prior_kwargs, simulation_dimensions)

    def _load(self, path: str) -> Any:
        assert path.split(".")[-1] == "pkl", "File extension must be 'pkl'"
        with open(path, "rb") as file:
            data = pickle.load(file)
        return data

    def save(self, data: Any, path: str) -> None:
        """ 
        Save data in the form of a .pkl file. 

        Args:
            data (Any): Data that can be encoded into a pkl. 
            path (str): Out file path for the data. Must have a .pkl extension. 
        """
        assert path.split(".")[-1] == "pkl", "File extension must be 'pkl'"
        with open(path, "wb") as file:
            pickle.dump(data, file)
