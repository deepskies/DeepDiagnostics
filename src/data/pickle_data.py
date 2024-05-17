import pickle
from typing import Any, Callable
from data.data import Data


class PickleData(Data):
    def __init__(self, path: str, simulator: Callable):
        super().__init__(path, simulator)

    def _load(self, path: str):
        assert path.split(".")[-1] == "pkl", "File extension must be 'pkl'"
        with open(path, "rb") as file:
            data = pickle.load(file)
        return data

    def save(self, data: Any, path: str):
        assert path.split(".")[-1] == "pkl", "File extension must be 'pkl'"
        with open(path, "wb") as file:
            pickle.dump(data, file)
