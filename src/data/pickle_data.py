import pickle
from typing import Any 

class PickleData: 
    def __init__(self, path:str) -> None:
       return super().__init__(path)

    def _load(self, path:str): 
        assert path.split('.')[-1] == 'pkl', "File extension must be 'pkl'"
        with open(path, "rb") as file:
            data = pickle.load(file)
        return data 

    def save(self, data:Any, path:str): 
        assert path.split('.')[-1] == 'pkl', "File extension must be 'pkl'"
        with open(path, "wb") as file:
            pickle.dump(data, file)