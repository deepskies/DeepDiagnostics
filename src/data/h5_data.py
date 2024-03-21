from typing import Any
import h5py
import numpy as np
import torch
import os 

from src.data.data import Data

class H5Data(Data): 
    def __init__(self, path:str):
       return super().__init__(path)

    def _load(self, path): 
        assert path.split(".")[-1] == "h5", "File extension must be h5"
        loaded_data = {}
        with h5py.File(path, "r") as file:
            for key in file.keys():
                loaded_data[key] = torch.Tensor(file[key][...])
        return loaded_data
 
    def save(self, data:dict[str, Any], path: str): # Todo typing for data dict
        assert path.split(".")[-1] == "h5", "File extension must be h5"
        if not os.path.exists(os.path.dirname(path)): 
            os.makedirs(os.path.dirname(path))

        data_arrays = {key: np.asarray(value) for key, value in data.items()}
        with h5py.File(path, "w") as file:
            # Save each array as a dataset in the HDF5 file
            for key, value in data_arrays.items():
                file.create_dataset(key, data=value)