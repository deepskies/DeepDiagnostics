from typing import Any, Callable, Optional
import h5py
import numpy as np
import torch
import os

from data.data import Data


class H5Data(Data):
    def __init__(self, 
        path: str, 
        simulator: Callable, 
        simulator_kwargs: dict = None,
        prior: str = None,
        prior_kwargs: dict = None,
        simulation_dimensions:Optional[int] = None,
    ):
        super().__init__(path, simulator, simulator_kwargs, prior, prior_kwargs, simulation_dimensions)

    def _load(self, path):
        assert path.split(".")[-1] == "h5", "File extension must be h5"
        loaded_data = {}
        with h5py.File(path, "r") as file:
            for key in file.keys():
                loaded_data[key] = torch.Tensor(file[key][...])
        return loaded_data

    def save(self, data: dict[str, Any], path: str):  # Todo typing for data dict
        assert path.split(".")[-1] == "h5", "File extension must be h5"
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        data_arrays = {key: np.asarray(value) for key, value in data.items()}
        with h5py.File(path, "w") as file:
            # Save each array as a dataset in the HDF5 file
            for key, value in data_arrays.items():
                file.create_dataset(key, data=value)

    def true_context(self):
        # From Data
        return self.data["xs"]  # TODO change name

    def prior(self):
        # From Data
        raise NotImplementedError

    def get_theta_true(self):
        return self.data["thetas"]

    def get_sigma_true(self):
        try:
            return super().get_sigma_true()
        except (AssertionError, KeyError):
            return 1
