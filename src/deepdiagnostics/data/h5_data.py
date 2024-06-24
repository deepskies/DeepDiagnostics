from typing import Any
import h5py
import numpy as np
import torch
import os

from deepdiagnostics.data.data import Data


class H5Data(Data):
    """
    Load data that has been saved in a h5 format. 

    .. attribute:: Data Parameters  

        :xs: [REQUIRED] The context, the x values. The data that was used to train a model on what conditions produce what posterior. 
        :thetas: [REQUIRED] The theta, the parameters of the external model. The data used to train the model's posterior. 
        :prior: Distribution used to initialize the posterior before training. 
        :sigma: True standard deviation of the actual thetas, if known. 
    
    """

    def __init__(self, 
        path, 
        simulator, 
        simulator_kwargs = None,
        prior=None,
        prior_kwargs = None,
        simulation_dimensions = None,
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
        """
        Try to get the `xs` field of the loaded data.

        Raises:
            NotImplementedError: The data does not have a `xs` field. 
        """
        try: 
            return self.data["xs"]
        except KeyError: 
            raise NotImplementedError("Cannot find `xs` in data. Please supply it.")

    def prior(self):
        """
        If the data has a supplied prior, return it. If not, the data module will default back to picking a prior from a random distribution. 

        Raises:
            NotImplementedError: The data does not have a `prior` field. 
        """
        try: 
            return self.data['prior']
        except KeyError: 
            raise NotImplementedError("Data does not have a `prior` field.")

    def get_theta_true(self):
        """ Get stored theta used to train the model. 

        Returns:
            theta array 

        Raise: 
            NotImplementedError: Data does not have thetas. 
        """
        try: 
            return self.data["thetas"]
        except KeyError: 
            raise NotImplementedError("Data does not have a `thetas` field.")

    def get_sigma_true(self):
        """
        Try to get the true standard deviation of the data. If it is not supplied, return 1. 

        Returns:
            Any: sigma. 
        """
        try:
            return super().get_sigma_true()
        except (AssertionError, KeyError):
            return 1
