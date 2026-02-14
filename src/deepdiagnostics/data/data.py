from typing import Any, Optional, Union
import numpy as np

from deepdiagnostics.utils.config import get_item
from deepdiagnostics.data.lookup_table_simulator import LookupTableSimulator
from deepdiagnostics.utils.simulator_utils import load_simulator

class Data:
    """
        Load stored data to use in diagnostics

        Args:
            path (str): path to the data file.
            simulator_name (str): Name of the register simulator. If your simulator is not registered with utils.register_simulator, it will produce an error here. 
            simulator_kwargs (dict, optional): Any additional kwargs used set up your simulator. Defaults to None.
            prior (str, optional): If the prior is not given in the data, use a numpy random distribution. Specified by name. Choose from: {
                "normal"
                "poisson"
                "uniform"
                "gamma"
                "beta"
                "binominal}. Defaults to None.
            prior_kwargs (dict, optional): kwargs for the numpy prior. `View this page for a description <https://numpy.org/doc/stable/reference/random/generator.html#distributions>`_. Defaults to None.
            simulation_dimensions (Optional[int], optional): 1 or 2. 1->output of the simulator has one dimensions, 2->output has two dimensions (is an image). Defaults to None.
    """
    def __init__(
        self,
        path: str,
        simulator_name: str,
        simulator_kwargs: dict = None,
        prior: str = None,
        prior_kwargs: dict = None,
        simulation_dimensions:Optional[int] = None,
    ):
        self.rng = np.random.default_rng(
            get_item("common", "random_seed", raise_exception=False)
        )
        self.data = self._load(path)
        try: 
            self.simulator = load_simulator(simulator_name, simulator_kwargs)
        except RuntimeError: 
            print("Warning: Simulator not loaded. Using a lookup table simulator.")
            try: 
                self.simulator = LookupTableSimulator(self.data, self.rng)
            except ValueError as e:
                msg = f"Could not load the lookup table simulator - {e}. You cannot use online diagnostics."
                print(msg)

        self.context = self._context()
        self.thetas = self._thetas()

        self.prior_dist = self.load_prior(prior, prior_kwargs)
        self.n_dims = self.thetas.shape[1]
        self.simulator_dimensions = simulation_dimensions if simulation_dimensions is not None else get_item("data", "simulator_dimensions", raise_exception=False)

        self.simulator_outcome = self._simulator_outcome()

    def _load(self, path: str):
        raise NotImplementedError

    def _simulator_outcome(self):
        raise NotImplementedError
    
    def _context(self):
        raise NotImplementedError
    
    def _thetas(self):
        raise NotImplementedError

    def save(self, data, path: str):
        raise NotImplementedError

    def read_prior(self):
        raise NotImplementedError

    def get_sigma_true(self) -> Union[Any, float, int, np.ndarray]:
        """
        Look for the true sigma of data. If supplied in the method, use that, other look in the configuration file. 
        If neither are supplied, return 1. 

        Returns:
            Any: Sigma value selected by the search. 
        """
        if hasattr(self, "sigma_true"):
            return self.sigma_true()
        else:
            return get_item("data", "sigma_true", raise_exception=True)

    def load_prior(self, prior:str, prior_kwargs:dict[str, any]) -> callable:
        """
        Load the prior. 
        Either try to get it from data (if it has been implemented for the type of data), 
        or use numpy to initialize a random distribution using the prior argument. 

        Args:
            prior (str): Name of prior. 
            prior_kwargs (dict[str, any]): kwargs for initializing the prior. 

        Raises:
            NotImplementedError: The selected prior is not included.
            RuntimeError: The selected prior is missing arguments to initialize. 

        Returns:
            callable: Prior that can be sampled from by calling it with prior(n_samples)
        """

        if prior is None:
            prior = get_item("data", "prior", raise_exception=False)
        try:
            prior = self.read_prior()
        except NotImplementedError:
            choices = {
                "normal": self.rng.normal,
                "poisson": self.rng.poisson,
                "uniform": self.rng.uniform,
                "gamma": self.rng.gamma,
                "beta": self.rng.beta,
                "binominal": self.rng.binomial,
            }

            if prior not in choices.keys():
                raise NotImplementedError(
                    f"{prior} is not an option for a prior, choose from {list(choices.keys())}"
                )
            if prior_kwargs is None:
                prior_kwargs = {}
            return lambda size: choices[prior](**prior_kwargs, size=size)

        except KeyError as e:
            raise RuntimeError(f"Data missing a prior specification - {e}")

    def sample_prior(self, n_samples: int) -> np.ndarray:
        """
        Sample from the prior. 

        Args:
            n_samples (int): Number of samples to draw. 

        Returns:
            np.ndarray: Samples drawn from the prior. 
        """

        if self.prior_dist is None:
            prior = self.read_prior()
            sample = self.rng.randint(0, len(prior), size=n_samples)
            return prior[sample]
        else: 
            return self.prior_dist(size=(n_samples, self.n_dims))