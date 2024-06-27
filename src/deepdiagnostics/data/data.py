from typing import Any, Optional, Sequence, Union
import numpy as np

from deepdiagnostics.utils.config import get_item
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
            print("Warning: Simulator not loaded. Can only run non-generative metrics.")
            
        self.prior_dist = self.load_prior(prior, prior_kwargs)
        self.n_dims = self.get_theta_true().shape[1]
        self.simulator_dimensions = simulation_dimensions if simulation_dimensions is not None else get_item("data", "simulator_dimensions", raise_exception=False)

    def get_simulator_output_shape(self) -> tuple[Sequence[int]]: 
        """
        Run a single sample of the simulator to verify the out-shape. 

        Returns:
             tuple[Sequence[int]]: Output shape of a single sample of the simulator. 
        """
        context_shape = self.true_context().shape
        sim_out = self.simulator(theta=self.get_theta_true()[0:1, :], n_samples=context_shape[-1])
        return sim_out.shape

    def _load(self, path: str):
        raise NotImplementedError

    def true_context(self):
        """
        True data x values, if supplied by the data method. 
        """
        # From Data
        raise NotImplementedError

    def true_simulator_outcome(self) -> np.ndarray:
        """
        Run the simulator on all true theta and true x values. 

        Returns:
            np.ndarray: array of (n samples, simulator shape) showing output of the simulator on all true samples in data.
        """
        return self.simulator(self.get_theta_true(), self.true_context())

    def sample_prior(self, n_samples: int) -> np.ndarray:
        """
        Draw samples from the simulator

        Args:
            n_samples (int): Number of samples to draw

        Returns:
            np.ndarray: 
        """
        return self.prior_dist(size=(n_samples, self.n_dims))

    def simulator_outcome(self, theta:np.ndarray, condition_context:np.ndarray=None, n_samples:int=None):
        """_summary_

        Args:
            theta (np.ndarray): Theta value of shape (n_samples, theta_dimensions)
            condition_context (np.ndarray, optional): If x values for theta are known, use them. Defaults to None.
            n_samples (int, optional): If x values are not known for theta, draw them randomly. Defaults to None.

        Raises:
            ValueError: If either n samples or content samples is supplied. 

        Returns:
            np.ndarray: Simulator output of shape (n samples, simulator_dimensions)
        """
        if condition_context is None:
            if n_samples is None:
                raise ValueError(
                    "Samples required if condition context is not specified"
                )
            return self.simulator(theta, n_samples)
        else:
            return self.simulator.simulate(theta, condition_context)

    def simulated_context(self, n_samples:int) -> np.ndarray:
        """
        Call the simulator's `generate_context` method. 

        Args:
            n_samples (int): Number of samples to draw. 

        Returns:
            np.ndarray: context (x values), as defined by the simulator. 
        """
        return self.simulator.generate_context(n_samples)

    def get_theta_true(self) -> Union[Any, float, int, np.ndarray]:
        """
        Look for the true theta given by data. If supplied in the method, use that, other look in the configuration file. 
        If neither are supplied, return None.

        Returns:
            Any: Theta value selected by the search. 
        """
        if hasattr(self, "theta_true"):
            return self.theta_true
        else:
            return get_item("data", "theta_true", raise_exception=True)

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

    def save(self, data, path: str):
        raise NotImplementedError

    def read_prior(self):
        raise NotImplementedError

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
