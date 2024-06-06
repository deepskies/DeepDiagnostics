import importlib.util
import sys
import os
import numpy as np

from utils.config import get_item
from utils.register import load_simulator

class Data:
    def __init__(
        self,
        path: str,
        simulator_name: str,
        simulator_kwargs: dict = None,
        prior: str = None,
        prior_kwargs: dict = None,
    ):
        self.rng = np.random.default_rng(
            get_item("common", "random_seed", raise_exception=False)
        )
        self.data = self._load(path)
        self.simulator = load_simulator(simulator_name, simulator_kwargs)
        self.prior_dist = self.load_prior(prior, prior_kwargs)
        self.n_dims = self.get_theta_true().shape[1]

    def _load(self, path: str):
        raise NotImplementedError

    def true_context(self):
        # From Data
        raise NotImplementedError

    def true_simulator_outcome(self):
        return self.simulator(self.get_theta_true(), self.true_context())

    def sample_prior(self, n_samples: int):
        return self.prior_dist(size=(n_samples, self.n_dims))

    def simulator_outcome(self, theta, condition_context=None, n_samples=None):
        if condition_context is None:
            if n_samples is None:
                raise ValueError(
                    "Samples required if condition context is not specified"
                )
            return self.simulator(theta, n_samples)
        else:
            return self.simulator.simulate(theta, condition_context)

    def simulated_context(self, n_samples):
        return self.simulator.generate_context(n_samples)

    def get_theta_true(self):
        if hasattr(self, "theta_true"):
            return self.theta_true
        else:
            return get_item("data", "theta_true", raise_exception=True)

    def get_sigma_true(self):
        if hasattr(self, "sigma_true"):
            return self.sigma_true
        else:
            return get_item("data", "sigma_true", raise_exception=True)

    def save(self, data, path: str):
        raise NotImplementedError

    def read_prior(self):
        raise NotImplementedError

    def load_prior(self, prior, prior_kwargs):
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
