import importlib.util
import sys
import os
import numpy as np

from utils.config import get_item


class Data:
    def __init__(
        self,
        path: str,
        simulator_name: str,
        simulator_kwargs: dict = None,
        prior: str = "data",
        prior_kwargs: dict = None,
    ):
        self.rng = np.random.default_rng(
            get_item("common", "random_seed", raise_exception=False)
        )
        self.data = self._load(path)
        self.simulator = self._load_simulator(simulator_name, simulator_kwargs)
        self.prior_dist = self.load_prior(prior, prior_kwargs)
        self.n_dims = self.theta_true().shape[1]

    def _load_simulator(self, name):
        try:
            sim_location = get_item("common", "sim_location", raise_exception=False)
            simulator_path = os.environ[f"{sim_location}:{name}"]
        except KeyError as e:
            raise RuntimeError(
                f"Simulator cannot be found using env var {e}. Hint: have you registered your simulation with utils.register_simulator?"
            )

        new_class = os.path.dirname(simulator_path)
        sys.path.insert(1, new_class)

        # TODO robust error checks
        module_name = os.path.basename(simulator_path.rstrip(".py"))
        m = importlib.import_module(module_name)

        simulator = getattr(m, name)

        simulator_kwargs = get_item("data", "simulator_kwargs", raise_exception=False)
        simulator_kwargs = {} if simulator_kwargs is None else simulator_kwargs
        simulator_instance = simulator(**simulator_kwargs)

        if not hasattr(simulator_instance, "generate_context"):
            raise RuntimeError(
                "Simulator improperly formed - requires a generate_context method."
            )

        if not hasattr(simulator_instance, "simulate"):
            raise RuntimeError(
                "Simulator improperly formed - requires a simulate method."
            )

        return simulator_instance

    def _load(self, path: str):
        raise NotImplementedError

    def true_context(self):
        # From Data
        raise NotImplementedError

    def true_simulator_outcome(self):
        return self.simulator(self.theta_true(), self.true_context())

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

    def theta_true(self):
        if hasattr(self, "theta_true"):
            return self.theta_true
        else:
            return get_item("data", "theta_true")

    def sigma_true(self):
        if hasattr(self, "sigma_true"):
            return self.sigma_true
        else:
            return get_item("data", "sigma_true")

    def save(self, data, path: str):
        raise NotImplementedError

    def read_prior(self):
        raise NotImplementedError

    def load_prior(self, prior, prior_kwargs):
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
