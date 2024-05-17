from typing import Any
import numpy as np
from abc import abstractmethod, ABC


class Simulator(ABC):
    def __init__(self) -> None:
        """
        Create a simulator that can be used to get simulated outcomes
        for different theta parameters and number of samples
        """
        return None

    @abstractmethod
    def generate_context(self, n_samples: int) -> np.ndarray:
        """
        [ABSTRACT, MUST BE FILLED]
        Specify how the conditioning context is generated.
        Can come from data, or from a generic distribution.

        Args:
            n_samples (int): Number of samples of context to pull

        Returns:
            np.ndarray: Conditioning context used to produce simulated outcomes with a given theta.
        """
        raise NotImplementedError

    @abstractmethod
    def simulate(self, theta: np.ndarray, context_samples: np.ndarray) -> np.ndarray:
        """
        [ABSTRACT, MUST BE FILLED]
        Specify a simulation S such that y_{theta} = S(context_samples|theta)

        Args:
            theta (np.ndarray): Parameters of the simulation model
            context_samples (np.ndarray): Samples to use with the theta-primed simulation model

        Returns:
            np.ndarray: Simulated outcome.
        """
        raise NotImplementedError

    def __call__(self, theta: np.ndarray, n_samples: int) -> np.ndarray:
        context = self.generate_context(n_samples=n_samples)
        simulated_outcome = self.simulate(theta, context_samples=context)
        return simulated_outcome
