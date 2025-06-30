from typing import Union
from deepdiagnostics.data.simulator import Simulator
import numpy as np


class LookupTableSimulator(Simulator):
    """
    A lookup table that mocks a simulator - assume your data is perfectly representative of a simulator (or else you are okay with nearest neighbor matching)

    Does not need to be registered, it is automatically available as the default simulator

    Assumes your has the following fields accessible as data["xs"], data["thetas"], data["ys"]
    where xs is the context, thetas are the parameters, and ys are the outcomes
    """

    def __init__(self, data: np.ndarray, random_state: np.random.Generator) -> None:
        super().__init__()
        # Normalizing for finding nearest neighbors
        self.max_theta = np.max(data["thetas"], axis=0)
        self.min_theta = np.min(data["thetas"], axis=0)
        self.max_x = np.max(data["xs"], axis=0)
        self.min_x = np.min(data["xs"], axis=0)

        self.table = self._build_table(data)
        self.rng = random_state

        for key in ["xs", "thetas", "ys"]:
            if key not in data.keys():
                msg = f"Data must have a field `{key}` - found {data.keys()}"
                raise ValueError(msg)
        
    def _build_table(self, data): 
        "Takes all the theta, context and outcome data and builds a lookup table"
        table = {
            self._build_hash(theta, context): {
                "y": outcome,
                "loc": self._calc_hash_distance(theta, context),
                "theta": theta,
                "x": context,
            }
            for theta, context, outcome in zip(data["thetas"], data["xs"], data["ys"])
        }
        return table

    def _build_hash(self, theta, context): 
        "Take a theta and context, and build a hashable key for the lookup table"
        return hash(tuple(np.concatenate([theta, context])))

    def _calc_hash_distance(self, theta: Union[np.ndarray, float], context: Union[np.ndarray, float]) -> float:
        "Create a distance (as the norm) metric between pairs of theta and context"
        theta = (theta - self.min_theta) / (self.max_theta - self.min_theta)
        context = (context - self.min_x) / (self.max_x - self.min_x)
        return np.linalg.norm(np.concatenate([theta, context]))

    def generate_context(self, n_samples):
        "Draw samples from the context data"
        keys = list(self.table.keys())
        chosen_keys = self.rng.choice(keys, size=n_samples, replace=True)
        contexts = [self.table[k]["x"] for k in chosen_keys]
        return np.array(contexts)
    
    def simulate(self, theta: Union[np.ndarray, float], context_samples: Union[np.ndarray, float]) -> np.ndarray:
        """
        Find the outcome y for a given theta and context sample. 
        If no exact match, take the nearest neighbor (via the L2 norm of normalized theta and context)

        Args: 
            theta (Union[np.ndarray, float]): parameter(s) to simulate
            context_samples (Union[np.ndarray, float]): context(s) to condition on
        Returns:
            np.ndarray: Simulated outcomes(s)
        """

        results = []
        if not isinstance(theta, np.ndarray):
            theta = np.array([theta])

        if not isinstance(context_samples, np.ndarray):
            context_samples = np.array([context_samples])

        for t, x in zip(theta, context_samples):
            key = self._build_hash(t, x)
            try: 
                results.append(self.table[key]["y"])
            except KeyError:
                print(f"Could not match theta {t} and x {x} to a result - taking the nearest neighbor")
                space_hit = self._calc_hash_distance(t, x)
                nearest_key = min(self.table.keys(), key=lambda k: abs(self.table[k]["loc"] - space_hit))
                results.append(self.table[nearest_key]["y"])

        return np.array(results)

