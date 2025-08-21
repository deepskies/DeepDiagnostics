from typing import Union

import torch
from deepdiagnostics.data.simulator import Simulator
import numpy as np


class LookupTableSimulator(Simulator):
    """
    A lookup table that mocks a simulator - assume your data is perfectly representative of a simulator (or else you are okay with nearest neighbor matching)

    Does not need to be registered, it is automatically available as the default simulator

    Assumes your has the following fields accessible as data["context"], data["thetas"], data["simulator_outcome"],
    where xs is the context, thetas are the parameters, and ys are the outcomes
    """

    def __init__(self, data: torch.tensor, random_state: np.random.Generator, outside_range_limit: float = 2.0, hash_precision: int = 10) -> None:
        """

        Parameters
        ----------
        data : torch.tensor
            _description_
        random_state : np.random.Generator
            Random state used for sampling
        outside_range_limit : float, optional
            When values of theta and x are passed that are not in the dataset, 
            if they are greater than the max by this threshold, (where value > outside_range_limit*value_max), 
            a value error is raised instead of taking the nearest neighbor; by default 1.5
        hash_precision : int, optional
            Number of decimal places to round to when creating hash keys, by default 10
            
        Raises
        ------
        ValueError
            If the passed data does not have thetas, xs, and ys fields
        """
        super().__init__()
        # Normalizing for finding nearest neighbors
        for key in ["simulator_outcome", "thetas", "context"]:
            if key not in data.keys():
                msg = f"Data must have a field `{key}` - found {data.keys()}"
                raise ValueError(msg)

        self.precision = hash_precision

        self.max_theta = torch.max(data["thetas"], axis=0).values
        self.min_theta = torch.min(data["thetas"], axis=0).values
        self.max_x = torch.max(data["context"], axis=0).values
        self.min_x = torch.min(data["context"], axis=0).values

        self.threshold = outside_range_limit * torch.max(self.max_theta)

        self.table = self._build_table(data)
        self.rng = random_state
        
    def _build_table(self, data): 
        "Takes all the theta, context and outcome data and builds a lookup table"
        table = {
            self._build_hash(theta, simulator_outcome): {
                "y": simulator_outcome,
                "loc": self._calc_hash_distance(theta, context),
                "theta": theta,
                "x": context,
            }
            for theta, simulator_outcome, context in zip(data["thetas"], data["simulator_outcome"], data['context'])
        }
        return table

    def _build_hash(self, theta, context): 
        "Take a theta and context, and build a hashable key for the lookup table"
        hashval = torch.concat([theta, context], dim=-1).flatten()
        rounded_values = [round(val.item(), self.precision) for val in hashval]
        return hash(tuple(rounded_values))

    def _calc_hash_distance(self, theta: Union[torch.Tensor, float], context: Union[torch.Tensor, float]) -> float:
        "Create a distance (as the norm) metric between pairs of theta and context"
        theta = (theta - self.min_theta) / (self.max_theta - self.min_theta)
        context = (context - self.min_x) / (self.max_x - self.min_x)
        theta = theta.unsqueeze(0) if theta.dim() == 0 else theta
        context = context.unsqueeze(0) if context.dim() == 0 else context
        return torch.linalg.norm(torch.concat([theta, context], dim=-1))

    def generate_context(self, n_samples):
        "Draw samples from the context data"
        keys = list(self.table.keys())
        chosen_keys = self.rng.choice(keys, size=n_samples, replace=True)
        contexts = [self.table[k]["x"] for k in chosen_keys]
        return np.array(contexts)
    
    def simulate(self, theta: Union[torch.Tensor, float], context_samples: Union[torch.Tensor, float]) -> np.ndarray:
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
        if not isinstance(theta, torch.Tensor):
            theta = torch.Tensor([theta])

        if not isinstance(context_samples, torch.Tensor):
            context_samples = torch.Tensor([context_samples])

        for t, x in zip(theta, context_samples):
            t = t.unsqueeze(0) if t.dim() == 0 else t
            x = x.unsqueeze(0) if x.dim() == 0 else x
            key = self._build_hash(t, x)
            try: 
                results.append(self.table[key]["y"])
            except KeyError:
                print(f"Could not match theta {t} and context {x} to a result - taking the nearest neighbor")
                space_hit = self._calc_hash_distance(t, x)
                nearest_key = min(self.table.keys(), key=lambda k: abs(self.table[k]["loc"] - space_hit))

                if abs(self.table[nearest_key]["loc"] - space_hit) > self.threshold:
                    raise ValueError(f"Value {t} and {x} are outside the range of the data by more than {self.threshold} times the max value in the dataset. Please provide a value within the range of the data.")

                results.append(self.table[nearest_key]["y"])

        return np.array(results)

