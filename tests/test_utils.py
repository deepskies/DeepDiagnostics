import torch
import numpy as np
from deepdiagnostics.data.lookup_table_simulator import LookupTableSimulator


class TestLookupTableSimulator:
    @staticmethod
    def test_lookup_table_simulator():
        # fake data
        data = {
            "context": torch.Tensor([[0.1], [0.2], [0.3], [0.4]]),  # context
            "thetas": torch.Tensor([[1.0], [2.0], [3.0], [4.0]]),  # parameters
            "simulator_outcome": torch.Tensor([[10.0], [20.0], [30.0], [40.0]]),  # outcomes
        }   
        rng = np.random.default_rng(42)
        sim = LookupTableSimulator(data, rng)
        
        assert hasattr(sim, "table")

        # Test generate_context
        contexts = sim.generate_context(2)
        assert contexts.shape == (2, 1)
        assert all(context in data["context"].tolist() for context in contexts) # Only getting contexts from data

        # Test exact match outcome
        theta = torch.Tensor([2.0])
        context = torch.Tensor([0.2])
        outcome = sim.simulate(theta, context)
        assert outcome.shape == (1, 1)
        assert outcome[0] == 20.0

        # Test getting multiple outcomes
        thetas = torch.Tensor([[1.0], [3.0]])
        contexts = torch.Tensor([[0.1], [0.3]])
        outcomes = sim.simulate(thetas, contexts)
        assert outcomes.shape == (2, 1)
        assert outcomes[0] == 10.0
        assert outcomes[1] == 30.0

        # Test nearest neighbor outcome
        theta = torch.Tensor([[2.1]])
        context = torch.Tensor([[0.2]])
        outcome = sim.simulate(theta, context)
        assert outcome[0] == 20.0

    @staticmethod
    def test_lookup_table_simulator_multidim_params(): 
        rng = np.random.default_rng(42)

        data = {
            "context": torch.tensor(rng.random((10, 2))),  # context
            "thetas": torch.tensor(rng.random((10, 3))),  # parameters
            "simulator_outcome": torch.tensor(rng.random((10, 1))),  # outcomes
        }

        rng = np.random.default_rng(42)
        sim = LookupTableSimulator(data, rng)

        # Test exact match outcome
        theta = data["thetas"][:2, :]
        context = data["context"][:2, :]
        outcome = sim.simulate(theta, context)
        assert outcome.shape == (2, 1)
        assert outcome[0] == data["simulator_outcome"][0]
        assert outcome[1] == data["simulator_outcome"][1]

        # Test nearest neighbor outcome
        theta = rng.random((1, 3))
        context = rng.random((1, 2))
        outcome = sim.simulate(theta, context)
        assert outcome.shape == (1, 1)