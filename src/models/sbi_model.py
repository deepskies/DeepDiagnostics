import os
import pickle

from models.model import Model


class SBIModel(Model):
    def __init__(self, model_path: str):
        super().__init__(model_path)

    def _load(self, path: str) -> None:
        assert os.path.exists(path), f"Cannot find model file at location {path}"
        assert path.split(".")[-1] == "pkl", "File extension must be 'pkl'"

        with open(path, "rb") as file:
            posterior = pickle.load(file)
        self.posterior = posterior

    def sample_posterior(self, n_samples: int, y_true):  # TODO typing
        return self.posterior.sample(
            (n_samples,), x=y_true, show_progress_bars=False
        ).cpu()  # TODO Unbind from cpu

    def predict_posterior(self, data):
        posterior_samples = self.sample_posterior(data.y_true)
        posterior_predictive_samples = data.simulator(
            data.get_theta_true(), posterior_samples
        )
        return posterior_predictive_samples

    