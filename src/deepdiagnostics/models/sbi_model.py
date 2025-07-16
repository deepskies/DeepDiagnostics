import os
import pickle

from deepdiagnostics.models.model import Model


class SBIModel(Model):
    """
    Load a trained model that was generated with Mackelab SBI :cite:p:`centero2020sbi`. 
    `Read more about saving and loading requirements here <https://sbi-dev.github.io/sbi/latest/faq/question_05_pickling/>`_. 

    Args:
        model_path (str): relative path to a model - must be a .pkl file. 
    """
    def __init__(self, model_path):
        super().__init__(model_path)

    def _load(self, path: str) -> None:
        assert os.path.exists(path), f"Cannot find model file at location {path}"
        assert path.split(".")[-1] == "pkl", "File extension must be 'pkl'"

        with open(path, "rb") as file:
            posterior = pickle.load(file)
        self.posterior = posterior

    def sample_posterior(self, n_samples: int, x_true):
        """
        Sample the posterior 

        Args:
            n_samples (int): Number of samples to draw
            x_true (np.ndarray): Context samples. (must be dims=(n_samples, M))

        Returns:
            np.ndarray: Posterior samples
        """
        return self.posterior.sample(
            (n_samples,), x=x_true, show_progress_bars=False
        ).cpu()  # TODO Unbind from cpu

    def predict_posterior(self, data, context_samples):
        """
        Sample the posterior and then 

        Args:
            data (deepdiagnostics.data.Data): Data module with the loaded simulation
            context_samples (np.ndarray): X values to test the posterior over. 

        Returns:
            np.ndarray: Simulator output 
        """
        posterior_samples = self.sample_posterior(context_samples)
        posterior_predictive_samples = data.simulator(
            posterior_samples, context_samples
        )
        return posterior_predictive_samples
