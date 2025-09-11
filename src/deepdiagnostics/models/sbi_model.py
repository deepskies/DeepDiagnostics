import os
import pickle

from sbi.inference.posteriors.base_posterior import NeuralPosterior

from deepdiagnostics.models.model import Model


class SBIModel(Model):
    """
    Load a trained model that was generated with Mackelab SBI :cite:p:`centero2020sbi`.
    `Read more about saving and loading requirements here <https://sbi-dev.github.io/sbi/latest/faq/question_05_pickling/>`_.

    Args:
        model_path (str): Relative path to a model - must be a .pkl file.
    """

    def __init__(self, model_path):
        super().__init__(model_path)

    def _load(self, path: str) -> None:
        if not os.path.exists(path):
            raise ValueError(f"Cannot find model file at location {path}")
        if path.split(".")[-1] != "pkl":
            raise ValueError("File extension must be 'pkl'")

        with open(path, "rb") as file:
            posterior = pickle.load(file)
        self.posterior = posterior

    @staticmethod
    def save_posterior(
        neural_posterior: NeuralPosterior, path: str, allow_overwrite: bool = False
    ) -> None:
        """
        Save an SBI posterior to a pickle file.

        Args:
            neural_posterior (NeuralPosterior): A neural posterior object.
                Must be an instance of the base class 'NeuralPosterior'
                from the sbi package.
            path (str): Relative path to a model - must be a .pkl file.
            allow_overwrite (bool, optional): Controls whether an attempt to
                overwrite succeeds or results in an error. Defaults to False.
        """
        if not isinstance(NeuralPosterior):
            raise ValueError(
                f"'neural_posterior' must be an instance of the base class 'NeuralPosterior' from the 'sbi' package."
            )
        if os.path.exists(path) and (not allow_overwrite):
            raise ValueError(
                f"The path {path} already exists. To overwrite, use 'save_posterior(..., allow_overwrite=True)'"
            )
        if path.split(".")[-1] != "pkl":
            raise ValueError("File extension must be 'pkl'")

        with open(path, "wb") as file:
            pickle.dump(neural_posterior, file)

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
