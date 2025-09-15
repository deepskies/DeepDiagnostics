import numpy as np
from tqdm import tqdm
from typing import Any, Sequence

from deepdiagnostics.metrics.metric import Metric

class CoverageFraction(Metric):
    """
    Calculate the coverage of a set number of inferences over different confidence regions. 

    .. code-block:: python 

        from deepdiagnostics.metrics import CoverageFraction 

        samples, coverage = CoverageFraction(model, data, save=False).calculate()
    """

    def __init__(
        self,
        model,
        data,
        run_id,
        out_dir= None,
        save=True,
        use_progress_bar = None,
        samples_per_inference = None,
        percentiles = None,
        number_simulations = None,
    ) -> None:
        
        super().__init__(model, data, run_id, out_dir,
            save,
            use_progress_bar,
            samples_per_inference,
            percentiles,
            number_simulations)
        self._collect_data_params()

    def _collect_data_params(self):
        self.thetas = self.data.thetas
        self.simulator_outcome = self.data.simulator_outcome

    def calculate(self) -> tuple[Sequence, Sequence]:
        """
        Calculate the coverage fraction of the given model and data

        Returns:
            tuple[Sequence, Sequence]: A tuple of the samples tested (M samples, Samples per inference, N parameters) and the coverage over those samples. 
        """

        test_data_len = self.simulator_outcome.shape[0]
        assert self.thetas.shape[0] == test_data_len

        theta_shape = self.thetas.shape[1:]

        theta_posterior_samples = np.empty(
            (self.number_simulations, self.samples_per_inference, *theta_shape)
        )

        theta_true_values = np.empty((self.number_simulations, *theta_shape))

        iterator = range(self.number_simulations)
        if self.use_progress_bar:
            iterator = tqdm(
                iterator,
                desc="Sampling from the posterior for each observation",
                unit=" observation",
            )

        for sim_idx in iterator:
            test_data_idx = self.data.rng.integers(0, test_data_len)

            theta_true_values[sim_idx] = self.thetas[test_data_idx]
            observed_data = self.simulator_outcome[test_data_idx]

            theta_posterior_samples[sim_idx] = self.model.sample_posterior(
                self.samples_per_inference, observed_data
            ).numpy()

        confidence_lower = np.percentile(
            theta_posterior_samples, 50 - np.asarray(self.percentiles) / 2, axis=1
        )  # shape: (len(self.percentile), self.number_simulations, *theta_shape)

        confidence_upper = np.percentile(
            theta_posterior_samples, 50 + np.asarray(self.percentiles) / 2, axis=1
        )  # shape: (len(self.percentile), self.number_simulations, *theta_shape)

        is_covered = np.logical_and(
            confidence_upper - theta_true_values > 0,
            theta_true_values - confidence_lower > 0,
        )  # shape: (len(self.percentile), self.number_simulations, *theta_shape)

        coverage_mean = np.mean(is_covered, axis=1)
        coverage_std = np.std(is_covered, axis=1, ddof=1) / np.sqrt(
            self.number_simulations
        )

        self.output = {
            "coverage": coverage_mean.tolist(),
            "coverage_std": coverage_std.tolist(),
        }

        return theta_posterior_samples, (coverage_mean, coverage_std)

    def __call__(self, **kwds: Any) -> Any:
        self.calculate()
        self._finish()
