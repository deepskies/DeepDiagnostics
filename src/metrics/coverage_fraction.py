import numpy as np
from torch import tensor
from tqdm import tqdm
from typing import Any

from metrics.metric import Metric
from utils.config import get_item


class CoverageFraction(Metric):
    """ """

    def __init__(
        self,
        model: Any,
        data: Any,
        out_dir: str | None = None,
        samples_per_inference=None,
        percentiles=None,
        progress_bar: bool = None,
    ) -> None:
        super().__init__(model, data, out_dir)
        self._collect_data_params()

        self.samples_per_inference = (
            samples_per_inference
            if samples_per_inference is not None
            else get_item(
                "metrics_common", "samples_per_inference", raise_exception=False
            )
        )
        self.percentiles = (
            percentiles
            if percentiles is not None
            else get_item("metrics_common", "percentiles", raise_exception=False)
        )
        self.progress_bar = (
            progress_bar
            if progress_bar is not None
            else get_item("metrics_common", "use_progress_bar", raise_exception=False)
        )

    def _collect_data_params(self):
        self.thetas = self.data.theta_true()
        self.y_true = self.data.x_true()

    def _run_model_inference(self, samples_per_inference, y_inference):
        samples = self.model.sample_posterior(samples_per_inference, y_inference)
        return samples

    def calculate(self):
        all_samples = np.empty(
            (len(self.y_true), self.samples_per_inference, np.shape(self.thetas)[1])
        )
        count_array = []
        iterator = enumerate(self.y_true)
        if self.progress_bar:
            iterator = tqdm(
                iterator,
                desc="Sampling from the posterior for each observation",
                unit="observation",
            )
        for y_sample_index, y_sample in iterator:
            samples = self._run_model_inference(self.samples_per_inference, y_sample)
            all_samples[y_sample_index] = samples

            count_vector = []
            # step through the percentile list
            for cov in self.percentiles:
                percentile_lower = 50.0 - cov / 2
                percentile_upper = 50.0 + cov / 2

                # find the percentile for the posterior for this observation
                # this is n_params dimensional
                # the units are in parameter space
                confidence_lower = tensor(
                    np.percentile(samples.cpu(), percentile_lower, axis=0)
                )
                confidence_upper = tensor(
                    np.percentile(samples.cpu(), percentile_upper, axis=0)
                )

                # this is asking if the true parameter value
                # is contained between the
                # upper and lower confidence intervals
                # checks separately for each side of the 50th percentile

                count = np.logical_and(
                    confidence_upper - self.thetas[y_sample_index, :] > 0,
                    self.thetas[y_sample_index, :] - confidence_lower > 0,
                )
                count_vector.append(count)
            # each time the above is > 0, adds a count
            count_array.append(count_vector)

        count_sum_array = np.sum(count_array, axis=0)
        frac_lens_within_vol = np.array(count_sum_array)
        coverage = frac_lens_within_vol / len(self.y_true)

        self.output = coverage

        return all_samples, coverage

    def __call__(self, **kwds: Any) -> Any:
        self.calculate()
        self._finish()
