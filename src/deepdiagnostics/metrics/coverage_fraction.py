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
        out_dir= None,
        save=True,
        use_progress_bar = None,
        samples_per_inference = None,
        percentiles = None,
        number_simulations = None,
    ) -> None:
        
        super().__init__(model, data, out_dir,
            save,
            use_progress_bar,
            samples_per_inference,
            percentiles,
            number_simulations)
        self._collect_data_params()

    def _collect_data_params(self):
        self.thetas = self.data.get_theta_true()
        self.context = self.data.true_context()

    def _run_model_inference(self, samples_per_inference, y_inference):
        samples = self.model.sample_posterior(samples_per_inference, y_inference)
        return samples.numpy()

    def calculate(self) -> tuple[Sequence, Sequence]:
        """
        Calculate the coverage fraction of the given model and data

        Returns:
            tuple[Sequence, Sequence]: A tuple of the samples tested (M samples, Samples per inference, N parameters) and the coverage over those samples. 
        """
        all_samples = np.empty(
            (self.number_simulations, self.samples_per_inference, np.shape(self.thetas)[1])
        )
        count_array = []
        iterator = range(self.number_simulations)
        if self.use_progress_bar:
            iterator = tqdm(
                iterator,
                desc="Sampling from the posterior for each observation",
                unit=" observation",
            )
        for sample_index in iterator:
            context_sample = self.context[self.data.rng.integers(0, len(self.context))]
            samples = self._run_model_inference(self.samples_per_inference, context_sample)

            all_samples[sample_index] = samples

            count_vector = []
            # step through the percentile list
            for cov in self.percentiles:
                percentile_lower = 50.0 - cov / 2
                percentile_upper = 50.0 + cov / 2

                # find the percentile for the posterior for this observation
                # this is n_params dimensional
                # the units are in parameter space
                confidence_lower = np.percentile(samples, percentile_lower, axis=0)
                
                confidence_upper = np.percentile(samples, percentile_upper, axis=0)
                

                # this is asking if the true parameter value
                # is contained between the
                # upper and lower confidence intervals
                # checks separately for each side of the 50th percentile

                count = np.logical_and(
                    confidence_upper - self.thetas[sample_index, :].numpy() > 0,
                    self.thetas[sample_index, :].numpy() - confidence_lower > 0,
                )
                count_vector.append(count)

            # each time the above is > 0, adds a count
            count_array.append(count_vector)

        count_sum_array = np.sum(count_array, axis=0)
        frac_lens_within_vol = np.array(count_sum_array)
        coverage = frac_lens_within_vol / len(self.context)

        self.output = coverage

        return all_samples, coverage

    def __call__(self, **kwds: Any) -> Any:
        self.calculate()
        self._finish()
