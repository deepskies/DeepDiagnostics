import numpy as np
from scipy import stats as sps
from typing import Any

from metrics.metric import Metric
from utils.config import get_item


class SBIstats(Metric):
    def __init__(self, samples, truths, samps_already_sorted = False):
        """
        this starts off by sorting the samples (if they aren't sorted already), then it generates
        the sorted, *normalized* rank_arr (eg, if the truth value is the 400th largest out of 1000,
        it is assigned rank 0.4)
        the rank array can be binned for a normal rank histogram, and we also use it to calculate
        the coverage_arr by default for 50 credible intervals between 0 and 1 inclusive
        finally, we calculate the rank_cdf and generate a KDE of the residuals

        we provide methods to compare the distribution to U(0,1) via: the KS test, the difference
        of the rank_arr and the uniform distribution, and the difference of the coverage_arr and
        the uniform distribution

        we also provide convenience functions for the sample percentiles and the respective residuals

        possible future functionality would be to include the option for a self-contained `sbi` run,
        given a dictionary that has labelled training data, a well-defined prior, labelled test data,
        and some number N_samples of samples to draw from the posterior
        """
        super().__init__(model, data, out_dir)
        self._collect_data_params()
        
    def _collect_data_params(self):
        self.thetas = self.data.theta_true()
        self.y_true = self.data.x_true()
        
        self.samples_per_inference = get_item(
            "metrics_common", "samples_per_inference", raise_exception = False
        )
        self.percentiles = get_item(
            "metrics_common", "percentiles", raise_exception = False
        )
    
    def _run_model_inference(self, samples, truths, samps_already_sorted = False):
        # samples = self.model.sample_posterior(samples_per_inference, y_inference)
        # return samples
        self.samples = np.sort(samples, axis = 1) if not samps_already_sorted else samples
        self.truths = truths
        self.ntruths, self.nsamps = self.samples.shape
        
        self.rank_arr = np.sort(
            np.array([np.searchsorted(self.samples[a], self.truths[a]) for a in range(self.ntruths)])
                                )/self.nsamps
        
        self.coverage_arr = np.linspace(0, 1), np.array([((self.rank_arr > 0.5-c) & (self.rank_arr <= 0.5+c)).sum()
                                      for c in np.linspace(0, 0.5)])
        
        r_x, r_y = np.unique(self.rank_arr, return_counts = True)
        r_y = np.cumsum(r_y) # overwriting itself -- did this for memory reasons but feel free to change
        r_y = r_y/r_y[-1]
        
        self.rank_cdf = r_x, r_y

        self.kde_resid = sps.gaussian_kde((self.samples/self.truths[:,None]).flatten()-1)

    def _coverage_arr(self, percentiles = None):
        self.coverage_arr = percentiles, np.array([((self.rank_arr > 0.5-c) & (self.rank_arr <= 0.5+c)).sum()
                                      for c in percentiles/2])
        return self.coverage_arr

    def _arr_percentiles(self, percentiles = [10,50,90]):
        inds = np.array(percentiles)*len(self.samples)//100
        return np.array([x[inds] for x in self.samples])
    
    def ksU_rank(self):
        return sps.ks_1samp(self.rank_arr, sps.uniform.cdf)

    def diffU_rank(self):
        return self.rank_arr - np.linspace(0, 1, len(self.rank_arr))
    
    def ksU_coverage(self):
        return sps.ks_1samp(self.coverage_arr/len(self.truths), sps.uniform.cdf)

    def diffU_coverage(self):
        return self.coverage_arr/len(self.truths) - np.linspace(0, 1, len(self.coverage_arr))

    def arr_relresid_percentiles(self, percentiles = [10,50,90]):
        return self._arr_percentiles(percentiles)/self.truths[:,None] - 1

    def tot_relresid_percentiles(self, percentiles = [10,50,90]):
        return np.percentile(self.samples/self.truths[:,None] - 1, percentiles, axis=1)
    
    def __call__(self, **kwds: Any) -> Any:
        self.calculate()
        self._finish()

    