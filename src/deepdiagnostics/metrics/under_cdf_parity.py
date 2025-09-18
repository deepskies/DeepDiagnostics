from typing import Union, TYPE_CHECKING, Any, Optional, Sequence

import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from scipy.stats import ecdf, binom

from deepdiagnostics.data import data
from deepdiagnostics.models import model
from deepdiagnostics.utils.config import get_item

import numpy as np
from tqdm import tqdm
from typing import Any, Sequence

from torch import tensor

from deepdiagnostics.metrics.metric import Metric


class CDFParityAreaUnderCurve(Metric):
    def __init__(
         self,
         model,
         data,
         run_id,
         out_dir= None,
         save = True,
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
        self.n_dims = self.data.n_dims
        theta_true = self.data.thetas#self.data.get_theta_true()#self.data.thetas
        self.posterior_samples = np.zeros(
             (self.number_simulations, self.samples_per_inference, self.n_dims)
        )
        thetas = np.zeros((self.number_simulations, self.samples_per_inference, self.n_dims))

        for n in range(self.number_simulations):
            sample_index = self.data.rng.integers(0, len(theta_true))
            
            theta = theta_true[sample_index, :]
            x = self.data.context[sample_index, :]
            self.posterior_samples[n] = self.model.sample_posterior(
                self.samples_per_inference, x
            )

            thetas[n] = np.array([theta for _ in range(self.samples_per_inference)])

        thetas = thetas.reshape(
            (self.number_simulations * self.samples_per_inference, self.n_dims)
        )

        """
        Compute the ECDF for post posteriors samples against the true parameter values.
        Uses [scipy.stats.ecdf](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ecdf.html) to compute the distributions from both given data and samples from the posterior

        ..code-block:: python

            from deepdiagnostics.plots import CDFParityPlot

            CDFParityPlot(model, data, save=False, show=True)()

        """
    def calculate(self) -> dict[str, float]:
        #one dimensional 
        #ecdf_sample = ecdf(self.posterior_samples[:, 0].ravel())
        #print("ecdf_sample ", ecdf_sample)
        results = {}
        # Loop through each parameter dimension
        for d in range(self.n_dims):
          # Flatten posterior samples for this dimension
          ecdf_sample = ecdf(self.posterior_samples[:, :, d].ravel())
          # Calculate the area under the ECDF curve
          # Compute the ECDF
          x = ecdf_sample.cdf.quantiles
          y = ecdf_sample.cdf.probabilities
          #print("x",x," y ", y)
          area_under_ecdf = np.trapezoid(y, x)
          #print(f"Area under the ECDF: {area_under_ecdf:.4f}")
          #auc="Area_Under_Curve"
          results[f"Area_Under_Curve_dim{d}"] = area_under_ecdf
        self.output = results#{auc: area_under_ecdf} ##Need to run calculate
        return self.output
    def __call__(self, **kwds: Any) -> Any:
        self._collect_data_params()
        self.calculate()
        self._finish()
