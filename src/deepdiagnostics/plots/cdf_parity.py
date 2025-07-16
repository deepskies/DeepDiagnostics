from typing import Union, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ecdf

from deepdiagnostics.plots.plot import Display
from deepdiagnostics.utils.config import get_item
from deepdiagnostics.utils.utils import DataDisplay

if TYPE_CHECKING:
    from matplotlib.figure import Figure as fig
    from matplotlib.axes import Axes as ax

class CDFParityPlot(Display):
    def __init__(
        self, 
        model, 
        data, 
        run_id,
        save, 
        show, 
        out_dir=None, 
        percentiles = None, 
        use_progress_bar= None,
        samples_per_inference = None,
        number_simulations= None,
        parameter_names = None, 
        parameter_colors = None, 
        colorway =None): 
        """
        Compute the ECDF for post posteriors samples against the true parameter values.
        Uses [scipy.stats.ecdf](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ecdf.html) to compute the distributions from both given data and samples from the posterior

        ..code-block:: python

            from deepdiagnostics.plots import CDFParityPlot 

            CDFParityPlot(model, data, save=False, show=True)()
        
        """
        super().__init__(model, data, run_id, save, show, out_dir, percentiles, use_progress_bar, samples_per_inference, number_simulations, parameter_names, parameter_colors, colorway)


    def plot_name(self):
        return "cdf_parity.png"
    
    def _data_setup(self) -> DataDisplay:
        if all([p >= 1 for p in self.percentiles]): 
            percentiles = [p/100 for p in self.percentiles]
        else: 
            percentiles = self.percentiles or [0.95]

        n_dims = self.data.n_dims
        theta_true = self.data.get_theta_true()
        posterior_samples = np.zeros(
            (self.number_simulations, self.samples_per_inference, n_dims)
        )
        thetas = np.zeros((self.number_simulations, self.samples_per_inference, n_dims))


        for n in range(self.number_simulations):
            sample_index = self.data.rng.integers(0, len(theta_true))

            theta = theta_true[sample_index, :]
            x = self.data.true_context()[sample_index, :]
            posterior_samples[n] = self.model.sample_posterior(
                self.samples_per_inference, x
            )

            thetas[n] = np.array([theta for _ in range(self.samples_per_inference)])

        theory_cdf = {}
        sample_cdf = {}
        for dim, name in zip(range(n_dims), self.parameter_names):
            cdf = ecdf(thetas[:, dim].ravel())
            ecdf_sample = ecdf(posterior_samples[:, dim].ravel())

            # Create a common grid for comparison
            # Use the union of both quantile grids for accurate interpolation
            all_quantiles = np.unique(np.concatenate([cdf.cdf.quantiles, ecdf_sample.cdf.quantiles]))
            
            # Evaluate both CDFs at the common grid
            theory_probs_common = cdf.cdf.evaluate(all_quantiles)
            sample_probs_common = ecdf_sample.cdf.evaluate(all_quantiles)

            theory_cdf[f"theory_probability_{name}"] = theory_probs_common
            theory_cdf[f"quantiles_{name}"] = all_quantiles
            theory_cdf[f"sample_probability_{name}"] = sample_probs_common

            for interval in percentiles:

                range_cdf = cdf.cdf.confidence_interval(confidence_level=interval)

                # Evaluate confidence intervals at common grid
                theory_cdf[f"low_theory_probability_{interval}_{name}"] = range_cdf.low.evaluate(all_quantiles)
                theory_cdf[f"high_theory_probability_{interval}_{name}"] = range_cdf.high.evaluate(all_quantiles)

            sample_cdf[name] = ecdf_sample

        display_data = DataDisplay({
            **theory_cdf,
            "percentiles": np.array(percentiles),
        })

        return display_data

    def plot(
            self, 
            data_display: Union[DataDisplay, str], 
            include_residuals: bool = False,
            include_theory_intervals: bool = True,
            x_label: str = "", 
            y_label: str = "",
            title: str = "CDF Parity Plot",
            samples_label = "Posterior Samples",
            theory_label = "Theory",
            samples_color = "k", 
            samples_line_style = "-",
            theory_color = "gray",
            theory_line_style = "--",
            **kwargs
        ) -> tuple["fig", "ax"]: 
        """
        Compute the ECDF for post posteriors samples against the true parameter values.
        Adds a new horizontal axis for each parameter included. 

        Args:
            data_display : Union[DataDisplay, str]
                DataDisplay object or path to h5 file containing the data to plot.
            include_residuals : bool, optional
                Whether to include the residuals in the plot, by default False
            include_theory_intervals : bool, optional
                Whether to include the theory intervals in the plot, by default True
            x_label : str, optional
                Label for the x-axis, by default ""
            y_label : str, optional
                Label for the y-axis, by default ""
            title : str, optional
                Title of the plot, by default "CDF Parity Plot"
            samples_label : str, optional
                Label for the posterior samples line, by default "Posterior Samples"
            theory_label : str, optional
                Label for the theory line, by default "Theory"
            samples_color : str, optional
                Color for the posterior samples line, by default "k"
            samples_line_style : str, optional
                Line style for the posterior samples line, by default "-"
            theory_color : str, optional
                Color for the theory line, by default "gray"
            theory_line_style : str, optional
                Line style for the theory line, by default "--"
        """

        if not isinstance(data_display, DataDisplay):
            data_display = DataDisplay().from_h5(data_display, self.plot_name)

        # Used if theory intervals are included
        colors = self._get_hex_sigma_colors(len(data_display["percentiles"]))

        if include_residuals: 
            fig, ax = plt.subplots(len(self.parameter_names), 2, figsize=(6*len(self.parameter_names), 12), height_ratios=[3, 1])
            residual_ax = ax[1]
            ax = ax[0]

        else: 
            fig, ax = plt.subplots(1, len(self.parameter_names), figsize=(6*len(self.parameter_names), 6))
            if len(self.parameter_names) == 1:
                ax = [ax]

        for i, name in enumerate(self.parameter_names):

            ax[i].plot(
                data_display[f"quantiles_{name}"], 
                data_display[f"sample_probability_{name}"], 
                linestyle=samples_line_style,
                color=samples_color,
                label=samples_label
            )

            ax[i].set_title(name)

            if include_residuals:

                # Now we can directly compare probabilities since they're evaluated at the same quantiles
                residual = (data_display[f"theory_probability_{name}"] - data_display[f"sample_probability_{name}"])

                residual_ax[i].plot(
                    data_display[f"quantiles_{name}"], 
                    residual,
                    linestyle=samples_line_style,
                    color=samples_color,
                    )
                if include_theory_intervals: 
                    for interval, color in zip(data_display["percentiles"], colors):
                        low = data_display[f"low_theory_probability_{interval}_{name}"] - data_display[f"theory_probability_{name}"]
                        high = data_display[f"high_theory_probability_{interval}_{name}"] - data_display[f"theory_probability_{name}"]

                        residual_ax[i].fill_between(
                            data_display[f"quantiles_{name}"], 
                            low, 
                            high, 
                            alpha=0.2, color=color
                        )
                else: 
                    residual_ax[i].axhline(0, color=theory_color, linestyle=theory_line_style)

            if include_theory_intervals:
                
                for interval, color in zip(data_display["percentiles"], colors):
                    ax[i].fill_between(
                        data_display[f"quantiles_{name}"], 
                        data_display[f"low_theory_probability_{interval}_{name}"], 
                        data_display[f"high_theory_probability_{interval}_{name}"], 
                        alpha=0.2, color=color
                    )

                theory_labels = [f"CDF {int(interval*100)}% CI {theory_label or ''}" for interval in data_display["percentiles"]]
                theory_handles = [
                    plt.Line2D([0], [0], color=color) for color in colors
                ]
            else: 
                # Just plot the standard theory
                ax[i].plot(
                    data_display[f"quantiles_{name}"], 
                    data_display[f"theory_probability_{name}"], 
                    linestyle=theory_line_style, color=theory_color
                )
                theory_labels = [theory_label]
                theory_handles = [
                    plt.Line2D([0], [0], color=theory_color, linestyle=theory_line_style)
                ]

        handles = [
            plt.Line2D([0], [0], color=samples_color, linestyle=samples_line_style),
            *theory_handles
        ]
        labels = [samples_label, *theory_labels]

        fig.legend(handles, labels=labels)
        fig.suptitle(title)
        fig.supxlabel(x_label)
        fig.supylabel(y_label)

        return fig, ax
