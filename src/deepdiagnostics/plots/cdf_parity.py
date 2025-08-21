from typing import Union, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from scipy.stats import ecdf, binom

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
        self.line_cycle =  tuple(get_item("plots_common", "line_style_cycle", raise_exception=False))
        self.labels_dict = {}
        self.theory_alpha = 0.2

    def plot_name(self):
        return "cdf_parity.png"
    
    def _calculate_theory_cdf(self, probability:float, num_bins:int=100) -> tuple[np.array, np.array]:
        """
        Calculate the theoretical limits for the CDF of `distribution` with the percentile `probability`.
        Assumes the distribution is a binomial distribution
        """
        
        n_dims = self.data.n_dims
        bounds = np.zeros((num_bins, 2, n_dims))
        cdf = np.zeros((num_bins, n_dims))

        for dim in range(n_dims):

            # Construct uniform histogram.
            uni_bins = binom(self.samples_per_inference, p=1 / num_bins).ppf(0.5) * np.ones(num_bins)
            uni_bins_cdf = uni_bins.cumsum() / uni_bins.sum()
            # Decrease value one in last entry by epsilon to find valid
            # confidence intervals.
            uni_bins_cdf[-1] -= 1e-9
            lower = [binom(self.samples_per_inference, p=p).ppf(1-probability) for p in uni_bins_cdf]
            upper = [binom(self.samples_per_inference, p=p).ppf(probability) for p in uni_bins_cdf]

            bounds[:, 0, dim] = lower/np.max(lower) 
            bounds[:, 1, dim] = upper/np.max(upper)

            cdf[:, dim] = uni_bins_cdf 
            
        return bounds, cdf

    def _data_setup(self, num_bins:int=100, **kwargs) -> DataDisplay:
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

        thetas = thetas.reshape(
            (self.number_simulations * self.samples_per_inference, n_dims)
        )

        calculated_ecdf = {}
        theory_cdf = {}
        # sample_quartiles based off the first dimension
        # Not always perfect, but it ensures that the quantiles are consistent across all dimensions - required for the residuals
        ecdf_sample = ecdf(posterior_samples[:, 0].ravel())
        
        all_bands = {}
        for interval in percentiles:
            bands, cdf = self._calculate_theory_cdf(interval, num_bins)
            all_bands[interval] = bands

        for dim, name in zip(range(n_dims), self.parameter_names):
            parameter_quantiles = np.linspace(
                np.min(thetas[:, dim]), 
                np.max(thetas[:, dim]), 
                num=num_bins
            )
            ecdf_sample = ecdf(posterior_samples[:, dim].ravel())
            sample_probs_common = ecdf_sample.cdf.evaluate(parameter_quantiles)
            for interval in percentiles:
                all_bands[f"low_theory_probability_{interval}_{name}"] = all_bands[interval][:, 0, dim]
                all_bands[f"high_theory_probability_{interval}_{name}"] = all_bands[interval][:, 1, dim]

            theory_cdf[f"theory_probability_{name}"] = cdf[:, dim]
            calculated_ecdf[f"quantiles_{name}"] = parameter_quantiles
            calculated_ecdf[f"sample_probability_{name}"] = sample_probs_common

        display_data = DataDisplay({
            **calculated_ecdf,
            **all_bands,
            **theory_cdf,  # CDF Isn't calculated differently for percentiles, it's fine to use the last one
            "percentiles": np.array(percentiles),
        })
        return display_data

    def _plot_base_plot(self, data_display, ax,  parameter_name, sample_label, line_style, color, theory_color, theory_line_style): 
        "Just plot the CDF of the posterior ECDF"
        ax.plot(
            data_display[f"quantiles_{parameter_name}"],
            data_display[f"theory_probability_{parameter_name}"],  
            color=theory_color, 
            linestyle=theory_line_style
            )

        ax.plot(
            data_display[f"quantiles_{parameter_name}"], 
            data_display[f"sample_probability_{parameter_name}"],
            ls=line_style, 
            color=color,
        )

    def _plot_theory_intervals(self, data_display, ax, parameter_name, theory_label, color, interval): 
        lower, upper = (
            data_display[f"low_theory_probability_{interval}_{parameter_name}"],
            data_display[f"high_theory_probability_{interval}_{parameter_name}"]
        )

        ax.fill_between(
            data_display[f"quantiles_{parameter_name}"], 
            lower, 
            upper, 
            alpha=self.theory_alpha,
            color=color
        )

    def _plot_theory_intervals_residual(self, data_display, ax, parameter_name, theory_label, color, interval):
        if data_display[f"low_theory_probability_{interval}_{parameter_name}"] is None: 
            bound_low, bound_high = self._compute_intervals(
                data_display[f"theory_probability_{parameter_name}"], 
                interval, 
                self.parameter_names.index(parameter_name)
            )
            low = bound_low - data_display[f"theory_probability_{parameter_name}"]
            high = bound_high - data_display[f"theory_probability_{parameter_name}"]
        else:
            # Use the precomputed values for the fill-between
            low = data_display[f"low_theory_probability_{interval}_{parameter_name}"] - data_display[f"theory_probability_{parameter_name}"]
            high = data_display[f"high_theory_probability_{interval}_{parameter_name}"] - data_display[f"theory_probability_{parameter_name}"]

        ax.fill_between(
            data_display[f"quantiles_{parameter_name}"], 
            low, 
            high, 
            alpha=self.theory_alpha,
            color=color
        )

    def _compute_intervals(self, cdf: np.ndarray, probability: float, dimension:int) -> tuple[np.ndarray, np.ndarray]:
        "Use the Dvoretzky-Kiefer-Wolfowitz confidence bands as an approximation for plotting purposes."

        bound =  np.sqrt(np.log(2.0 / (1 - probability)) / (2.0 * float(cdf.shape[0])))
        lower = cdf[:, dimension] - bound
        upper = cdf[:, dimension] + bound
        return lower, upper
        

    def plot(
            self, 
            data_display: Union[DataDisplay, str], 
            include_residuals: bool = False,
            include_theory_intervals: bool = True,
            display_parameters_separate: bool = False,
            x_label: str = "Quantiles", 
            y_label: str = "CDF",
            title: str = "CDF Parity Plot",
            samples_label = "Posterior Samples",
            theory_label = "Theory",
            theory_color = "gray",
            theory_line_style = "--",
            normalize_view: bool = True,
            theory_alpha: float = 0.2,
            **kwargs
        ) -> tuple["fig", "ax"]: 
        """
        Compute the ECDF for post posteriors samples against the true parameter values.
        Uses [scipy.stats.ecdf](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ecdf.html) to compute the distributions for sampled posterior samples. 

        To show the all distributions on one plot - set `display_parameters_separate` to `False` and verify `normalize_view` is set to `True`, this will ensure the x & y axes is normalized to [0, 1] for all parameters.

        Args: 
            data_display (DataDisplay or str): The data to plot. If a string, it is assumed to be the path to an HDF5 file.
            include_residuals (bool): Whether to include the residuals between the theory and sample distributions.
            include_theory_intervals (bool): Whether to include the theory intervals (percentiles given in the 'percentiles' field of the config) in the plot
            display_parameters_separate (bool): Whether to display each parameter in a separate subplot.
            x_label (str): Label for the x-axis.
            y_label (str): Label for the y-axis.
            title (str): Title of the plot.
            samples_label (str): Label for the samples in the plot.
            theory_label (str): Label for the theory in the plot.
            theory_color (str): Color for the center theory line.
            theory_line_style (str): Line style for center theory line.
            normalize_view (bool): Whether to normalize the x axis of the plot to [0, 1] for all parameters.
            theory_alpha (float): Alpha (transparency) value for the fill between the theory intervals. Between 0 and 1.
        """

        self.theory_alpha = theory_alpha
        color_cycler = iter(plt.cycler("color", self.parameter_colors))
        line_style_cycler = iter(plt.cycler("line_style", self.line_cycle))

        if not isinstance(data_display, DataDisplay):
            data_display = DataDisplay().from_h5(data_display, self.plot_name)

        # Used if theory intervals are included
        theory_color_cycle = self._get_hex_sigma_colors(len(data_display["percentiles"]))


        if include_residuals: 
            row_len = self.figure_size[0] * .8*len(self.parameter_names) if display_parameters_separate else self.figure_size[0]
            figsize = (row_len, 1.5*self.figure_size[1])
            fig, ax = plt.subplots(
                2, len(self.parameter_names) if display_parameters_separate else 1, 
                figsize=figsize, 
                height_ratios=[3, 1], 
                sharex='col',
                sharey='row'
            )
            plt.subplots_adjust(hspace=0.01)

        else: 
            row_len = self.figure_size[0] * .8*len(self.parameter_names) if display_parameters_separate else self.figure_size[0]
            fig, ax = plt.subplots(
                1, len(self.parameter_names) if display_parameters_separate else 1, 
                figsize=(row_len, self.figure_size[1]),
                sharey='row')

        if normalize_view: 
            for parameter_name in self.parameter_names:
                data_display[f"quantiles_{parameter_name}"] = np.linspace(0, 1, num=len(data_display[f"quantiles_{parameter_name}"]))
        
        if include_theory_intervals:  # Each plot needs to iterate over the percentiles in the main plot and the residuals 
            if display_parameters_separate: 
                for index, parameter_name in enumerate(self.parameter_names):
                    plot_ax = ax[index] if not include_residuals else ax[0][index]
                    plot_ax.plot(data_display[f"quantiles_{parameter_name}"], data_display[f"theory_probability_{parameter_name}"], color=theory_color, linestyle=theory_line_style)
                    
                    plot_ax.set_title(f"{samples_label} {parameter_name}")
                    color = next(color_cycler)["color"]
                    line_style = next(line_style_cycler)["line_style"]
                    self._plot_base_plot(data_display, plot_ax, parameter_name, samples_label, line_style, color, theory_color, theory_line_style)

                    for interval_index, interval in enumerate(data_display["percentiles"]):
                        self._plot_theory_intervals(
                            data_display, plot_ax, parameter_name, theory_label, 
                            theory_color_cycle[interval_index], interval
                        )

                        if include_residuals: 
                            self._plot_theory_intervals_residual(
                                data_display, ax[1, index], parameter_name, theory_label, 
                                theory_color_cycle[interval_index], interval
                            )

                    if include_residuals:
                        # Plot the residuals between the theory and sample
                        residual = data_display[f"sample_probability_{parameter_name}"] - data_display[f"theory_probability_{parameter_name}"]

                        ax[1, index].plot(
                            data_display[f"quantiles_{parameter_name}"], 
                            residual,
                            linestyle=line_style,
                            color=color,
                        )
                        ax[1, index].axhline(0, color=theory_color, linestyle=theory_line_style)


            else: # The plot_ax is the same for all parameters
                plot_ax = ax if not include_residuals else ax[0]
                for parameter_name in self.parameter_names:
                    color = next(color_cycler)["color"]
                    line_style = next(line_style_cycler)["line_style"]
                    self._plot_base_plot(data_display, plot_ax, parameter_name, samples_label, line_style, color, theory_color, theory_line_style)
                    for interval_index, interval in enumerate(data_display["percentiles"]):  # iterate over the percentiles
                        if self.parameter_names.index(parameter_name) == 0: # Only plot for the first theory interval when not displaying parameters separately
                            self._plot_theory_intervals(
                                data_display, plot_ax, parameter_name, theory_label, 
                                theory_color_cycle[interval_index], interval
                            )

                            if include_residuals: 
                                self._plot_theory_intervals_residual(
                                    data_display, ax[1], parameter_name, theory_label, 
                                    theory_color_cycle[interval_index], interval
                                )

                    if include_residuals:
                        # Plot the residuals between the theory and sample
                        residual = data_display[f"sample_probability_{parameter_name}"] - data_display[f"theory_probability_{parameter_name}"]

                        ax[1].plot(
                            data_display[f"quantiles_{parameter_name}"], 
                            residual,
                            linestyle=line_style,
                            color=color,
                        )
                        ax[1].axhline(0, color=theory_color, linestyle=theory_line_style)


        else: # Do not include the theory intervals - no fill-betweens here!
            if display_parameters_separate: 
                for index, parameter_name in enumerate(self.parameter_names):
                    # Each parameter gets it's own subplot
                    plot_ax = ax[index] if not include_residuals else ax[0][index]
                    plot_ax.set_title(f"{samples_label} {parameter_name}")
                    color = next(color_cycler)["color"]
                    line_style = next(line_style_cycler)["line_style"]
                    self._plot_base_plot(data_display, plot_ax, parameter_name, samples_label, line_style, color, theory_color, theory_line_style)

                    if include_residuals: 
                        residual = data_display[f"sample_probability_{parameter_name}"] - data_display[f"theory_probability_{parameter_name}"]

                        ax[1, index].plot(
                            data_display[f"quantiles_{parameter_name}"], 
                            residual,
                            linestyle=line_style,
                            color=color,
                            )
                        ax[1, index].axhline(0, color=theory_color, linestyle=theory_line_style)

            else:  # Everything goes on the one column
                plot_ax = ax if not include_residuals else ax[0]
                for index, parameter_name in enumerate(self.parameter_names):
                    color = next(color_cycler)["color"]
                    line_style = next(line_style_cycler)["line_style"]

                    self._plot_base_plot(data_display, plot_ax, parameter_name, samples_label, line_style, color, theory_color, theory_line_style)

                    if include_residuals: 
                        residual = data_display[f"sample_probability_{parameter_name}"] - data_display[f"theory_probability_{parameter_name}"]
                        ax[1].plot(
                            data_display[f"quantiles_{parameter_name}"], 
                            residual,
                            linestyle=line_style,
                            color=color,
                        )
                        ax[1].axhline(0, color=theory_color, linestyle=theory_line_style)


        handles = [
            plt.Line2D([0], [0], color=theory_color, linestyle=theory_line_style, label=theory_label)
        ]
        if include_theory_intervals:
            handles += [
                mpatches.Rectangle((0, 0), 0, 0, facecolor=theory_color_cycle[i], alpha=self.theory_alpha, edgecolor='none', label=f"CDF {int(data_display['percentiles'][i]*100)}% CI {theory_label}")
                for i in range(len(data_display["percentiles"]))
            ]

        if not display_parameters_separate:
            # reset the color and line style cyclers for the handles
            color_cycler = iter(plt.cycler("color", self.parameter_colors))
            line_style_cycler = iter(plt.cycler("line_style", self.line_cycle))
            handles += [
                plt.Line2D([0], [0], color=color['color'], linestyle=line_style['line_style'], label=f"{samples_label} {parameter_name}")
                for parameter_name, color, line_style in zip(self.parameter_names, color_cycler, line_style_cycler)
            ]

        fig.legend(handles=handles)
        fig.suptitle(title)
        fig.supxlabel(x_label)
        fig.supylabel(y_label)

        return fig, ax
