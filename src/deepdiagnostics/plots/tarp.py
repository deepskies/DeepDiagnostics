from typing import Union
from deepdiagnostics.utils.utils import DataDisplay
import numpy as np
import tarp

import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
from matplotlib.axes import Axes as ax
from matplotlib.figure import Figure as fig

from deepdiagnostics.plots.plot import Display
from deepdiagnostics.utils.config import get_item


class TARP(Display):
    """
    Produce a TARP plot as described in Lemos et. al. :cite:p:`lemos2023samplingbased`.
    Utilizes the implementation from `here <https://github.com/Ciela-Institute/tarp>`_. 

    .. code-block:: python 
        
        from deepdiagnostics.plots import TARP 

        TARP(models, data, show=True, save=False)(
            coverage_sigma=2, 
            coverage_alpha=0.4, 
            y_label="Credibility Level"
        )

    """
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
        
        super().__init__(model, data, run_id, save, show, out_dir, percentiles, use_progress_bar, samples_per_inference, number_simulations, parameter_names, parameter_colors, colorway)
        self.line_style = get_item(
            "plots_common", "line_style_cycle", raise_exception=False
        )
    def plot_name(self):
        return "tarp.png"

    def _data_setup(self, **kwargs) -> DataDisplay:
        self.theta_true = self.data.get_theta_true()
        n_dims = self.theta_true.shape[1]
        posterior_samples = np.zeros(
            (self.number_simulations, self.samples_per_inference, n_dims)
        )
        thetas = np.zeros((self.number_simulations, n_dims))
        for n in range(self.number_simulations):
            sample_index = self.data.rng.integers(0, len(self.theta_true))

            theta = self.theta_true[sample_index, :]
            x = self.data.true_context()[sample_index, :]
            posterior_samples[n] = self.model.sample_posterior(
                self.samples_per_inference, x
            )
            thetas[n] = theta

        posterior_samples = np.swapaxes(posterior_samples, 0, 1)
        return DataDisplay(
            posterior_samples=posterior_samples,
            thetas=thetas,
        )
    def plot_settings(self):
        self.line_style = get_item(
            "plots_common", "line_style_cycle", raise_exception=False
        )

    def _get_hex_sigma_colors(self, n_colors):

        cmap = plt.get_cmap(self.colorway)
        hex_colors = []
        arr = np.linspace(0, 1, n_colors)
        for hit in arr:
            hex_colors.append(plt_colors.rgb2hex(cmap(hit)))

        return hex_colors

    def plot(
        self,
        data_display: Union[DataDisplay, dict] = None,
        coverage_sigma: int = 3,
        reference_point: Union[str, np.ndarray] = "random",
        metric: bool = "euclidean",
        normalize: bool = True,
        bootstrap_calculation: bool = True,
        coverage_alpha: float = 0.2,
        y_label: str = "Expected Coverage",
        x_label: str = "Expected Coverage",
        title: str = "Test of Accuracy with Random Points",
    ) -> tuple["fig", "ax"]:
        """
        Args:
            coverage_sigma (int, optional): Number of sigma to use for coverage. Defaults to 3.
            reference_point (Union[str, np.ndarray], optional): Reference points in the parameter space to test against. Defaults to "random".
            metric (bool, optional): Distance metric ot use between reference points. Use "euclidean" or "manhattan".Defaults to "euclidean".
            normalize (bool, optional): Normalize input space to 1. Defaults to True.
            bootstrap_calculation (bool, optional): Estimate uncertainties using bootstrapped examples. Increases efficiency. Defaults to True.
            coverage_alpha (float, optional): Opacity of the difference coverage sigma. Defaults to 0.2.
            y_label (str, optional): Sup. label on the y axis. Defaults to "Expected Coverage".
            x_label (str, optional): Sup. label on the x axis. Defaults to "Expected Coverage".
            title (str, optional): Title of the entire figure. Defaults to "Test of Accuracy with Random Points".

        """
        if not isinstance(data_display, DataDisplay):
            data_display = DataDisplay().from_h5(data_display, self.plot_name)

        coverage_probability, credibility = tarp.get_tarp_coverage(
            data_display.posterior_samples,
            data_display.thetas,
            references=reference_point,
            metric=metric,
            norm=normalize,
            bootstrap=bootstrap_calculation,
        )
        figure_size = get_item("plots_common", "figure_size", raise_exception=False)
        k_sigma = range(1, coverage_sigma + 1)
        fig, ax = plt.subplots(1, 1, figsize=figure_size)

        ax.plot([0, 1], [0, 1], ls=self.line_style[0], color="k", label="Ideal")
        ax.plot(
            credibility,
            coverage_probability.mean(axis=0),
            ls=self.line_style[-1],
            label="TARP",
        )

        k_sigma = range(1, coverage_sigma + 1)
        colors = self._get_hex_sigma_colors(coverage_sigma)
        for sigma, color in zip(k_sigma, colors):
            ax.fill_between(
                credibility,
                coverage_probability.mean(axis=0)
                - sigma * coverage_probability.std(axis=0),
                coverage_probability.mean(axis=0)
                + sigma * coverage_probability.std(axis=0),
                alpha=coverage_alpha,
                color=color,
            )

        ax.legend()
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.set_title(title)

        return fig, ax