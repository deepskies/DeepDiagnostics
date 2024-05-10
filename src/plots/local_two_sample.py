import os
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np 
from matplotlib import rcParams

from plots.plot import Display
from metrics.local_two_sample import LocalTwoSampleTest as l2st
from utils.config import get_item
from utils.plotting_utils import get_hex_colors

class LocalTwoSampleTest(Display): 
    def __init__(self, model, data, save:bool, show:bool, out_dir:Optional[str]=None): 
        super().__init__(model, data, save, show, out_dir)
 
    def _plot_name(self): 
        return "local_C2ST.png"

    def _data_setup(self): 
        self.percentiles = get_item("metrics_common", item='percentiles', raise_exception=False)
        self.region_colors = get_hex_colors(n_colors=len(self.percentiles))

        self.probability, self.null_hypothesis_probability = l2st.calculate()

    def _plot_settings(self): 
        self.param_names = get_item("plots_common", item="parameter_labels", raise_exception=False)
        self.param_colors = get_item("plots_common", item="parameter_colors", raise_exception=False)
        self.figure_size = get_item("plots_common", item="figure_size", raise_exception=False)
    
    def _make_pairplot_values(self, random_samples): 
        pp_vals = [np.mean(random_samples <= alpha) for alpha in self.cdf_alphas]
        return pp_vals

    def lc2st_pairplot(self, subplot, confidence_region_alpha=0.2): 

        subplot.plot(
            self.cdf_alphas, self._make_pairplot_values([0.5] * len(self.probability)), "--", color="black",
        )   

        null_hypothesis_pairplot = np.zeros_like(self.null_hypothesis_probability)
        for t in range(len(self.null_hypothesis_probability)):
            null_hypothesis_pairplot[t] = self._make_pairplot_values(self.null_hypothesis_probability[t])

        for percentile, color in zip(self.percentiles, self.region_colors): 
            low_null = null_hypothesis_pairplot.quantile(percentile/100, axis=1)
            up_null = null_hypothesis_pairplot.quantile((100-percentile)/100, axis=1)

            subplot.fill_between(
                self.cdf_alphas,
                low_null,
                up_null,
                color=color,
                alpha=confidence_region_alpha,
                label=f"{percentile}% confidence region",
            )

        for prob, label, color in zip(self.probability, self.param_names, self.param_colors):
            pairplot_values = self._make_pairplot_values(self, prob)
            subplot.plot(self.cdf_alphas, pairplot_values, label=label, color=color)

    def probability_intensity(self, subplot, dim, n_bins=20, vmin=0, vmax=1): 
        
        if dim==1: 
            _, bins, patches = subplot.hist(df_probas.z, n_bins, density=True, color="green")
            df_probas["bins"] = np.select(
                [df_probas.z <= i for i in bins[1:]], list(range(n_bins))
            )
            # get mean predicted proba for each bin
            weights = df_probas.groupby(["bins"]).mean().probas

            id = list(set(range(n_bins)) - set(df_probas.bins))
            patches = np.delete(patches, id)
            bins = np.delete(bins, id)

            norm = Normalize(vmin=vmin, vmax=vmax)

            for w, p in zip(weights, patches):
                p.set_facecolor(cmap(w)) 
        
        else: 
            _, x, y = np.histogram2d(df_probas.z_1, df_probas.z_2, bins=n_bins)
            df_probas["bins_x"] = np.select(
                [df_probas.z_1 <= i for i in x[1:]], list(range(n_bins))
            )
            df_probas["bins_y"] = np.select(
                [df_probas.z_2 <= i for i in y[1:]], list(range(n_bins))
            )
            # get mean predicted proba for each bin
            prob_mean = df_probas.groupby(["bins_x", "bins_y"]).mean().probas

            weights = np.zeros((n_bins, n_bins))
            for i in range(n_bins):
                for j in range(n_bins):
                    try:
                        weights[i, j] = prob_mean.loc[i].loc[j]
                    except KeyError:
                        # if no sample in bin, set color to white
                        weights[i, j] = np.nan

            norm = Normalize(vmin=vmin, vmax=vmax)
            for i in range(len(x) - 1):
                for j in range(len(y) - 1):
                    facecolor = cmap(norm(weights.T[j, i]))
                    # if no sample in bin, set color to white
                    if weights.T[j, i] == np.nan:
                        facecolor = "white"
                    rect = Rectangle(
                        (x[i], y[j]),
                        x[i + 1] - x[i],
                        y[j + 1] - y[j],
                        facecolor=facecolor,  # color is mean predicted proba
                        edgecolor="none",
                    )
                    subplot.add_patch(rect)

    def _plot(self, 
            use_intensity_plot:bool=True, 
            n_alpha_samples:int=100, 
            confidence_region_alpha:float=0.2,
            n_intensity_bins:int=20, 
            intensity_dimension:int=1,
            intensity_range:tuple=(0,1),
            y_label="",
            x_label="", 
            title=""
        ):

        # Plots to make - 
        # pp_plot_lc2st: https://github.com/JuliaLinhart/lc2st/blob/e221cc326480cb0daadfd2ba50df4eefd374793b/lc2st/graphical_diagnostics.py#L49
        # eval_space_with_proba_intensity: https://github.com/JuliaLinhart/lc2st/blob/e221cc326480cb0daadfd2ba50df4eefd374793b/lc2st/graphical_diagnostics.py#L133 

        n_plots = 1 if not use_intensity_plot else 2
        if intensity_dimension and intensity_dimension not in (1,2): 
            raise NotImplementedError("LC2ST Intensity Plot only implemented in 1D and 2D")
        
        fig, subplots = plt.subplot(1, n_plots, figsize=self.figure_size)
        self.cdf_alphas = np.linspace(0, 1, n_alpha_samples)
        self.lc2st_pairplot(subplots[0], confidence_region_alpha=confidence_region_alpha)
        if use_intensity_plot: 
            self.probability_intensity(
                subplots[1], 
                intensity_dimension, 
                n_bins=n_intensity_bins, 
                vmin=intensity_range[0], 
                vmax=intensity_range[1]
            )

        fig.legend()
        fig.supylabel(y_label)
        fig.supxlabel(x_label)
        fig.set_title(title)
