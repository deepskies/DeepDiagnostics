from typing import Optional, Union
from torch import tensor 
import numpy as np
import tarp 

import matplotlib.pyplot as plt 
import matplotlib.colors as plt_colors

from plots.plot import Display
from utils.config import get_item

class TARP(Display): 
    def __init__(self, model, data, save: bool, show: bool, out_dir: str | None = None):
        super().__init__(model, data, save, show, out_dir)

    def _plot_name(self):
        return "tarp.png"
    
    def _data_setup(self):   
        self.rng = np.random.default_rng(get_item("common", "random_seed", raise_exception=False))
        samples_per_inference = get_item(
            "metrics_common", "samples_per_inference", raise_exception=False
        )        
        num_simulations = get_item("metrics_common", "number_simulations", raise_exception=False)

        n_dims = self.data.theta_true().shape[1]
        self.posterior_samples = np.zeros((num_simulations, samples_per_inference, n_dims))
        self.thetas = np.zeros((num_simulations, n_dims))
        for n in range(num_simulations): 
            sample_index = self.rng.integers(0, len(self.data.theta_true()))

            theta = self.data.theta_true()[sample_index,:]
            x = self.data.x_true()[sample_index,:]
            self.posterior_samples[n] = self.model.sample_posterior(samples_per_inference, x)
            self.thetas[n] = theta

        self.posterior_samples = np.swapaxes(self.posterior_samples, 0,1)
    def _plot_settings(self):
        self.line_style = get_item("plots_common", "line_style_cycle", raise_exception=False)


    def _get_hex_sigma_colors(self, n_colors, colorway=None): 

        if colorway is None: 
            colorway = get_item("plots_common", "default_colorway", raise_exception=False)

        cmap = plt.cm.get_cmap(colorway)
        hex_colors = []
        arr=np.linspace(0,1, n_colors)
        for hit in arr: 
            hex_colors.append(plt_colors.rgb2hex(cmap(hit)))

        return hex_colors

    def _plot(
        self, 
        coverage_sigma:int = 3,
        reference_point:Union[str, np.ndarray]='random', 
        metric:bool="euclidean", 
        normalize:bool=True, 
        bootstrap_calculation:bool=True, 
        coverage_colorway:Optional[str]=None,
        coverage_alpha:float=0.2,
        y_label:str="Expected Coverage", 
        x_label:str="Expected Coverage", 
        title:str='Test of Accuracy with Random Points'
    ):

        coverage_probability, credibility = tarp.get_tarp_coverage(
            self.posterior_samples, 
            self.thetas, 
            references=reference_point, 
            metric = metric, 
            norm = normalize, 
            bootstrap=bootstrap_calculation
        )
        figure_size = get_item("plots_common", "figure_size", raise_exception=False)
        k_sigma = range(1,coverage_sigma+1)
        _, ax = plt.subplots(1, 1, figsize=figure_size)

        ax.plot([0, 1], [0, 1], ls=self.line_style[0], color='k', label="Ideal")
        ax.plot(
            credibility, 
            coverage_probability.mean(axis=0), 
            ls=self.line_style[-1],
            label='TARP')
        
        k_sigma = range(1,coverage_sigma+1)
        colors = self._get_hex_sigma_colors(coverage_sigma, colorway=coverage_colorway)
        for sigma, color in zip(k_sigma, colors):
            ax.fill_between(
                credibility, 
                coverage_probability.mean(axis=0) - sigma * coverage_probability.std(axis=0), 
                coverage_probability.mean(axis=0) + sigma * coverage_probability.std(axis=0), 
                alpha = coverage_alpha, 
                color=color
            )

        ax.legend()
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.set_title(title)
            