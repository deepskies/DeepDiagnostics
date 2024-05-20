from typing import Optional, Sequence, Union
import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from plots.plot import Display
from metrics.local_two_sample import LocalTwoSampleTest as l2st
from utils.config import get_item
from utils.plotting_utils import get_hex_colors

class LocalTwoSampleTest(Display): 

    # https://github.com/JuliaLinhart/lc2st/blob/e221cc326480cb0daadfd2ba50df4eefd374793b/lc2st/graphical_diagnostics.py#L133 

    def __init__(self, 
                 model, 
                 data, 
                 save:bool, 
                 show:bool, 
                 out_dir:Optional[str]=None, 
                 percentiles: Optional[Sequence] = None, 
                 parameter_names: Optional[Sequence] = None, 
                 parameter_colors: Optional[Sequence]= None, 
                 figure_size: Optional[Sequence] = None,  
                 num_simulations: Optional[int] = None, 
                 colorway: Optional[str]=None): 
        super().__init__(model, data, save, show, out_dir)
        self.percentiles = percentiles if percentiles is not None else get_item("metrics_common", item='percentiles', raise_exception=False)

        self.param_names = parameter_names if parameter_names is not None else get_item("plots_common", item="parameter_labels", raise_exception=False)
        self.param_colors =  parameter_colors if parameter_colors is not None else get_item("plots_common", item="parameter_colors", raise_exception=False)
        self.figure_size =  figure_size if figure_size is not None else get_item("plots_common", item="figure_size", raise_exception=False)

        colorway = colorway if colorway is not None else get_item(
                "plots_common", "default_colorway", raise_exception=False
            )
        self.region_colors = get_hex_colors(n_colors=len(self.percentiles), colorway=colorway)

        num_simulations = num_simulations if num_simulations is not None else get_item(
            "metrics_common", "number_simulations", raise_exception=False
        )
        self.l2st = l2st(model, data, out_dir, num_simulations)

    def _plot_name(self): 
        return "local_C2ST.png"

    def _make_pairplot_values(self, random_samples): 
        pp_vals = np.array([np.mean(random_samples <= alpha) for alpha in self.cdf_alphas])
        return pp_vals

    def lc2st_pairplot(self, subplot, confidence_region_alpha=0.2): 

        null_cdf = self._make_pairplot_values([0.5] * len(self.probability))
        subplot.plot(
            self.cdf_alphas, null_cdf, "--", color="black", label="Theoretical Null CDF"
        )   

        null_hypothesis_pairplot = np.zeros((len(self.cdf_alphas), *null_cdf.shape))

        for t in range(len(self.null_hypothesis_probability)):
            null_hypothesis_pairplot[t] = self._make_pairplot_values(self.null_hypothesis_probability[t])


        for percentile, color in zip(self.percentiles, self.region_colors): 
            low_null = np.quantile(null_hypothesis_pairplot, percentile/100, axis=1)
            up_null = np.quantile(null_hypothesis_pairplot, (100-percentile)/100, axis=1)

            subplot.fill_between(
                self.cdf_alphas,
                low_null,
                up_null,
                color=color,
                alpha=confidence_region_alpha,
                label=f"{percentile}% Conf. region",
            )

        for prob, label, color in zip(self.probability, self.param_names, self.param_colors):
            pairplot_values = self._make_pairplot_values(prob)
            subplot.plot(self.cdf_alphas, pairplot_values, label=label, color=color)

    def probability_intensity(self, subplot, plot_dims, features, n_bins=20): 
        evaluation_data = self.l2st.evaluation_data

        if len(evaluation_data.shape) >=3: # Used the kfold option 
            evaluation_data = evaluation_data.reshape((
                evaluation_data.shape[0]*evaluation_data.shape[1], 
                evaluation_data.shape[-1]))
            self.probability = self.probability.ravel()

        if plot_dims==1: 

            _, bins, patches = subplot.hist(
                evaluation_data[:,features], n_bins, weights=self.probability, density=True, color=self.param_colors[features])
            
            eval_bins = np.select(
                [evaluation_data[:,features] <= i for i in bins[1:]], list(range(n_bins))
            )

            # get mean predicted proba for each bin
            weights = np.array([self.probability[eval_bins==i].mean() for i in np.unique(eval_bins)]) #df_probas.groupby(["bins"]).mean().probas
            colors = plt.get_cmap(self.colorway)

            for w, p in zip(weights, patches):
                p.set_facecolor(colors(w))  # color is mean predicted proba
            
        else: 

            _, x_edges, y_edges, patches = subplot.hist2d(
                evaluation_data[:,features[0]], 
                evaluation_data[:,features[1]], 
                n_bins, 
                density=True,  color=self.param_colors[features[0]])
            
            eval_bins_dim_1 = np.select(
                [evaluation_data[:,features[0]] <= i for i in x_edges[1:]], list(range(n_bins))
            )
            eval_bins_dim_2 = np.select(
                [evaluation_data[:,features[1]] <= i for i in y_edges[1:]], list(range(n_bins))
            )

            colors = plt.get_cmap(self.colorway)

            weights = np.empty((n_bins, n_bins)) 
            for i in range(n_bins):
                for j in range(n_bins):
                    try:
                        weights[i, j] = self.probability[np.logical_and(eval_bins_dim_1==i, eval_bins_dim_2==j)].mean() 
                    except KeyError:
                        pass 

            for i in range(len(x_edges) - 1):
                for j in range(len(y_edges) - 1):
                    weight = weights[i,j]
                    facecolor = colors(weight)
                    # if no sample in bin, set color to white
                    if weight == np.nan: 
                        facecolor = "white"
                    rect = Rectangle(
                        (x_edges[i], y_edges[j]),
                        x_edges[i + 1] - x_edges[i],
                        y_edges[j + 1] - y_edges[j],
                        facecolor=facecolor,
                        edgecolor="none",
                    )
                    subplot.add_patch(rect)

            
    def _plot(self, 
            use_intensity_plot:bool=True, 
            n_alpha_samples:int=100, 
            confidence_region_alpha:float=0.2,
            n_intensity_bins:int=20, 
            intensity_dimension:int=2,
            intensity_feature_index:Union[int, Sequence[int]]=[0,1],
            linear_classifier:Union[str, list[str]]='MLP', 
            cross_evaluate:bool=True, 
            n_null_hypothesis_trials=100, 
            classifier_kwargs:Union[dict, list[dict]]=None, 
            y_label="Empirical CDF",
            x_label="", 
            title="Local Classifier 2-Sample Test"
        ):

        if use_intensity_plot: 
            if intensity_dimension not in (1, 2): 
                raise NotImplementedError("LC2ST Intensity Plot only implemented in 1D and 2D")
        
            if intensity_dimension == 1: 
                try: 
                    int(intensity_feature_index)
                except TypeError: 
                    raise ValueError(f"Cannot use {intensity_feature_index} to plot, please supply an integer value index.")
                
            else: 
                try: 
                    assert len(intensity_feature_index) == intensity_dimension 
                    int(intensity_feature_index[0])
                    int(intensity_feature_index[1])
                except (AssertionError, TypeError): 
                    raise ValueError(f"Cannot use {intensity_feature_index} to plot, please supply a list of 2 integer value indices.")
                    
        self.l2st(**{
            "linear_classifier":linear_classifier, 
            "cross_evaluate": cross_evaluate, 
            "n_null_hypothesis_trials": n_null_hypothesis_trials, 
            "classifier_kwargs": classifier_kwargs})
        
        self.probability, self.null_hypothesis_probability = self.l2st.output["lc2st_probabilities"], self.l2st.output["lc2st_null_hypothesis_probabilities"]
        
        # Plots to make - 
        # pp_plot_lc2st: https://github.com/JuliaLinhart/lc2st/blob/e221cc326480cb0daadfd2ba50df4eefd374793b/lc2st/graphical_diagnostics.py#L49
        # eval_space_with_proba_intensity: https://github.com/JuliaLinhart/lc2st/blob/e221cc326480cb0daadfd2ba50df4eefd374793b/lc2st/graphical_diagnostics.py#L133 

        n_plots = 1 if not use_intensity_plot else 2
        figure_size = self.figure_size if n_plots==1 else (int(self.figure_size[0]*1.8),self.figure_size[1])
        fig, subplots = plt.subplots(1, n_plots, figsize=figure_size)
        self.cdf_alphas = np.linspace(0, 1, n_alpha_samples)

        self.lc2st_pairplot(subplots[0] if n_plots == 2 else subplots, confidence_region_alpha=confidence_region_alpha)
        if use_intensity_plot: 
            self.probability_intensity(
                subplots[1], 
                intensity_dimension, 
                n_bins=n_intensity_bins, 
                features=intensity_feature_index
            )

        fig.legend()
        fig.supylabel(y_label)
        fig.supxlabel(x_label)
        fig.suptitle(title)