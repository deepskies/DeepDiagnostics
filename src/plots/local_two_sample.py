from typing import Optional, Sequence, Union
import matplotlib.pyplot as plt
from matplotlib import cm
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

    def probability_intensity(self, subplot, features, n_bins=20): 
        evaluation_data = self.l2st.evaluation_data
        norm = Normalize(vmin=0, vmax=1)
        if len(evaluation_data.shape) >=3: # Used the kfold option 
            evaluation_data = evaluation_data.reshape((
                evaluation_data.shape[0]*evaluation_data.shape[1], 
                evaluation_data.shape[-1]))
            self.probability = self.probability.ravel()

        try: 
            # If there is only one feature
            int(features)

            _, bins, patches = subplot.hist(
                evaluation_data[:,features], n_bins, weights=self.probability, density=True, color=self.param_colors[features])

            eval_bins = np.select(
                [evaluation_data[:,features] <= i for i in bins[1:]], list(range(n_bins))
            )

            # get mean predicted proba for each bin
            weights = np.array([self.probability[eval_bins==i].mean() for i in np.unique(eval_bins)]) #df_probas.groupby(["bins"]).mean().probas
            colors = plt.get_cmap(self.colorway)

            for w, p in zip(weights, patches):
                p.set_facecolor(colors(norm(w)))  # color is mean predicted proba


        except TypeError: 
            _, x_edges, y_edges, image = subplot.hist2d(
                evaluation_data[:,features[0]], 
                evaluation_data[:,features[1]], 
                n_bins, 
                density=True,  color="white")
            
            image.remove() 

            eval_bins_dim_1 = np.select(
                [evaluation_data[:,features[0]] <= i for i in x_edges[1:]], list(range(n_bins))
            )
            eval_bins_dim_2 = np.select(
                [evaluation_data[:,features[1]] <= i for i in y_edges[1:]], list(range(n_bins))
            )

            colors = plt.get_cmap(self.colorway)

            weights = np.empty((n_bins, n_bins)) * np.nan
            for i in range(n_bins):
                for j in range(n_bins):
                    local_and = np.logical_and(eval_bins_dim_1==i, eval_bins_dim_2==j)
                    if local_and.any(): 
                        weights[i, j] = self.probability[np.logical_and(eval_bins_dim_1==i, eval_bins_dim_2==j)].mean() 
                        

            for i in range(len(x_edges) - 1):
                for j in range(len(y_edges) - 1):
                    weight = weights[i,j]
                    facecolor = colors(norm(weight))
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
            linear_classifier:Union[str, list[str]]='MLP', 
            cross_evaluate:bool=True, 
            n_null_hypothesis_trials=100, 
            classifier_kwargs:Union[dict, list[dict]]=None, 
            pairplot_y_label="Empirical CDF",
            pairplot_x_label="", 
            pairplot_title="Local Classifier PP-Plot", 
            intensity_plot_ylabel="", 
            intensity_plot_xlabel="", 
            intensity_plot_title="Local Classifier Intensity Distribution",
        ):

        # Plots to make - 
        # pp_plot_lc2st: https://github.com/JuliaLinhart/lc2st/blob/e221cc326480cb0daadfd2ba50df4eefd374793b/lc2st/graphical_diagnostics.py#L49
        # eval_space_with_proba_intensity: https://github.com/JuliaLinhart/lc2st/blob/e221cc326480cb0daadfd2ba50df4eefd374793b/lc2st/graphical_diagnostics.py#L133 
  
        self.l2st(**{
            "linear_classifier":linear_classifier, 
            "cross_evaluate": cross_evaluate, 
            "n_null_hypothesis_trials": n_null_hypothesis_trials, 
            "classifier_kwargs": classifier_kwargs})
        
        self.probability, self.null_hypothesis_probability = self.l2st.output["lc2st_probabilities"], self.l2st.output["lc2st_null_hypothesis_probabilities"]
        
        fig, subplots = plt.subplots(1, 1, figsize=self.figure_size)
        self.cdf_alphas = np.linspace(0, 1, n_alpha_samples)

        self.lc2st_pairplot(subplots, confidence_region_alpha=confidence_region_alpha)

        fig.legend()
        fig.supylabel(pairplot_y_label)
        fig.supxlabel(pairplot_x_label)
        fig.suptitle(pairplot_title)

        self.plot_name = "local_c2st_pp_plot.png"
        self._finish()

        if use_intensity_plot: 

            fig, subplots = plt.subplots(len(self.param_names), len(self.param_names), figsize=(self.figure_size[0]*1.2, self.figure_size[1]))
            combos_run = []
            for x_index, x_param in enumerate(self.param_names): 
                for y_index, y_param in enumerate(self.param_names): 
                    
                    if ({x_index, y_index} not in combos_run) and (x_index>=y_index): 
                        subplot = subplots[x_index][y_index]

                        if x_index == y_index: 
                            features = x_index
                        else: 
                            features = [x_index, y_index]

                        self.probability_intensity(
                            subplot, 
                            features=features,
                            n_bins=n_intensity_bins
                        )
                        combos_run.append({x_index, y_index})

                    if (x_index<y_index): 
                        subplots[x_index][y_index].axes.get_xaxis().set_visible(False)
                        subplots[x_index][y_index].axes.get_yaxis().set_visible(False)
                    
                    if x_index == len(self.param_names)-1: 
                        subplots[x_index][y_index].set_xlabel(x_param)

                    if y_index == 0: 
                        subplots[x_index][y_index].set_ylabel(y_param)

        for index, y_label in enumerate(self.param_names): 
            subplots[index][0].set_ylabel(y_label)

        for index, x_label in enumerate(self.param_names): 
            subplots[len(self.param_names)-1][-1*index].set_xlabel(x_label)


        fig.supylabel(intensity_plot_ylabel)
        fig.supxlabel(intensity_plot_xlabel)
        fig.suptitle(intensity_plot_title)
        norm = Normalize(vmin=0, vmax=1)

        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=self.colorway), ax=subplots.ravel().tolist())

        self.plot_name = "local_c2st_corner_plot.png"
        self._finish()

    def __call__(self, **plot_args) -> None:
        try: 
            self._data_setup()
        except NotImplementedError: 
            pass 
        try: 
            self._plot_settings() 
        except NotImplementedError: 
            pass 
        
        self._plot(**plot_args)