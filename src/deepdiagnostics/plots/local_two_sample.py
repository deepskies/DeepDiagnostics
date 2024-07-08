from typing import Union
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from deepdiagnostics.plots.plot import Display
from deepdiagnostics.metrics.local_two_sample import LocalTwoSampleTest as l2st
from deepdiagnostics.utils.plotting_utils import get_hex_colors

class LocalTwoSampleTest(Display): 
    """
    Produce plots showing the local evaluation of a posterior estimator for a given observation. 
    Adapted fom Linhart et. al. :cite:p:`linhart2023lc2st`.

    Implements a pair plot, showing regions confidence regions of the CDF in comparison with the null hypothesis classifier results, 
    and an intensity plot, showing the regions of accuracy for each parameter of theta. 

    Uses the following code as reference material: 

    `github.com/JuliaLinhart/lc2st/graphical_diagnostics.py::pp_plot_lc2st <https://github.com/JuliaLinhart/lc2st/blob/e221cc326480cb0daadfd2ba50df4eefd374793b/lc2st/graphical_diagnostics.py#L49>`_.

    `github.com/JuliaLinhart/lc2st/graphical_diagnostics.py::eval_space_with_proba_intensity <https://github.com/JuliaLinhart/lc2st/blob/e221cc326480cb0daadfd2ba50df4eefd374793b/lc2st/graphical_diagnostics.py#L133>`_.


    .. code-block:: python 

        from deepdiagnostics.plots import LC2ST 

        LC2ST(model, data, save=False, show=True).plot(use_intensity_plot=True, n_alpha_samples=100, linear_classifier="MLP", n_null_hypothesis_trials=20)
    """


    def __init__(
        self, 
        model, 
        data, 
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
        
        super().__init__(model, data, save, show, out_dir, percentiles, use_progress_bar, samples_per_inference, number_simulations, parameter_names, parameter_colors, colorway)
        self.region_colors = get_hex_colors(n_colors=len(self.percentiles), colorway=self.colorway)
        self.l2st = l2st(model, data, out_dir, True, self.use_progress_bar, self.samples_per_inference, self.percentiles, self.number_simulations)

    def plot_name(self):
        return "local_C2ST.png"

    def _make_pairplot_values(self, random_samples):
        pp_vals = np.array(
            [np.mean(random_samples <= alpha) for alpha in self.cdf_alphas]
        )
        return pp_vals

    def lc2st_pairplot(self, subplot, confidence_region_alpha=0.2):
        null_cdf = self._make_pairplot_values([0.5] * len(self.probability))
        subplot.plot(
            self.cdf_alphas, null_cdf, "--", color="black", label="Theoretical Null CDF"
        )

        null_hypothesis_pairplot = np.zeros((len(self.cdf_alphas), *null_cdf.shape))

        for t in range(len(self.null_hypothesis_probability)):
            null_hypothesis_pairplot[t] = self._make_pairplot_values(
                self.null_hypothesis_probability[t]
            )

        for percentile, color in zip(self.percentiles, self.region_colors):
            low_null = np.quantile(null_hypothesis_pairplot, percentile / 100, axis=1)
            up_null = np.quantile(
                null_hypothesis_pairplot, (100 - percentile) / 100, axis=1
            )

            subplot.fill_between(
                self.cdf_alphas,
                low_null,
                up_null,
                color=color,
                alpha=confidence_region_alpha,
                label=f"{percentile}% Conf. region",
            )

        for prob, label, color in zip(self.probability, self.parameter_names, self.parameter_colors):
            pairplot_values = self._make_pairplot_values(prob)
            subplot.plot(self.cdf_alphas, pairplot_values, label=label, color=color)

    def probability_intensity(self, subplot, features, n_bins=20):
        evaluation_data = self.l2st.evaluation_data
        norm = Normalize(vmin=0, vmax=1)
        if len(evaluation_data.shape) >= 3:  # Used the kfold option
            evaluation_data = evaluation_data.reshape(
                (
                    evaluation_data.shape[0] * evaluation_data.shape[1],
                    evaluation_data.shape[-1],
                )
            )
            self.probability = self.probability.ravel()

        try:
            # If there is only one feature
            int(features)

            _, bins, patches = subplot.hist(
                evaluation_data[:,features], n_bins, weights=self.probability, density=True, color=self.parameter_colors[features])

            eval_bins = np.select(
                [evaluation_data[:, features] <= i for i in bins[1:]],
                list(range(n_bins)),
            )

            # get mean predicted proba for each bin
            weights = np.array(
                [self.probability[eval_bins == i].mean() for i in np.unique(eval_bins)]
            )  # df_probas.groupby(["bins"]).mean().probas
            colors = plt.get_cmap(self.colorway)

            for w, p in zip(weights, patches):
                p.set_facecolor(colors(norm(w)))  # color is mean predicted proba

        except TypeError:
            _, x_edges, y_edges, image = subplot.hist2d(
                evaluation_data[:, features[0]],
                evaluation_data[:, features[1]],
                n_bins,
                density=True,
                color="white",
            )

            image.remove()

            eval_bins_dim_1 = np.select(
                [evaluation_data[:, features[0]] <= i for i in x_edges[1:]],
                list(range(n_bins)),
            )
            eval_bins_dim_2 = np.select(
                [evaluation_data[:, features[1]] <= i for i in y_edges[1:]],
                list(range(n_bins)),
            )

            colors = plt.get_cmap(self.colorway)

            weights = np.empty((n_bins, n_bins)) * np.nan
            for i in range(n_bins):
                for j in range(n_bins):
                    local_and = np.logical_and(
                        eval_bins_dim_1 == i, eval_bins_dim_2 == j
                    )
                    if local_and.any():
                        weights[i, j] = self.probability[
                            np.logical_and(eval_bins_dim_1 == i, eval_bins_dim_2 == j)
                        ].mean()

            for i in range(len(x_edges) - 1):
                for j in range(len(y_edges) - 1):
                    weight = weights[i, j]
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

    def plot(
        self,
        use_intensity_plot: bool = True,
        n_alpha_samples: int = 100,
        confidence_region_alpha: float = 0.2,
        n_intensity_bins: int = 20,
        linear_classifier: Union[str, list[str]] = "MLP",
        cross_evaluate: bool = True,
        n_null_hypothesis_trials=100,
        classifier_kwargs: Union[dict, list[dict]] = None,
        pairplot_y_label="Empirical CDF",
        pairplot_x_label="",
        pairplot_title="Local Classifier PP-Plot",
        intensity_plot_ylabel="",
        intensity_plot_xlabel="",
        intensity_plot_title="Local Classifier Intensity Distribution"):
        """
        Args:
            use_intensity_plot (bool, optional): Use the additional intensity plots showing regions of prediction accuracy for different theta values. Defaults to True.
            n_alpha_samples (int, optional): Number of samples to use to produce the cdf region. Defaults to 100.
            confidence_region_alpha (float, optional): Opacity of the cdf region plots. Defaults to 0.2.
            n_intensity_bins (int, optional): Number of bins to use when producing the intensity plots. Number of regions. Defaults to 20.
            linear_classifier (Union[str, list[str]], optional): Type of linear classifiers to use. Only MLP is currently implemented. Defaults to "MLP".
            cross_evaluate (bool, optional): Split the validation data in K folds to produce an uncertainty of the classification results. Defaults to True.
            n_null_hypothesis_trials (int, optional): Number of inferences to classify under the null hypothesis. Defaults to 100.
            classifier_kwargs (Union[dict, list[dict]], optional): Additional kwargs for the classifier. Depend on the classifier choice. Defaults to None.
            pairplot_y_label (str, optional): Row label for the pairplot. Defaults to "Empirical CDF".
            pairplot_x_label (str, optional): Column label for the pairplot. Defaults to "".
            pairplot_title (str, optional): Title of the pair plot Defaults to "Local Classifier PP-Plot".
            intensity_plot_ylabel (str, optional): Column label for the intensity plot. Defaults to "".
            intensity_plot_xlabel (str, optional): Row label for the intensity plot. Defaults to "".
            intensity_plot_title (str, optional): Title for the intensity plot. Defaults to "Local Classifier Intensity Distribution".
        """

        # Plots to make -
        # pp_plot_lc2st: https://github.com/JuliaLinhart/lc2st/blob/e221cc326480cb0daadfd2ba50df4eefd374793b/lc2st/graphical_diagnostics.py#L49
        # eval_space_with_proba_intensity: https://github.com/JuliaLinhart/lc2st/blob/e221cc326480cb0daadfd2ba50df4eefd374793b/lc2st/graphical_diagnostics.py#L133 

        self.l2st._collect_data_params()
        self.probability, self.null_hypothesis_probability = self.l2st.calculate(
            linear_classifier=linear_classifier, 
            cross_evaluate=cross_evaluate, 
            n_null_hypothesis_trials = n_null_hypothesis_trials,
            classifier_kwargs = classifier_kwargs
        )
        
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

            fig, subplots = plt.subplots(len(self.parameter_names), len(self.parameter_names), figsize=(self.figure_size[0]*1.2, self.figure_size[1]))
            combos_run = []
            for x_index, x_param in enumerate(self.parameter_names): 
                for y_index, y_param in enumerate(self.parameter_names):
                    if ({x_index, y_index} not in combos_run) and (x_index>=y_index): 
                        subplot = subplots[x_index][y_index]

                        if x_index == y_index:
                            features = x_index
                        else:
                            features = [x_index, y_index]

                        self.probability_intensity(
                            subplot, features=features, n_bins=n_intensity_bins
                        )
                        combos_run.append({x_index, y_index})

                    if x_index < y_index:
                        subplots[x_index][y_index].axes.get_xaxis().set_visible(False)
                        subplots[x_index][y_index].axes.get_yaxis().set_visible(False)
                    
                    if x_index == len(self.parameter_names)-1: 
                        subplots[x_index][y_index].set_xlabel(x_param)

                    if y_index == 0:
                        subplots[x_index][y_index].set_ylabel(y_param)

        for index, y_label in enumerate(self.parameter_names): 
            subplots[index][0].set_ylabel(y_label)

        for index, x_label in enumerate(self.parameter_names): 
            subplots[len(self.parameter_names)-1][-1*index].set_xlabel(x_label)

        fig.supylabel(intensity_plot_ylabel)
        fig.supxlabel(intensity_plot_xlabel)
        fig.suptitle(intensity_plot_title)
        norm = Normalize(vmin=0, vmax=1)

        fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap=self.colorway),
            ax=subplots.ravel().tolist(),
        )

        self.plot_name = "local_c2st_corner_plot.png"
        self._finish()

    def __call__(self, **plot_args) -> None:
        self.plot(**plot_args)
