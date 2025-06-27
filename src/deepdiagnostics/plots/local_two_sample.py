from typing import Union
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure as fig
from matplotlib.axes import Axes as ax


from deepdiagnostics.plots.plot import Display
from deepdiagnostics.metrics.local_two_sample import LocalTwoSampleTest as l2st
from deepdiagnostics.utils.plotting_utils import get_hex_colors
from deepdiagnostics.utils.utils import DataDisplay

class LocalTwoSampleTest(Display): 
    """

    .. note:: 
        A simulator is required to run this plot.
        
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
        self.region_colors = get_hex_colors(n_colors=len(self.percentiles), colorway=self.colorway)
        self.l2st = l2st(model, data, run_id, out_dir, True, self.use_progress_bar, self.samples_per_inference, self.percentiles, self.number_simulations)

    def _make_pairplot_values(self, random_samples, cdf_alphas):
        pp_vals = np.array(
            [np.mean(random_samples <= alpha) for alpha in cdf_alphas]
        )
        return pp_vals

    def plot_name(self):
        return "local_c2st_pp_plot.png" # Resets the name but raises NotImplemented in init
    

    def _data_setup(
            self, 
            linear_classifier:str="MLP", 
            cross_evaluate:bool=True, 
            n_null_hypothesis_trials: int=100, 
            classifier_kwargs: dict = None, 
            n_alpha_samples = 100,
            **kwargs
        ) -> DataDisplay:
        """
        Helper function to collect the data for the local two-sample test plots.
        Args:
            n_alpha_samples (int, optional): Number of samples to use to produce the cdf region. Defaults to 100.
            confidence_region_alpha (float, optional): Opacity of the cdf region plots. Defaults to 0.2.
            n_intensity_bins (int, optional): Number of bins to use when producing the intensity plots. Number of regions. Defaults to 20.
            linear_classifier (Union[str, list[str]], optional): Type of linear classifiers to use. Only MLP is currently implemented. Defaults to "MLP".
            cross_evaluate (bool, optional): Split the validation data in K folds to produce an uncertainty of the classification results. Defaults to True.
            n_null_hypothesis_trials (int, optional): Number of inferences to classify under the null hypothesis. Defaults to 100.
            classifier_kwargs (Union[dict, list[dict]], optional): Additional kwargs for the classifier. Depend on the classifier choice. Defaults to None.

        """
        data_display = DataDisplay()

        self.l2st._collect_data_params()
        data_display.probability, data_display.null_hypothesis_probability = self.l2st.calculate(
                linear_classifier=linear_classifier, 
                cross_evaluate=cross_evaluate, 
                n_null_hypothesis_trials=n_null_hypothesis_trials,
                classifier_kwargs = classifier_kwargs
            )
        
        data_display.cdf_alphas = np.linspace(0, 1, n_alpha_samples)
        data_display.evaluation_data = self.l2st.evaluation_data

        return data_display

    def plot_intensity(
        self,
        data_display: Union[dict, DataDisplay] = None,
        n_intensity_bins: int = 20,
        intensity_plot_ylabel: str = "",
        intensity_plot_xlabel: str = "",
        intensity_plot_title: str = "Local Classifier Intensity Distribution", 
        **kwargs
    ) -> tuple[fig, ax]: 
        """
        Plot 1d or 2d marginal histogram of samples of the density estimator
            with probabilities as color intensity.
        References: https://github.com/JuliaLinhart/lc2st/blob/e221cc326480cb0daadfd2ba50df4eefd374793b/lc2st/graphical_diagnostics.py#L133

        Args:
            data_display (Union[dict, DataDisplay], optional): DataDisplay object containing the necessary
                data for plotting (requires 'probability', 'evaluation_data').
                See metrics.local_two_sample.LocalTwoSampleTest for more details.
            n_intensity_bins (int, optional): Number of bins to use when producing the intensity plots. Number of regions. Defaults to 20.
            intensity_plot_ylabel (str, optional): Y-axis label for the intensity plot. Defaults to "".
            intensity_plot_xlabel (str, optional): X-axis label for the intensity plot. Defaults to "".
            intensity_plot_title (str, optional): Title for the intensity plot. Defaults to "Local Classifier Intensity Distribution".
        """

        self.plot_name = "local_c2st_corner_plot_intensity.png"

        if not isinstance(data_display, DataDisplay):
            data_display = DataDisplay().from_h5(data_display, self.plot_name)


        fig, subplots = plt.subplots(
            len(self.parameter_names), 
            len(self.parameter_names), 
            figsize=(self.figure_size[0]*1.2, self.figure_size[1])
        )
        combos_run = []
        for x_index, x_param in enumerate(self.parameter_names): 
            for y_index, y_param in enumerate(self.parameter_names):
                if ({x_index, y_index} not in combos_run) and (x_index>=y_index): 
                    subplot = subplots[x_index][y_index]

                    if x_index == y_index:
                        features = x_index
                    else:
                        features = [x_index, y_index]

                    self._probability_intensity(
                        subplot, data_display, features=features, n_bins=n_intensity_bins
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

        return fig, subplots

    def _probability_intensity(self, subplot, data_display, features, n_bins=20):
        evaluation_data = data_display.evaluation_data
        norm = Normalize(vmin=0, vmax=1)
        if len(evaluation_data.shape) >= 3:  # Used the kfold option
            evaluation_data = evaluation_data.reshape(
                (
                    evaluation_data.shape[0] * evaluation_data.shape[1],
                    evaluation_data.shape[-1],
                )
            )
            probability = data_display.probability.ravel()

        try:
            # If there is only one feature
            int(features)

            _, bins, patches = subplot.hist(
                evaluation_data[:,features], n_bins, weights=probability, density=True, color=self.parameter_colors[features])

            eval_bins = np.select(
                [evaluation_data[:, features] <= i for i in bins[1:]],
                list(range(n_bins)),
            )

            # get mean predicted proba for each bin
            weights = np.array(
                [probability[eval_bins == i].mean() for i in np.unique(eval_bins)]
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
                        weights[i, j] = probability[
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

    def plot(self,
        data_display: Union[dict, DataDisplay] = None,
        confidence_region_alpha: float = 0.2, 
        pairplot_y_label: str = "",
        pairplot_x_label: str = "",
        pairplot_title: str = "Local Classifier PP-Plot",
        **kwargs) -> tuple[fig, ax]:
        """
        Probability - Probability (P-P) plot for the classifier predicted
            class probabilities in (L)C2ST to assess the validity of a (or several)
            density estimator(s).

        References: https://github.com/JuliaLinhart/lc2st/blob/e221cc326480cb0daadfd2ba50df4eefd374793b/lc2st/graphical_diagnostics.py#L49
            
        Args:
            data_display (Union[dict, DataDisplay], optional): DataDisplay object containing the necessary
                data for plotting (requires 'probability', 'null_hypothesis_probability', 'cdf_alphas').
                See metrics.local_two_sample.LocalTwoSampleTest for more details.
            confidence_region_alpha (float, optional): Opacity of the confidence region in the pair plot
                Defaults to 0.2.
            pairplot_y_label (str, optional): Y-axis label for the pair plot. Defaults
                to "".
            pairplot_x_label (str, optional): X-axis label for the pair plot. Defaults
                to "".
            pairplot_title (str, optional): Title for the pair plot. Defaults to "Local Classifier PP-Plot".
        """
        self.plot_name = "local_c2st_pp_plot.png"
        
        if not isinstance(data_display, DataDisplay):
            data_display = DataDisplay().from_h5(data_display, self.plot_name)

        figure, subplot = plt.subplots(1, 1, figsize=self.figure_size)

        null_cdf = self._make_pairplot_values([0.5] * len(data_display.probability), data_display.cdf_alphas)
        subplot.plot(
            data_display.cdf_alphas, null_cdf, "--", color="black", label="Theoretical Null CDF"
        )

        null_hypothesis_pairplot = np.zeros((len(data_display.cdf_alphas), *null_cdf.shape))

        for t in range(len(data_display.null_hypothesis_probability)):
            null_hypothesis_pairplot[t] = self._make_pairplot_values(
                data_display.null_hypothesis_probability[t], 
                data_display.cdf_alphas
            )

        for percentile, color in zip(self.percentiles, self.region_colors):
            low_null = np.quantile(null_hypothesis_pairplot, percentile / 100, axis=1)
            up_null = np.quantile(
                null_hypothesis_pairplot, (100 - percentile) / 100, axis=1
            )

            subplot.fill_between(
                data_display.cdf_alphas,
                low_null,
                up_null,
                color=color,
                alpha=confidence_region_alpha,
                label=f"{percentile}% Conf. region",
            )

        for prob, label, color in zip(data_display.probability, self.parameter_names, self.parameter_colors):
            pairplot_values = self._make_pairplot_values(prob, data_display.cdf_alphas)
            subplot.plot(data_display.cdf_alphas, pairplot_values, label=label, color=color)

        figure.legend()
        figure.supylabel(pairplot_y_label)
        figure.supxlabel(pairplot_x_label)
        figure.suptitle(pairplot_title)

        return figure, subplot

    def __call__(self, **plot_args) -> None:
        data_display = self._data_setup(**plot_args)

        figure = self.plot(
            data_display=data_display, 
            **plot_args
        )
        self._finish(data_display)

        if plot_args.get("use_intensity_plot", True):
            figure, _ = self.plot_intensity(
                data_display=data_display, 
                **plot_args
            )
            # Don't want to do the full _finish, it will duplicate the metrics
            if self.show:
                plt.show()

            if self.save:
                figure.savefig(f"{self.out_dir.rstrip('/')}/{self.run_id}_{self.plot_name}")
                plt.cla()
