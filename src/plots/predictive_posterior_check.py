from typing import Optional, Sequence
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

from plots.plot import Display
from utils.plotting_utils import get_hex_colors

class PPC(Display):
    def __init__(
        self, 
        model, 
        data, 
        save:bool, 
        show:bool, 
        out_dir:Optional[str]=None, 
        percentiles: Optional[Sequence] = None, 
        use_progress_bar: Optional[bool] = None,
        samples_per_inference: Optional[int] = None,
        number_simulations: Optional[int] = None,
        parameter_names: Optional[Sequence] = None, 
        parameter_colors: Optional[Sequence]= None, 
        colorway: Optional[str]=None
    ):
        super().__init__(model, data, save, show, out_dir, percentiles, use_progress_bar, samples_per_inference, number_simulations, parameter_names, parameter_colors, colorway)

    def _plot_name(self):
        return "predictive_posterior_check.png"

    def get_posterior_2d(self, n_simulator_draws): 
        context_shape = self.data.true_context().shape
        sim_out_shape = self.data.get_simulator_output_shape()
        remove_first_dim = False
        if len(sim_out_shape) != 2: 
            # TODO Debug log with a warning
            sim_out_shape = (sim_out_shape[1], sim_out_shape[2])
            remove_first_dim = True

        self.posterior_predictive_samples = np.zeros((n_simulator_draws, *sim_out_shape))
        self.posterior_true_samples = np.zeros_like(self.posterior_predictive_samples)

        random_context_indices = self.data.rng.integers(0, context_shape[0], n_simulator_draws)
        for index, sample in enumerate(random_context_indices): 
            context_sample = self.data.true_context()[sample, :]
            posterior_sample = self.model.sample_posterior(1, context_sample)

            # get the posterior samples for that context 
            sim_out_posterior =  self.data.simulator.simulate(
                theta=posterior_sample, context_samples = context_sample
            )
            sim_out_true = self.data.simulator.simulate(
                theta=self.data.get_theta_true()[sample, :], context_samples=context_sample
            )
            if remove_first_dim: 
                sim_out_posterior = sim_out_posterior[0]
                sim_out_true = sim_out_true[0]

            self.posterior_predictive_samples[index] = sim_out_posterior
            self.posterior_true_samples[index] = sim_out_true


    def get_posterior_1d(self, n_simulator_draws):
        context_shape = self.data.true_context().shape
        self.posterior_predictive_samples = np.zeros((n_simulator_draws, self.samples_per_inference, context_shape[-1]))
        self.posterior_true_samples = np.zeros_like(self.posterior_predictive_samples)
        self.context = np.zeros((n_simulator_draws, context_shape[-1]))

        random_context_indices = self.data.rng.integers(0, context_shape[0], n_simulator_draws)
        for index, sample in enumerate(random_context_indices): 
            context_sample = self.data.true_context()[sample, :]
            self.context[index] = context_sample

            posterior_sample = self.model.sample_posterior(self.samples_per_inference, context_sample)

            # get the posterior samples for that context 
            self.posterior_predictive_samples[index] = self.data.simulator.simulate(
                theta=posterior_sample, context_samples = context_sample
            )
            self.posterior_true_samples[index] = self.data.simulator.simulate(
                theta=self.data.get_theta_true()[sample, :], context_samples=context_sample
            )

    def _plot_1d(self, 
        subplots: np.ndarray, 
        subplot_index: int,
        n_coverage_sigma: Optional[int] = 3, 
        theta_true_marker: Optional[str] = '^'
    ):
        
        dimension_y_simulation = self.posterior_predictive_samples[subplot_index]
        y_simulation_mean = np.mean(dimension_y_simulation, axis=0).ravel()
        y_simulation_std = np.std(dimension_y_simulation, axis=0).ravel()

        for sigma, color in zip(range(n_coverage_sigma), self.colors):
                subplots[0, subplot_index].fill_between(
                self.context[subplot_index].ravel(),
                y_simulation_mean - sigma * y_simulation_std,
                y_simulation_mean + sigma * y_simulation_std,
                color=color,
                alpha=0.6,
                label=rf"Pred. with {sigma} $\sigma$",
            )

        subplots[0, subplot_index].plot(
            self.context[subplot_index],
            y_simulation_mean - self.true_sigma,
            color="black",
            linestyle="dashdot",
            label="True Input Error"
        )
        subplots[0, subplot_index].plot(
            self.context[subplot_index],
            y_simulation_mean + self.true_sigma,
            color="black",
            linestyle="dashdot",
        )

        true_y = np.mean(self.posterior_true_samples[subplot_index, :, :], axis=0).ravel()
        subplots[1, subplot_index].scatter(
            self.context[subplot_index], 
            true_y, 
            marker=theta_true_marker, 
            label='Theta True'
        )

    def _plot_2d(self, subplots, subplot_index, include_axis_ticks): 
        subplots[1, subplot_index].imshow(self.posterior_predictive_samples[subplot_index])
        subplots[0, subplot_index].imshow(self.posterior_true_samples[subplot_index])

        if not include_axis_ticks: 
            subplots[1, subplot_index].set_xticks([])
            subplots[1, subplot_index].set_yticks([])

            subplots[0, subplot_index].set_xticks([])
            subplots[0, subplot_index].set_yticks([])

    def _plot(
            self, 
            n_coverage_sigma: Optional[int] = 3, 
            true_sigma: Optional[float] = None, 
            theta_true_marker: Optional[str] = '^', 
            n_unique_plots: Optional[int] = 3,
            include_axis_ticks: bool = False,
            title:str="Predictive Posterior", 
            y_label:str="Simulation Output", 
            x_label:str="X"): 
        
        if self.data.simulator_dimensions == 1: 
            self.get_posterior_1d(n_unique_plots)
            self.true_sigma = true_sigma if true_sigma is not None else self.data.get_sigma_true()
            self.colors = get_hex_colors(n_coverage_sigma, self.colorway)

        elif self.data.simulator_dimensions == 2: 
            self.get_posterior_2d(n_unique_plots)

        else: 
            raise NotImplementedError("Posterior Checks only implemented for 1 or two dimensions.")
        
        figure, subplots = plt.subplots(
            2, 
            n_unique_plots, 
            figsize=(int(self.figure_size[0]*n_unique_plots*.6), self.figure_size[1]), 
            sharex=False, 
            sharey=True
        )

        for plot_index in range(n_unique_plots): 
            if self.data.simulator_dimensions == 1: 
                self._plot_1d(subplots, plot_index, n_coverage_sigma, theta_true_marker)

            else: 
                self._plot_2d(subplots, plot_index, include_axis_ticks)


        subplots[1, 0].set_ylabel("True Parameters")
        subplots[0, 0].set_ylabel("Predicted Parameters")

        figure.supylabel(y_label)
        figure.supxlabel(x_label)
        figure.suptitle(title)