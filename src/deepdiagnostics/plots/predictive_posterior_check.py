from typing import Optional, Union, TYPE_CHECKING
import matplotlib.pyplot as plt
from deepdiagnostics.utils.utils import DataDisplay
import numpy as np

from deepdiagnostics.plots.plot import Display
from deepdiagnostics.utils.plotting_utils import get_hex_colors
from deepdiagnostics.utils.simulator_utils import SimulatorMissingError

if TYPE_CHECKING:
    from matplotlib.figure import Figure as fig
    from matplotlib.axes import Axes as ax

class PPC(Display):
    """

        .. note:: 
            A simulator is required to run this plot.

        Show the output of the model's generated posterior against the true values for the same context. 
        Can show either output vs input (in 1D) or examples of simulation output (in 2D). 

        .. code-block:: python
        
            from deepdiagnostics.plots import PPC 

            PPC(model, data, save=False, show=True)(n_unique_plots=5)  # Plot 5 examples

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
        if (self.data is not None) and (not hasattr(self.data, "simulator")): 
            raise SimulatorMissingError("Missing a simulator to run PPC.")

    def plot_name(self):
        return "predictive_posterior_check.png"

    def _get_posterior_2d(self, n_simulator_draws): 
        sim_out_shape = self.data.simulator(
            theta=self.data.thetas[0].unsqueeze(0), 
            n_samples=1
        )[0].shape

        remove_first_dim = False
        if len(sim_out_shape) > 2: 
            # TODO Debug log with a warning
            sim_out_shape = (sim_out_shape[1], sim_out_shape[2])
            remove_first_dim = True

        posterior_predictive_samples = np.zeros((n_simulator_draws, *sim_out_shape))

        random_context_indices = self.data.rng.integers(0, self.data.simulator_outcome.shape[0], n_simulator_draws)
        simulator_true = self.data.simulator_outcome[random_context_indices, :].numpy()
        for index, sample in enumerate(simulator_true): 
            posterior_sample = self.model.sample_posterior(1, sample)

            # get the posterior samples for that context 
            sim_out_posterior =  self.data.simulator(n_samples=1,
                theta=posterior_sample
            )
            if remove_first_dim: 
                sim_out_posterior = sim_out_posterior[0]

            posterior_predictive_samples[index] = sim_out_posterior

        return posterior_predictive_samples, simulator_true

    def _get_posterior_1d(self, n_simulator_draws):
        simulator_outcome_shape = self.data.simulator_dimensions

        posterior_predictive_samples = np.zeros((n_simulator_draws, self.samples_per_inference, simulator_outcome_shape))

        # Sample one random simulator output for each draw
        random_context_indices = self.data.rng.integers(0, self.data.simulator_outcome.shape[0], n_simulator_draws)
        simulator_samples = self.data.simulator_outcome[random_context_indices, :].numpy()
        posterior_predictive_samples = np.zeros((n_simulator_draws, self.samples_per_inference, *simulator_samples[0].shape))

        for index, sample in enumerate(simulator_samples): 
            posterior_sample = self.model.sample_posterior(self.samples_per_inference, sample)
            posterior_predictive_samples[index] = self.data.simulator(n_samples=len(sample),
                theta=posterior_sample
            )

        return posterior_predictive_samples, simulator_samples

    def _data_setup(self, n_unique_plots: Optional[int] = 3, **kwargs) -> DataDisplay:
        true_sigma = None
        if self.data.simulator_dimensions == 1: 
            posterior_predictive_samples, posterior_true_samples = self._get_posterior_1d(n_unique_plots)
            true_sigma = self.data.get_sigma_true()

        elif self.data.simulator_dimensions == 2: 
            posterior_predictive_samples, posterior_true_samples = self._get_posterior_2d(n_unique_plots)

        else: 
            raise NotImplementedError("Posterior Checks only implemented for 1 or two dimensions.")

        return DataDisplay(
            n_unique_plots=n_unique_plots,
            posterior_predictive_samples=posterior_predictive_samples,
            n_dims=self.data.simulator_dimensions,
            posterior_true_samples=posterior_true_samples,
            true_sigma=true_sigma
        )


    def plot(
            self,
            data_display: Union[DataDisplay, dict] = None,
            n_coverage_sigma: Optional[int] = 3, 
            theta_true_marker: Optional[str] = '^', 
            include_axis_ticks: bool = False,
            title:str="Predictive Posterior", 
            y_label:str="Simulation Output", 
            x_label:str="X",
            **kwargs) -> tuple['fig', 'ax']: 
        """
        Args:
            n_coverage_sigma (Optional[int], optional): Show the N different standard dev. sigma of the posterior results. Only used in 1D. Defaults to 3.
            true_sigma (Optional[float], optional): True std. of the known posterior. Used only if supplied. Defaults to None.
            theta_true_marker (Optional[str], optional): Marker to use for output of the true theta parameters. Only used in 1d. Defaults to '^'.
            n_unique_plots (Optional[int], optional): Number of samples of theta/x to use. Each one corresponds to a column. Defaults to 3.
            include_axis_ticks (bool, optional): _description_. Defaults to False.
            title (str, optional): Title of the plot. Defaults to "Predictive Posterior".
            y_label (str, optional): Row label. Defaults to "Simulation Output".
            x_label (str, optional): Column label. Defaults to "X".

        Raises:
            NotImplementedError: If trying to plot results of a simulation with more than 2 output dimensions. 
        """

        if not isinstance(data_display, DataDisplay):
            data_display = DataDisplay().from_h5(data_display, self.plot_name)

        self.colors = get_hex_colors(n_coverage_sigma, self.colorway)
        figure, subplots = plt.subplots(
            2, 
            data_display.n_unique_plots, 
            figsize=(int(self.figure_size[0]*data_display.n_unique_plots*.6), self.figure_size[1]), 
            sharex=False, 
            sharey=True
        )

        for plot_index in range(data_display.n_unique_plots): 
            if data_display.n_dims == 1: 

                dimension_y_simulation = data_display.posterior_predictive_samples[plot_index]
                y_simulation_mean = np.mean(dimension_y_simulation, axis=0).ravel()
                y_simulation_std = np.std(dimension_y_simulation, axis=0).ravel()

                for sigma, color in zip(range(n_coverage_sigma), self.colors):
                        subplots[0, plot_index].fill_between(
                        range(len(y_simulation_mean)),
                        y_simulation_mean - sigma * y_simulation_std,
                        y_simulation_mean + sigma * y_simulation_std,
                        color=color,
                        alpha=0.6,
                        label=rf"Pred. with {sigma} $\sigma$",
                    )

                subplots[0, plot_index].plot(
                    range(len(y_simulation_mean)),
                    y_simulation_mean - data_display.true_sigma,
                    color="black",
                    linestyle="dashdot",
                    label="True Input Error"
                )
                subplots[0, plot_index].plot(
                    range(len(y_simulation_mean)),
                    y_simulation_mean + data_display.true_sigma,
                    color="black",
                    linestyle="dashdot",
                )

                true_y = data_display.posterior_true_samples[plot_index, :].ravel()
                subplots[1, plot_index].scatter(
                    range(len(true_y)),
                    true_y, 
                    marker=theta_true_marker, 
                    label='Theta True'
                )

            else: 
                subplots[1, plot_index].imshow(data_display.posterior_predictive_samples[plot_index])
                subplots[0, plot_index].imshow(data_display.posterior_true_samples[plot_index])

                if not include_axis_ticks: 
                    subplots[1, plot_index].set_xticks([])
                    subplots[1, plot_index].set_yticks([])

                    subplots[0, plot_index].set_xticks([])
                    subplots[0, plot_index].set_yticks([])


        subplots[1, 0].set_ylabel("True Parameters")
        subplots[0, 0].set_ylabel("Predicted Parameters")

        figure.supylabel(y_label)
        figure.supxlabel(x_label)
        figure.suptitle(title)

        return figure, subplots