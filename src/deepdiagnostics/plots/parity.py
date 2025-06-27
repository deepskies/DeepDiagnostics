from typing import Union, TYPE_CHECKING
import matplotlib.pyplot as plt
from deepdiagnostics.utils.utils import DataDisplay
import numpy as np



from deepdiagnostics.plots.plot import Display

if TYPE_CHECKING:
    from matplotlib.figure import Figure as fig
    from matplotlib.axes import Axes as ax

class Parity(Display):
    """
        Show plots directly comparing the posterior vs. true theta values. Make a plot that is (number of selected metrics) X dimensions of theta. 
        Includes the option to show differences, residual, and percent residual as plots under the main parity plot. 

    .. code-block:: python 
    
        from deepdiagnostics.plots import Parity

        Parity(model, data, show=True, save=False)(
            n_samples=200 # 200 samples of the posterior
            include_residual = True # Add a plot showing the residual
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

    def plot_name(self):
        return "parity.png"

    def _data_setup(self, n_samples: int = 80, **kwargs) -> DataDisplay:

        context_shape = self.data.true_context().shape
        posterior_sample_mean = np.zeros((n_samples, self.data.n_dims))
        posterior_sample_std = np.zeros_like(posterior_sample_mean)
        true_samples = np.zeros_like(posterior_sample_mean)

        random_context_indices = self.data.rng.integers(0, context_shape[0], n_samples)
        for index, sample in enumerate(random_context_indices): 

            posterior_sample = self.model.sample_posterior(self.samples_per_inference,  self.data.true_context()[sample, :]).numpy()
            posterior_sample_mean[index] = np.mean(posterior_sample, axis=0)
            posterior_sample_std[index] = np.std(posterior_sample, axis=0)

            true_samples[index] = self.data.get_theta_true()[sample, :]

        return DataDisplay(
            n_dims=self.data.n_dims,
            true_samples=true_samples, 
            posterior_sample_mean=posterior_sample_mean,
            posterior_sample_std=posterior_sample_std,

        )

    def plot(
        self,
        data_display: Union[str, DataDisplay] = None, 
        include_difference: bool = False, 
        include_residual: bool = False, 
        include_percentage: bool = False,
        show_ideal: bool = True,
        errorbar_color: str = 'black',
        title:str="Parity", 
        y_label:str=r"$\theta_{predicted}$", 
        x_label:str=r"$\theta_{true}$"
    ) -> tuple["fig", "ax"]: 
        """
        Args:
            include_difference (bool, optional): Include a plot that shows the difference between the posterior and true. Defaults to False.
            include_residual (bool, optional): Include a plot that shows the residual between posterior and true. Defaults to False.
            include_percentage (bool, optional): Include a plot that shows the residual as a percent between posterior and true. Defaults to False.
            show_ideal (bool, optional): Include a line showing where posterior=true. Defaults to True.
            errorbar_color (str, optional): _description_. Defaults to 'black'.
            title (str, optional): Title of the plot. Defaults to "Parity".
            y_label (str, optional): y axis label. Defaults to r"$\theta_{predicted}$".
            x_label (str, optional): x axis label. Defaults to r"$\theta_{true}$".
        """
        if not isinstance(data_display, DataDisplay):
            data_display = DataDisplay().from_h5(data_display, self.plot_name)

        # parity - predicted vs true
        # parity difference plot = true - predicted vs. true (y-axis vs x-axis)
        # residual: (true - predicted / true) vs. true
        # percentage: (true - predicted / true)*100 vs. true

        height_ratios = [3]
        n_rows = 1 
        if include_difference: 
            n_rows += 1 
            height_ratios.append(1)
        if include_residual: 
            n_rows += 1
            height_ratios.append(1)
        if include_percentage: 
            n_rows += 1
            height_ratios.append(1)

        figure, subplots = plt.subplots(
            nrows=n_rows, 
            ncols=data_display.n_dims, 
            figsize=(int(self.figure_size[0]*data_display.n_dims*.8), int(self.figure_size[1]*n_rows*.6)), 
            height_ratios=height_ratios,
            sharex="col", 
            sharey=False)

        figure.suptitle(title)
        figure.supxlabel(x_label)
        figure.supylabel(y_label)

        for theta_dimension in range(data_display.n_dims): 

            true = data_display.true_samples[:, theta_dimension]
            posterior_sample = data_display.posterior_sample_mean[:, theta_dimension]
            posterior_errorbar = data_display.posterior_sample_std[:, theta_dimension]

            title = self.parameter_names[theta_dimension]
            
            if n_rows != 1: 
                parity_plot = subplots[0, theta_dimension]
                subplots[0, 0].set_ylabel("Parity")

            else: 
                parity_plot = subplots[theta_dimension]
                subplots[0].set_ylabel("Parity")


            parity_plot.title.set_text(title)
            parity_plot.errorbar(true, posterior_sample, yerr=posterior_errorbar, fmt="o", ecolor=errorbar_color)

            if show_ideal: 
                parity_plot.plot([0, 1], [0, 1], transform=parity_plot.transAxes, color='black', linestyle="--")
            
            row_index = 1
            if include_difference: 
                subplots[row_index, 0].set_ylabel("Difference")
                subplots[row_index, theta_dimension].scatter(true, true-posterior_sample)
                if show_ideal: 
                    subplots[row_index, theta_dimension].hlines(0, xmin = true.min(), xmax=true.max(), alpha=0.4, color='black', linestyle="--")

                row_index += 1 

            if include_residual: 
                subplots[row_index, 0].set_ylabel("Residuals")
                subplots[row_index, theta_dimension].scatter(true, (true-posterior_sample)/true)
                if show_ideal: 
                    subplots[row_index, theta_dimension].hlines(0, xmin = true.min(), xmax=true.max(), alpha=0.4, color='black', linestyle="--")

                row_index += 1 

            if include_percentage: 
                subplots[row_index, 0].set_ylabel("Percentage")
                subplots[row_index, theta_dimension].scatter(true, (true-posterior_sample)*100/true)
                if show_ideal: 
                    subplots[row_index, theta_dimension].hlines(0, xmin = true.min(), xmax=true.max(), alpha=0.4, color='black', linestyle="--")

                row_index += 1 
        
        return figure, subplots