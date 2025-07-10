from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes as ax
from matplotlib.figure import Figure as fig

from deepdiagnostics.metrics.coverage_fraction import CoverageFraction as coverage_fraction_metric
from deepdiagnostics.plots.plot import Display
from deepdiagnostics.utils.config import get_item
from deepdiagnostics.utils.utils import DataDisplay


class CoverageFraction(Display):
    """
    Show posterior regions of confidence as a function of percentiles. 
    Each parameter of theta is plotted against a coverage fraction for each given theta. 

    .. code-block:: python 
    
        from deepdiagnostics.plots import CoverageFraction

        CoverageFraction(model, data, show=True, save=False)(
            figure_alpha=0.8, 
            legend_loc="upper left", 
            reference_line_label="Ideal"
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
            
        self.n_parameters = len(self.parameter_names)
        self.line_cycle =  tuple(get_item("plots_common", "line_style_cycle", raise_exception=False))

    def plot_name(self):
        return "coverage_fraction.png"

    def _data_setup(self, percentile_step_size:float=1) -> DataDisplay:
        _, (coverage_mean, coverage_std) = coverage_fraction_metric(
            self.model, self.data, self.run_id, out_dir=None, percentiles=np.arange(0, 100, percentile_step_size), use_progress_bar=self.use_progress_bar
        ).calculate()
        return DataDisplay(
            coverage_fractions=coverage_mean, 
            coverage_percentiles=np.arange(0, 100, percentile_step_size),
            coverage_std=coverage_std
        )

    def _plot_residual(self, data_display, ax, figure_alpha, line_width, reference_line_style, include_coverage_residual_std, include_ideal_range):
        color_cycler = iter(plt.cycler("color", self.parameter_colors))
        line_style_cycler = iter(plt.cycler("line_style", self.line_cycle))
        percentile_array = data_display.coverage_percentiles / 100.0

        ax.plot([0,1], [0, 0], reference_line_style, lw=line_width, zorder=1000)

        for i in range(self.n_parameters):
            color = next(color_cycler)["color"]
            line_style = next(line_style_cycler)["line_style"]

            residual = data_display.coverage_fractions[:, i] - np.linspace(0, 1, len(data_display.coverage_fractions[:,i]))

            ax.plot(
                percentile_array,
                residual,
                alpha=figure_alpha,
                lw=line_width*.8,
                linestyle=line_style,
                color=color,
                label=self.parameter_names[i],
            )
            if include_coverage_residual_std:

                ax.fill_between(
                    percentile_array,
                    residual - data_display.coverage_std[:, i],
                    residual + data_display.coverage_std[:, i],
                    color=color,
                    alpha=0.2,
                )

        if include_ideal_range:

            ax.fill_between(
                [0, 1],
                [-0.2]*2,
                [0.2]*2,
                color="gray",
                alpha=0.1,
            )
            ax.fill_between(
                [0, 1],
                [-0.1]*2,
                [0.1]*2,
                color="gray",
                alpha=0.2,
            )

    def plot(
        self,
        data_display: Union[DataDisplay, str],
        figure_alpha=1.0,
        line_width=3,
        legend_loc:Optional[str]=None,
        include_coverage_std:bool = False, 
        include_coverage_residual:bool = False,
        include_coverage_residual_std:bool = False,
        include_ideal_range: bool=True,
        reference_line_label="Reference Line",
        reference_line_style="k--",
        x_label="Confidence Interval of the Posterior Volume",
        y_label="Coverage fraction within posterior volume",
        residual_y_label="Coverage Fraction Residual",
        title=""
    ) -> tuple["fig", "ax"]:
        """
        Plot the coverage fraction and residuals if specified.

        Args: 
            data_display (Union[DataDisplay, str]): DataDisplay object or path to h5 file containing the data. If str, it will be loaded and requires the fields "coverage_fractions", "coverage_percentiles", and optionally "coverage_std".
            figure_alpha (float, optional): Opacity of parameter lines. Defaults to 1.0.
            line_width (int, optional): Width of parameter lines. Defaults to 3.
            legend_loc (str, optional): Location of the legend. Defaults to matplotlib specified. 
            include_coverage_std (bool, optional): Whether to include the standard deviation shading for coverage fractions . Defaults to False.
            include_coverage_residual (bool, optional): Whether to include the residual plot (coverage fraction - diagonal). Creates an additional subplot under the original plot. Defaults to False.
            include_coverage_residual_std (bool, optional): Whether to include the standard deviation shading for residuals. Defaults to False.
            include_ideal_range (bool, optional): Whether to include the ideal range shading (0.1/0.2 around the diagonal). Defaults to True.
            reference_line_label (str, optional): Label name for the diagonal ideal line. Defaults to "Reference Line".
            reference_line_style (str, optional): Line style for the reference line. Defaults to "k--".
            x_label (str, optional): y label. Defaults to "Confidence Interval of the Posterior Volume".
            y_label (str, optional): y label. Defaults to "Fraction of Lenses within Posterior Volume".
            residual_y_label (str, optional): y label for the residual plot. Defaults to "Coverage Fraction Residual".
            title (str, optional): plot title. Defaults to "NPE".

        """

        if not isinstance(data_display, DataDisplay):
            data_display = DataDisplay().from_h5(data_display, self.plot_name)


        percentile_array = data_display.coverage_percentiles / 100.0
        color_cycler = iter(plt.cycler("color", self.parameter_colors))
        line_style_cycler = iter(plt.cycler("line_style", self.line_cycle))

        # Plotting
        if include_coverage_residual:
            fig, subplots = plt.subplots(2, 1, figsize=(self.figure_size[0], self.figure_size[1]*1.2), height_ratios=[3, 1], sharex=True)
            ax = subplots[0]

            self._plot_residual(
                data_display, subplots[1], figure_alpha, line_width, reference_line_style, include_coverage_residual_std, include_ideal_range
            )
            subplots[1].set_ylabel(residual_y_label)
            subplots[1].set_xlabel(x_label)

        else:
            fig, ax = plt.subplots(1, 1, figsize=self.figure_size)
            ax.set_xlabel(x_label)

        # Iterate over the number of parameters in the model
        for i in range(self.n_parameters):
            color = next(color_cycler)["color"]
            line_style = next(line_style_cycler)["line_style"]
            ax.plot(
                percentile_array,
                data_display.coverage_fractions[:, i],
                alpha=figure_alpha,
                lw=line_width,
                linestyle=line_style,
                color=color,
                label=self.parameter_names[i],
            )
            if include_coverage_std:
                ax.fill_between(
                    percentile_array,
                    data_display.coverage_fractions[:, i] - data_display.coverage_std[:, i],
                    data_display.coverage_fractions[:, i] + data_display.coverage_std[:, i],
                    color=color,
                    alpha=0.2,
                )

        ax.plot(
            [0, 0.5, 1],
            [0, 0.5, 1],
            reference_line_style,
            lw=line_width,
            zorder=1000,
            label=reference_line_label,
        )

        if include_ideal_range:
            def add_clearance(ax, clearance=0.1, clearance_alpha=0.2):
                x_values = np.linspace(0, 1, 100)  # More points for smoother curves
                y_lower = np.maximum(0, x_values - clearance)  # Lower bound with clearance
                y_upper = np.minimum(1, x_values + clearance)  # Upper bound with clearance
                
                # Fill the area between the bounds
                ax.fill_between(
                x_values,
                y_lower,
                y_upper,
                color="gray",
                alpha=clearance_alpha,
                )
                
            add_clearance(ax, clearance=0.2, clearance_alpha=0.2)
            add_clearance(ax, clearance=0.1, clearance_alpha=0.1)
            

        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])

        # ax.text(-0.03, 0.93, "Under-confident", horizontalalignment="left")
        # ax.text(0.3, 0.05, "Overconfident", horizontalalignment="left")

        if legend_loc is not None:
            ax.legend(loc=legend_loc)
        else:
            ax.legend()

        ax.set_ylabel(y_label)
        ax.set_title(title)

        return fig, ax