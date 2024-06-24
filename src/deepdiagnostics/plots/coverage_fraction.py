import numpy as np
import matplotlib.pyplot as plt

from deepdiagnostics.metrics.coverage_fraction import CoverageFraction as coverage_fraction_metric
from deepdiagnostics.plots.plot import Display
from deepdiagnostics.utils.config import get_item


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
            
        self.n_parameters = len(self.parameter_names)
        self.line_cycle =  tuple(get_item("plots_common", "line_style_cycle", raise_exception=False))

    def plot_name(self):
        return "coverage_fraction.png"

    def _data_setup(self):
        _, coverage = coverage_fraction_metric(
            self.model, self.data, out_dir=None
        ).calculate()
        self.coverage_fractions = coverage

    def plot(
        self,
        figure_alpha=1.0,
        line_width=3,
        legend_loc="lower right",
        reference_line_label="Reference Line",
        reference_line_style="k--",
        x_label="Confidence Interval of the Posterior Volume",
        y_label="Fraction of Lenses within Posterior Volume",
        title="NPE"):
        """
        Args:
            figure_alpha (float, optional): Opacity of parameter lines. Defaults to 1.0.
            line_width (int, optional): Width of parameter lines. Defaults to 3.
            legend_loc (str, optional): Location of the legend, str based on `matplotlib <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html>`_. Defaults to "lower right".
            reference_line_label (str, optional): Label name for the diagonal ideal line. Defaults to "Reference Line".
            reference_line_style (str, optional): Line style for the reference line. Defaults to "k--".
            x_label (str, optional): y label. Defaults to "Confidence Interval of the Posterior Volume".
            y_label (str, optional): y label. Defaults to "Fraction of Lenses within Posterior Volume".
            title (str, optional): plot title. Defaults to "NPE".
        """

        n_steps = self.coverage_fractions.shape[0]
        percentile_array = np.linspace(0, 1, n_steps)
        color_cycler = iter(plt.cycler("color", self.parameter_colors))
        line_style_cycler = iter(plt.cycler("line_style", self.line_cycle))

        # Plotting
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size)

        # Iterate over the number of parameters in the model
        for i in range(self.n_parameters):
            color = next(color_cycler)["color"]
            line_style = next(line_style_cycler)["line_style"]

            ax.plot(
                percentile_array,
                self.coverage_fractions[:, i],
                alpha=figure_alpha,
                lw=line_width,
                linestyle=line_style,
                color=color,
                label=self.parameter_names[i],
            )

        ax.plot(
            [0, 0.5, 1],
            [0, 0.5, 1],
            reference_line_style,
            lw=line_width,
            zorder=1000,
            label=reference_line_label,
        )

        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])

        ax.text(0.03, 0.93, "Under-confident", horizontalalignment="left")
        ax.text(0.3, 0.05, "Overconfident", horizontalalignment="left")

        ax.legend(loc=legend_loc)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
