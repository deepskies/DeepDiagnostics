import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm

from metrics.coverage_fraction import CoverageFraction as coverage_fraction_metric
from plots.plot import Display
from utils.config import get_item


class CoverageFraction(Display):
    def __init__(
        self,
        model,
        data,
        save: bool,
        show: bool,
        out_dir: str | None = None,
        parameter_labels=None,
        figure_size=None,
        line_styles=None,
    ):
        super().__init__(model, data, save, show, out_dir)

        self.labels = (
            parameter_labels
            if parameter_labels is not None
            else get_item("plots_common", "parameter_labels", raise_exception=False)
        )
        self.n_parameters = len(self.labels)
        self.figure_size = (
            figure_size
            if figure_size is not None
            else tuple(get_item("plots_common", "figure_size", raise_exception=False))
        )
        self.line_cycle = (
            line_styles
            if line_styles is not None
            else tuple(
                get_item("plots_common", "line_style_cycle", raise_exception=False)
            )
        )

    def _plot_name(self):
        return "coverage_fraction.png"

    def _data_setup(self):
        _, coverage = coverage_fraction_metric(
            self.model, self.data, out_dir=None
        ).calculate()
        self.coverage_fractions = coverage

    def _plot_settings(self):
        pass

    def _plot(
        self,
        figure_alpha=1.0,
        line_width=3,
        legend_loc="lower right",
        reference_line_label="Reference Line",
        reference_line_style="k--",
        x_label="Confidence Interval of the Posterior Volume",
        y_label="Fraction of Lenses within Posterior Volume",
        title="NPE",
    ):
        n_steps = self.coverage_fractions.shape[0]
        percentile_array = np.linspace(0, 1, n_steps)
        color_cycler = iter(plt.cycler("color", cm.get_cmap(self.colorway).colors))
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
                label=self.labels[i],
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
