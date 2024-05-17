from sbi.analysis import sbc_rank_plot, run_sbc
from torch import tensor

from plots.plot import Display
from utils.config import get_item


class Ranks(Display):
    def __init__(self, model, data, save: bool, show: bool, out_dir: str | None = None):
        super().__init__(model, data, save, show, out_dir)

    def _plot_name(self):
        return "ranks.png"

    def _data_setup(self):
        thetas = tensor(self.data.theta_true())
        y_true = tensor(self.data.x_true())
        self.num_samples = get_item(
            "metrics_common", "samples_per_inference", raise_exception=False
        )

        ranks, _ = run_sbc(
            thetas, y_true, self.model.posterior, num_posterior_samples=self.num_samples
        )
        self.ranks = ranks

    def _plot_settings(self):
        self.colors = get_item(
            "plots_common", "parameter_colors", raise_exception=False
        )
        self.labels = get_item(
            "plots_common", "parameter_labels", raise_exception=False
        )

    def _plot(self, num_bins=None):
        sbc_rank_plot(
            ranks=self.ranks,
            num_posterior_samples=self.num_samples,
            plot_type="hist",
            num_bins=num_bins,
            parameter_labels=self.labels,
            colors=self.colors,
        )
