from sbi.analysis import sbc_rank_plot, run_sbc
from torch import tensor

from plots.plot import Display
from utils.config import get_item


class CDFRanks(Display):
    def __init__(
        self,
        model,
        data,
        save: bool,
        show: bool,
        out_dir: str | None = None,
        samples_per_inference=None,
        parameter_colors=None,
        parameter_labels=None,
    ):
        super().__init__(model, data, save, show, out_dir)

        self.num_samples = (
            samples_per_inference
            if samples_per_inference is not None
            else get_item(
                "metrics_common", "samples_per_inference", raise_exception=False
            )
        )
        self.colors = (
            parameter_colors
            if parameter_colors is not None
            else get_item("plots_common", "parameter_colors", raise_exception=False)
        )
        self.labels = (
            parameter_labels
            if parameter_labels is not None
            else get_item("plots_common", "parameter_labels", raise_exception=False)
        )

    def _plot_name(self):
        return "cdf_ranks.png"

    def _data_setup(self):
        thetas = tensor(self.data.get_theta_true())
        context = tensor(self.data.true_context())

        ranks, _ = run_sbc(
            thetas,
            context,
            self.model.posterior,
            num_posterior_samples=self.num_samples,
        )
        self.ranks = ranks

    def _plot_settings(self):
        pass

    def _plot(self):
        sbc_rank_plot(
            self.ranks,
            self.num_samples,
            plot_type="cdf",
            parameter_labels=self.labels,
            colors=self.colors,
        )
