from typing import Any
from torch import tensor
from sbi.analysis import run_sbc, check_sbc

from metrics.metric import Metric
from utils.config import get_item


class AllSBC(Metric):
    def __init__(
        self,
        model: Any,
        data: Any,
        out_dir: str | None = None,
        samples_per_inference=None,
    ) -> None:
        super().__init__(model, data, out_dir)

        if samples_per_inference is None:
            self.samples_per_inference = get_item(
                "metrics_common", "samples_per_inference", raise_exception=False
            )
        else:
            self.samples_per_inference = samples_per_inference

    def _collect_data_params(self):
        self.thetas = tensor(self.data.theta_true())
        self.y_true = tensor(self.data.x_true())

    def calculate(self):
        ranks, dap_samples = run_sbc(
            self.thetas,
            self.y_true,
            self.model.posterior,
            num_posterior_samples=self.samples_per_inference,
        )

        sbc_stats = check_sbc(
            ranks,
            self.thetas,
            dap_samples,
            num_posterior_samples=self.samples_per_inference,
        )
        self.output = sbc_stats
        return sbc_stats

    def __call__(self, **kwds: Any) -> Any:
        self._collect_data_params()
        self.calculate()
        self._finish()
