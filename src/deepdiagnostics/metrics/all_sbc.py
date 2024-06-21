from typing import Any, Optional, Sequence
from torch import tensor
from sbi.analysis import run_sbc, check_sbc

from deepdiagnostics.metrics.metric import Metric


class AllSBC(Metric):
    def __init__(
        self,
        model: Any,
        data: Any,
        out_dir: str | None = None,
        save: bool=True,
        use_progress_bar: Optional[bool] = None,
        samples_per_inference: Optional[int] = None,
        percentiles: Optional[Sequence[int]] = None,
        number_simulations: Optional[int] = None,
    ) -> None:
        
        super().__init__(model, data, out_dir,
            save,
            use_progress_bar,
            samples_per_inference,
            percentiles,
            number_simulations)

    def _collect_data_params(self):
        self.thetas = tensor(self.data.get_theta_true())
        self.context = tensor(self.data.true_context())

    def calculate(self):
        ranks, dap_samples = run_sbc(
            self.thetas,
            self.context,
            self.model.posterior,
            num_posterior_samples=self.samples_per_inference,
        )

        sbc_stats = check_sbc(
            ranks,
            self.thetas,
            dap_samples,
            num_posterior_samples=self.samples_per_inference,
        )
        self.output = {key: value.numpy().tolist() for key, value in sbc_stats.items()}
        return sbc_stats

    def __call__(self, **kwds: Any) -> Any:
        self._collect_data_params()
        self.calculate()
        self._finish()
