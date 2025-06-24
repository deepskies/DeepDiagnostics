from typing import Any, Sequence
from torch import tensor
from sbi.analysis import run_sbc, check_sbc

from deepdiagnostics.metrics.metric import Metric


class AllSBC(Metric):
    """
        Calculate SBC diagnostics metrics and add them to the output. 
        Adapted from :cite:p:`centero2020sbi`. 
        More information about specific metrics can be found `here <https://sbi-dev.github.io/sbi/tutorial/13_diagnostics_simulation_based_calibration/#a-shifted-posterior-mean>`_.

        .. code-block:: python 

            from deepdiagnostics.metrics import AllSBC

            metrics = AllSBC(model, data, save=False)()
            metrics = metrics.output
    """
    def __init__(
        self,
        model,
        data,
        run_id,
        out_dir= None,
        save = True,
        use_progress_bar = None,
        samples_per_inference = None,
        percentiles = None,
        number_simulations = None,
    ) -> None:
        
        super().__init__(model, data, run_id, out_dir,
            save,
            use_progress_bar,
            samples_per_inference,
            percentiles,
            number_simulations)

    def _collect_data_params(self):
        self.thetas = tensor(self.data.get_theta_true())
        self.context = tensor(self.data.true_context())

    def calculate(self) -> dict[str, Sequence]:
        """
        Calculate all SBC diagnostic metrics 

        Returns:
            dict[str, Sequence]: Dictionary with all calculations, labeled by their name. 
        """
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
