from typing import Optional, Sequence
from sbi.analysis import sbc_rank_plot, run_sbc
from torch import tensor

from deepdiagnostics.plots.plot import Display


class CDFRanks(Display):
    def __init__(
        self, 
        model, 
        data, 
        save:bool, 
        show:bool, 
        out_dir:Optional[str]=None, 
        percentiles: Optional[Sequence] = None, 
        use_progress_bar: Optional[bool] = None,
        samples_per_inference: Optional[int] = None,
        number_simulations: Optional[int] = None,
        parameter_names: Optional[Sequence] = None, 
        parameter_colors: Optional[Sequence]= None, 
        colorway: Optional[str]=None):
        
        """
        Adaptation of :cite:p:`centero2020sbi`.
        A wrapper around `SBI <https://github.com/sbi-dev/sbi>`_'s sbc_rank_plot function. 
        `More information can be found here <https://sbi-dev.github.io/sbi/tutorial/13_diagnostics_simulation_based_calibration/#visual-inspection>`_
        Plots the ranks as a CDF plot for each theta parameter. 

        .. code-block:: python
        
            from deepdiagnostics.plots import CDFRanks 

            CDFRanks(model, data, save=False, show=True)()

        """
        
        super().__init__(model, data, save, show, out_dir, percentiles, use_progress_bar, samples_per_inference, number_simulations, parameter_names, parameter_colors, colorway)

    def _plot_name(self):
        return "cdf_ranks.png"

    def _data_setup(self):
        thetas = tensor(self.data.get_theta_true())
        context = tensor(self.data.true_context())

        ranks, _ = run_sbc(
            thetas, context, self.model.posterior, num_posterior_samples=self.samples_per_inference
        )
        self.ranks = ranks

    def _plot_settings(self):
        pass

    def _plot(self):
        """
        """
        sbc_rank_plot(
            self.ranks,
            self.samples_per_inference,
            plot_type="cdf",
            parameter_labels=self.parameter_names,
            colors=self.parameter_colors,
        )
