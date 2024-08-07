from sbi.analysis import sbc_rank_plot, run_sbc
from torch import tensor

from deepdiagnostics.plots.plot import Display


class CDFRanks(Display):
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

    def plot_name(self):
        return "cdf_ranks.png"

    def _data_setup(self):
        thetas = tensor(self.data.get_theta_true())
        context = tensor(self.data.true_context())

        ranks, _ = run_sbc(
            thetas, context, self.model.posterior, num_posterior_samples=self.samples_per_inference
        )
        self.ranks = ranks

    def plot_settings(self):
        pass

    def plot(self):
        """
            Make the CDF Ranks plot
        """
        sbc_rank_plot(
            self.ranks,
            self.samples_per_inference,
            plot_type="cdf",
            parameter_labels=self.parameter_names,
            colors=self.parameter_colors,
        )
