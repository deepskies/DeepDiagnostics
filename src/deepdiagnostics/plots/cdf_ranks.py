from sbi.analysis import sbc_rank_plot, run_sbc
from torch import tensor

from typing import TYPE_CHECKING, Union
from deepdiagnostics.utils.utils import DataDisplay
from deepdiagnostics.plots.plot import Display

if TYPE_CHECKING:
    from matplotlib.figure import Figure as fig
    from matplotlib.axes import Axes as ax

class CDFRanks(Display):
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
        
        """
        Adaptation of :cite:p:`centero2020sbi`.
        A wrapper around `SBI <https://github.com/sbi-dev/sbi>`_'s sbc_rank_plot function. 
        `More information can be found here <https://sbi-dev.github.io/sbi/tutorial/13_diagnostics_simulation_based_calibration/#visual-inspection>`_
        Plots the ranks as a CDF plot for each theta parameter. 

        .. code-block:: python
        
            from deepdiagnostics.plots import CDFRanks 

            CDFRanks(model, data, save=False, show=True)()

        """
        
        super().__init__(model, data, run_id, save, show, out_dir, percentiles, use_progress_bar, samples_per_inference, number_simulations, parameter_names, parameter_colors, colorway)

    def plot_name(self):
        return "cdf_ranks.png"

    def _data_setup(self) -> DataDisplay:
        thetas = tensor(self.data.get_theta_true())
        context = tensor(self.data.true_context())

        ranks, _ = run_sbc(
            thetas, context, self.model.posterior, num_posterior_samples=self.samples_per_inference
        )
        display_data = DataDisplay(
            ranks=ranks
        )
        return display_data

    def plot_settings(self):
        pass

    def plot(self, data_display: Union[DataDisplay, str], **kwargs) -> tuple["fig", "ax"]:
        """
            Make the CDF Ranks plot

            Args:
                display (data_display, str): The data to plot, typically returned by the _data_setup. Must have the 'ranks' attribute.
        """
        if not isinstance(data_display, DataDisplay):
            data_display = DataDisplay().from_h5(data_display, self.plot_name)
            
        return sbc_rank_plot(
            data_display.ranks,
            self.samples_per_inference,
            plot_type="cdf",
            parameter_labels=self.parameter_names,
            colors=self.parameter_colors,
        )
