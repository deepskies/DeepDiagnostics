from typing import Union, TYPE_CHECKING
from sbi.analysis import sbc_rank_plot, run_sbc
from torch import tensor

from deepdiagnostics.plots.plot import Display
from deepdiagnostics.utils.utils import DataDisplay

if TYPE_CHECKING:
    from matplotlib.figure import Figure as fig
    from matplotlib.axes import Axes as ax

class Ranks(Display):
    """
                
        Adaptation of :cite:p:`centero2020sbi`.

        A wrapper around `SBI <https://github.com/sbi-dev/sbi>`_'s sbc_rank_plot function. 
        `More information can be found here <https://sbi-dev.github.io/sbi/tutorial/13_diagnostics_simulation_based_calibration/#visual-inspection>`_
        Plots the histogram of each theta parameter's rank. 

        .. code-block:: python
        
            from deepdiagnostics.plots import Ranks 

            Ranks(model, data, save=False, show=True)(num_bins=25)
    """
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
    
        super().__init__(model, data, run_id, save, show, out_dir, percentiles, use_progress_bar, samples_per_inference, number_simulations, parameter_names, parameter_colors, colorway)
            
    def plot_name(self):
        return "ranks.png"

    def _data_setup(self, **kwargs) -> DataDisplay:
        thetas = tensor(self.data.get_theta_true())
        context = tensor(self.data.true_context())
        ranks, _ = run_sbc(
            thetas, context, self.model.posterior, num_posterior_samples=self.samples_per_inference
        )
        return DataDisplay(ranks=ranks)

    def plot(self, data_display: Union[DataDisplay, dict], num_bins:int=20) -> tuple["fig", "ax"]:
        """
        Args:
            num_bins (int): Number of histogram bins. Defaults to 20.
        """
        if not isinstance(data_display, DataDisplay):
            data_display = DataDisplay().from_h5(data_display, self.plot_name)

        return sbc_rank_plot(
            ranks=data_display.ranks,
            num_posterior_samples=self.samples_per_inference,
            plot_type="hist",
            num_bins=num_bins,
            parameter_labels=self.parameter_names,
            colors=self.parameter_colors,
        )
