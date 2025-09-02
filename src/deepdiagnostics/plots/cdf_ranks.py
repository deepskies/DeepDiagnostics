from sbi.analysis import sbc_rank_plot, run_sbc
from torch import tensor
import torch
from tqdm import tqdm

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
        thetas = tensor(self.data.thetas)
        context = tensor(self.data.simulator_outcome)

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
    
class HierarchyCDFRanks(CDFRanks):
    def __init__(self, model, data, global_samples: bool = True, **kwargs):
        super().__init__(model, data, **kwargs)
        self.global_samples = bool(global_samples)

    def plot_name(self, **kwargs) -> str:
        gs = kwargs.pop("global_samples", self.global_samples)
        if gs:
            return "global_cdf_ranks.png"
        else:   
            return "local_cdf_ranks.png"
    
    def _data_setup(self, **kwargs) -> DataDisplay:
        gs = kwargs.pop("global_samples", self.global_samples)
        gs = bool(gs)

        # support attribute or callable access
        x = self.data.simulator_outcome() if callable(self.data.simulator_outcome) else self.data.simulator_outcome
        thetas = self.data.thetas() if callable(self.data.thetas) else self.data.thetas

        # sample hierarchical posterior
        posterior_samples = self.model.sample_posterior(self.samples_per_inference, x, global_samples=gs)

        print(f"Posterior samples shape: {posterior_samples.shape}") 

        n_params = posterior_samples.shape[-1]

        print(f"number of parameters: {n_params}")

        reduce_1d_fn = [eval(f"lambda theta, x: theta[:, {i}]") for i in range(n_params)]

        # pick global or local targets
        y_true = thetas[1] if gs else thetas[0]

        print(f"y_true shape: {y_true.shape}")

        # flatten if local samples such that [200, 25, 1000, 1] -> [200*25, 1000, 1]
        if not gs:
            posterior_samples = posterior_samples.reshape(-1, posterior_samples.shape[-2], posterior_samples.shape[-1])
            y_true = y_true.reshape(-1, y_true.shape[-1])
            print(f"Flattened posterior samples shape: {posterior_samples.shape}")
            print(f"Flattened y_true shape: {y_true.shape}")


        n_sbc_runs = posterior_samples.shape[0]
        
        # flatten x
        x = x.reshape(-1, x.shape[-1])
        print(f"Context shape: {x.shape}")


        ranks = torch.zeros((n_sbc_runs, len(reduce_1d_fn)))

        # calculate ranks
        for sbc_idx, (y_true0, x0) in tqdm(
                enumerate(zip(y_true, x, strict=False)),
                total=n_sbc_runs,
                desc=f"Calculating ranks for {n_sbc_runs} sbc samples.",
        ):
            for dim_idx, reduce_fn in enumerate(reduce_1d_fn):
                # rank posterior samples against true parameter, reduced to 1D.
                ranks[sbc_idx, dim_idx] = (
                    (reduce_fn(posterior_samples[sbc_idx, :, :], x0) < reduce_fn(y_true0.unsqueeze(0),
                                                                                    x0)).sum().item()
                )
        display_data = DataDisplay(
            ranks=ranks
        )
        return display_data