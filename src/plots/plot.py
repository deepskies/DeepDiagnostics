import os
from typing import Optional, Sequence
import matplotlib.pyplot as plt
from matplotlib import rcParams

from utils.config import get_item


class Display:
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
        colorway: Optional[str]=None
    ): 
        
        self.save = save
        self.show = show
        self.data = data

        self.use_progress_bar = use_progress_bar if use_progress_bar is not None else get_item("metrics_common", "use_progress_bar", raise_exception=False)
        self.samples_per_inference = samples_per_inference if samples_per_inference is not None else get_item("metrics_common", "samples_per_inference", raise_exception=False)
        self.percentiles = percentiles if percentiles is not None else get_item("metrics_common", "percentiles", raise_exception=False)
        self.number_simulations = number_simulations if number_simulations is not None else get_item("metrics_common", "number_simulations", raise_exception=False)

        self.parameter_names = parameter_names if parameter_names is not None else get_item("plots_common", "parameter_labels", raise_exception=False)
        self.parameter_colors = parameter_colors if parameter_colors is not None else get_item("plots_common", "parameter_colors", raise_exception=False)
        self.colorway =  colorway if colorway is not None else get_item(
                "plots_common", "default_colorway", raise_exception=False
            )
        
        if save: 
            self.out_dir = out_dir if out_dir is not None else get_item("common", "out_dir", raise_exception=False)

        if self.out_dir is not None:
            if not os.path.exists(os.path.dirname(self.out_dir)):
                os.makedirs(os.path.dirname(self.out_dir))

        self.model = model
        self._common_settings()
        self.plot_name = self._plot_name()

    def _plot_name(self):
        raise NotImplementedError

    def _data_setup(self):
        # Set all the vars used for the plot
        raise NotImplementedError

    def _plot(self, **kwrgs):
        # Make the plot object with plt.
        raise NotImplementedError

    def _common_settings(self):
        rcParams["axes.spines.right"] = bool(
            get_item("plots_common", "axis_spines", raise_exception=False)
        )
        rcParams["axes.spines.top"] = bool(
            get_item("plots_common", "axis_spines", raise_exception=False)
        )
        # Style
        tight_layout = bool(
            get_item("plots_common", "tight_layout", raise_exception=False)
        )
        if tight_layout:
            plt.tight_layout()
        plot_style = get_item("plots_common", "plot_style", raise_exception=False)
        plt.style.use(plot_style)

        self.figure_size = tuple(get_item("plots_common", "figure_size", raise_exception=False))

    def _finish(self):
        assert (
            os.path.splitext(self.plot_name)[-1] != ""
        ), f"plot name, {self.plot_name}, is malformed. Please supply a name with an extension."

        if self.show:
            plt.show()

        if self.save:
            plt.savefig(f"{self.out_dir.rstrip('/')}/{self.plot_name}")
            plt.cla()

    def __call__(self, **plot_args) -> None:
        try: 
            self._data_setup()
        except NotImplementedError: 
            pass 
        
        self._plot(**plot_args)
        self._finish()
