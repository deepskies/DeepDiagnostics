import os
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib import rcParams

from utils.config import get_item

class Display: 
    def __init__(self, model, data, save:bool, show:bool, out_dir:Optional[str]=None): 
        self.save = save
        self.show = show 
        self.data = data 

        self.out_path = None
        if (out_dir is None) and self.save: 
            self.out_path = get_item("common", "out_dir", raise_exception=False)
            
        elif self.save and (out_dir is not None): 
            self.out_path = out_dir

        if self.out_path is not None: 
            if not os.path.exists(os.path.dirname(self.out_path)): 
                os.makedirs(os.path.dirname(self.out_path))
                    
        self.model = model
        self._common_settings()
        self._plot_settings()
        self.plot_name = self._plot_name()

    def _plot_name(self): 
        raise NotImplementedError

    def _data_setup(self): 
        # Set all the vars used for the plot
        raise NotImplementedError
    
    def _plot_settings(self): 
        # TODO Pull fom a config for specific plots 
        raise NotImplementedError
    
    def _plot(self, **kwrgs):
        # Make the plot object with plt.
        raise NotImplementedError 
    
    def _common_settings(self): 

        rcParams["axes.spines.right"] = bool(get_item('plots_common', 'axis_spines', raise_exception=False))
        rcParams["axes.spines.top"] = bool(get_item('plots_common','axis_spines', raise_exception=False))

        # Style 
        self.colorway = get_item('plots_common', "default_colorway", raise_exception=False)
        tight_layout = bool(get_item('plots_common','tight_layout', raise_exception=False))
        if tight_layout: 
            plt.tight_layout() 
        plot_style = get_item('plots_common','plot_style', raise_exception=False)
        plt.style.use(plot_style)

    def _finish(self):
        assert os.path.splitext(self.plot_name)[-1] != '', f"plot name, {self.plot_name}, is malformed. Please supply a name with an extension."
        if self.save:
            plt.savefig(f"{self.out_path.rstrip('/')}/{self.plot_name}")
        if self.show:
            plt.show()

        plt.cla()
 
    def __call__(self, **plot_args) -> None:
        self._data_setup()
        self._plot(**plot_args)
        self._finish()