import os
from typing import Any, Optional
import matplotlib.pyplot as plt
from matplotlib import rcParams

from utils.config import get_item, get_section

class Display: 
    def __init__(self, model, data, save:bool, show:bool, out_path:Optional[str]): 
        self.save = save
        self.show = show 
        self.out_path = out_path.rstrip("/")
        if self.save: 
            assert self.out_path is not None, "out_path required to save files." 

            if not os.path.exists(os.path.dirname(out_path)): 
                os.makedirs(os.path.dirname(out_path))
                
        self.model = model
        self._data_setup(data)
        self._common_settings()
        self._plot_settings()
        self.plot_name = self._plot_name()

    def _plot_name(self): 
        raise NotImplementedError

    def _data_setup(self, data): 
        # Set all the vars used for the plot
        raise NotImplementedError
    
    def _plot_settings(self): 
        # TODO Pull fom a config for specific plots 
        raise NotImplementedError
    
    def _plot(self, **kwrgs):
        # Make the plot object with plt.
        raise NotImplementedError 
    
    def _common_settings(self): 
        plot_common = get_section("plot_common", raise_exception=False)
        rcParams["axes.spines.right"] = bool(plot_common['axis_spines'])
        rcParams["axes.spines.top"] = bool(plot_common['axis_spines'])

        # Style 
        self.colorway = plot_common["colorway"]
        tight_layout = bool(plot_common['tight_layout'])
        if tight_layout: 
            plt.tight_layout() 
        plot_style = plot_common['plot_style']
        plt.style.use(plot_style)

    def _finish(self):
        assert os.path.splitext(self.plot_name)[-1] != '', f"plot name, {self.plot_name}, is malformed. Please supply a name with an extension."
        if self.save:
            plt.savefig(f"{self.out_path}/{self.plot_name}")
        if self.plot:
            plt.show()
 
    def __call__(self, **kwargs) -> None:
        self._plot(**kwargs)
        self._finish()