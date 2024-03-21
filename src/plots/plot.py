import os
from typing import Any, Optional
import matplotlib.pyplot as plt
from matplotlib import rcParams

from src.data.data import Data
from src.utils.config import get_item, get_section

class Display: 
    def __init__(self, data:Data, save:bool, show:bool, out_path:Optional[str]): 
        self.save = save
        self.show = show 
        self.out_path = out_path.rstrip("/")
        if self.save: 
            assert self.out_path is not None, "out_path required to save files." 

        if not os.path.exists(os.path.dirname(out_path)): 
            os.makedirs(os.path.dirname(out_path))

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
    
    def _plot(self, *args, **kwrgs):
        # Make the plot object with plt.
        raise NotImplementedError 
    
    def _common_settings(self): 
        # TODO Pull from a common config 
        rcParams["axes.spines.right"] = False
        rcParams["axes.spines.top"] = False 

        # Style 
        self.colorway = ""
        tight_layout = ""

        if tight_layout: 
            plt.tight_layout() 

    def _finish(self):
        assert os.path.splitext(self.plot_name)[-1] != '', f"plot name, {self.plot_name}, is malformed. Please supply a name with an extension."
        if self.save:
            plt.savefig(f"{self.out_path}/{self.plot_name}")
        if self.plot:
            plt.show()
 
    def __call__(self) -> None:
        self._plot()
        self._finish()