from typing import Any, Optional, Sequence
import json
import os

from data import data
from models import model
from utils.config import get_item


class Metric:
    def __init__(
            self, 
            model: model,
            data: data, 
            out_dir: Optional[str] = None, 
            save: bool=True,
            use_progress_bar: Optional[bool] = None,
            samples_per_inference: Optional[int] = None,
            percentiles: Optional[Sequence[int]] = None,
            number_simulations: Optional[int] = None,
              ) -> None:
        self.model = model
        self.data = data

        if save: 
            self.out_dir = out_dir if out_dir is not None else get_item("common", "out_dir", raise_exception=False)

        self.output = None

        self.use_progress_bar = use_progress_bar if use_progress_bar is not None else get_item("metrics_common", "use_progress_bar", raise_exception=False)
        self.samples_per_inference = samples_per_inference if samples_per_inference is not None else get_item("metrics_common", "samples_per_inference", raise_exception=False)
        self.percentiles = percentiles if percentiles is not None else get_item("metrics_common", "percentiles", raise_exception=False)
        self.number_simulations = number_simulations if number_simulations is not None else get_item("metrics_common", "number_simulations", raise_exception=False)

    def _collect_data_params():
        raise NotImplementedError

    def _run_model_inference():
        raise NotImplementedError

    def calculate(self):
        raise NotImplementedError

    def _finish(self):
        assert (
            self.output is not None
        ), "Calculation has not been completed, have you run Metric.calculate?"

        if self.out_dir is not None:
            if not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir)

            with open(f"{self.out_dir.rstrip('/')}/diagnostic_metrics.json", "w+") as f:
                try: 
                    data = json.load(f)
                except json.decoder.JSONDecodeError: 
                    data = {}

                data.update(self.output)
                json.dump(data, f, ensure_ascii=True)
                f.close()

    def __call__(self, **kwds: Any) -> Any:
        
        try: 
            self._collect_data_params()
        except NotImplementedError: 
            pass 
        try: 
            self._run_model_inference() 
        except NotImplementedError: 
            pass 

        self.calculate(kwds)
        self._finish()
