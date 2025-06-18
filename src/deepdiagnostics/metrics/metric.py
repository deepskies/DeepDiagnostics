from typing import Any, Optional, Sequence
import json
import os

from deepdiagnostics.data import data
from deepdiagnostics.models import model
from deepdiagnostics.utils.config import get_item


class Metric:
    """
        These parameters are used for every metric calculated, and for plots that require new inference to be run. 
        Calculate a given metric. Save output to a json if out_dir and saving specified. 

        Args:
            model (deepdiagnostics.models.model): Model to calculate the metric for. Required. 
            data (deepdiagnostics.data.data): Data to test against. Required. 
            out_dir (Optional[str], optional): Directory to save a json (results.json) to. Defaults to None.
            save (bool, optional): Save the output to json. Defaults to True.
            use_progress_bar (Optional[bool], optional):Show a progress bar when iteratively performing inference. Defaults to None.
            samples_per_inference (Optional[int], optional) :Number of samples used in a single iteration of inference. Defaults to None.
            percentiles (Optional[Sequence[int]], optional): List of integer percentiles, for defining coverage regions. Defaults to None.
            number_simulations (Optional[int], optional):Number of different simulations to run. Often, this means that the number of inferences performed for a metric is samples_per_inference*number_simulations. Defaults to None.
    """
        
    def __init__(
            self, 
            model: model,
            data: data, 
            run_id: str,
            out_dir: Optional[str] = None, 
            save: bool=True,
            use_progress_bar: Optional[bool] = None,
            samples_per_inference: Optional[int] = None,
            percentiles: Optional[Sequence[int]] = None,
            number_simulations: Optional[int] = None,
              ) -> None:
        self.model = model
        self.data = data
        self.run_id = run_id

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

            with open(f"{self.out_dir.rstrip('/')}/{self.run_id}_diagnostic_metrics.json", "w+") as f:
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
