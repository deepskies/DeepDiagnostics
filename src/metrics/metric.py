from typing import Any, Optional
import json
import os

from data import data
from models import model


class Metric:
    def __init__(self, model: model, data: data, out_dir: Optional[str] = None) -> None:
        self.model = model
        self.data = data

        self.out_dir = out_dir
        self.output = None

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
            if not os.path.exists(os.path.dirname(self.out_dir)):
                os.makedirs(os.path.dirname(self.out_dir))

            with open(self.out_dir) as f:
                data = json.load(f)
                data.update(self.output)
                json.dump(data, f, ensure_ascii=True)
                f.close()

    def __call__(self, **kwds: Any) -> Any:
        self._collect_data_params()
        self._run_model_inference()
        self.calculate(kwds)
        self._finish()
