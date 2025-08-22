import importlib
import inspect
from pathlib import Path
from deepdiagnostics.metrics.metric import Metric

# Void is included as a placeholder for empty metrics
def void(*args, **kwargs): 
    def void2(*args, **kwargs):
        return None
    return void2

Metrics = {"": void}
__all__ = []
for file in Path(__file__).parent.glob("*.py"):
    if file.name.startswith("__") or file.name == "metric.py":
        continue
    module = importlib.import_module(f"deepdiagnostics.metrics.{file.stem}")
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if  issubclass(obj, Metric) and obj != Metric:
            Metrics[obj.__name__] = obj
            globals()[obj.__name__] = obj
            __all__.append(obj.__name__)

if 'LocalTwoSampleTest' in Metrics:
    Metrics['LC2ST'] = Metrics['LocalTwoSampleTest']
    globals()['lc2st'] = Metrics['LocalTwoSampleTest']
    __all__.append('lc2st')