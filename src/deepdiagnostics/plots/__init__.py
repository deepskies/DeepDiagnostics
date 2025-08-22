import importlib
import inspect
from pathlib import Path

from deepdiagnostics.plots.plot import Display

# Void is included as a placeholder for empty metrics
def void(*args, **kwargs): 
    def void2(*args, **kwargs):
        return None
    return void2

Plots = {"": void}
__all__ = []
for file in Path(__file__).parent.glob("*.py"):
    if file.name.startswith("__") or file.name == "plot.py":
        continue
    module = importlib.import_module(f"deepdiagnostics.plots.{file.stem}")
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if  issubclass(obj, Display) and obj != Display:
            Plots[obj.__name__] = obj
            globals()[obj.__name__] = obj
            __all__.append(obj.__name__)

if 'LocalTwoSampleTest' in Plots:
    Plots['LC2ST'] = Plots['LocalTwoSampleTest']
    globals()['lc2st'] = Plots['LocalTwoSampleTest']
    __all__.append('lc2st')
