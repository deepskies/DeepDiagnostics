from typing import Union, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from scipy.stats import ecdf

from deepdiagnostics.plots.plot import Display
from deepdiagnostics.utils.config import get_item
from deepdiagnostics.utils.utils import DataDisplay
