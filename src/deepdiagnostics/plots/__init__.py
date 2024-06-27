from deepdiagnostics.plots.cdf_ranks import CDFRanks
from deepdiagnostics.plots.coverage_fraction import CoverageFraction
from deepdiagnostics.plots.ranks import Ranks
from deepdiagnostics.plots.tarp import TARP
from deepdiagnostics.plots.local_two_sample import LocalTwoSampleTest as LC2ST
from deepdiagnostics.plots.predictive_posterior_check import PPC
from deepdiagnostics.plots.parity import Parity
from deepdiagnostics.plots.predictive_prior_check import PriorPC


def void(*args, **kwargs): 
    def void2(*args, **kwargs):
        return None
    return void2

Plots = {
    "": void,
    CDFRanks.__name__: CDFRanks,
    CoverageFraction.__name__: CoverageFraction,
    Ranks.__name__: Ranks,
    TARP.__name__: TARP,
    "LC2ST": LC2ST, 
    PPC.__name__: PPC, 
    "Parity": Parity,
    PriorPC.__name__: PriorPC
}
