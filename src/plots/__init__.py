from plots.cdf_ranks import CDFRanks
from plots.coverage_fraction import CoverageFraction
from plots.ranks import Ranks
from plots.tarp import TARP
from plots.local_two_sample import LocalTwoSampleTest
from plots.predictive_posterior_check import PPC

Plots = {
    CDFRanks.__name__: CDFRanks,
    CoverageFraction.__name__: CoverageFraction,
    Ranks.__name__: Ranks,
    TARP.__name__: TARP,
    "LC2ST": LocalTwoSampleTest, 
    PPC.__name__: PPC
}
