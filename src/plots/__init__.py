from plots.cdf_ranks import CDFRanks
from plots.coverage_fraction import CoverageFraction
from plots.ranks import Ranks
from plots.local_two_sample import LocalTwoSampleTest
from plots.tarp import TARP

_all = [CoverageFraction, CDFRanks, Ranks, LocalTwoSampleTest, TARP]
Metrics = {m.__name__: m for m in _all}