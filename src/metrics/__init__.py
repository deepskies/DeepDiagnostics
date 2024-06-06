from metrics.all_sbc import AllSBC
from metrics.coverage_fraction import CoverageFraction
from metrics.local_two_sample import LocalTwoSampleTest

Metrics = {
    CoverageFraction.__name__: CoverageFraction, 
    AllSBC.__name__: AllSBC, 
    "LC2ST": LocalTwoSampleTest
}
