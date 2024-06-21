from deepdiagnostics.metrics.all_sbc import AllSBC
from deepdiagnostics.metrics.coverage_fraction import CoverageFraction
from deepdiagnostics.metrics.local_two_sample import LocalTwoSampleTest

Metrics = {
    CoverageFraction.__name__: CoverageFraction, 
    AllSBC.__name__: AllSBC, 
    "LC2ST": LocalTwoSampleTest
}