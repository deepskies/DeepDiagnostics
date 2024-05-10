from metrics.all_sbc import AllSBC
from metrics.coverage_fraction import CoverageFraction
from metrics.local_two_sample import LocalTwoSampleTest


_all = [CoverageFraction, AllSBC, LocalTwoSampleTest]
Metrics = {m.__name__: m for m in _all}