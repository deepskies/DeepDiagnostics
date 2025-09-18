from deepdiagnostics.metrics.all_sbc import AllSBC
from deepdiagnostics.metrics.coverage_fraction import CoverageFraction
from deepdiagnostics.metrics.local_two_sample import LocalTwoSampleTest as LC2ST
from deepdiagnostics.metrics.under_cdf_parity import CDFParityAreaUnderCurve as CDFParityAreaUnderCurve

def void(*args, **kwargs): 
    def void2(*args, **kwargs):
        return None
    return void2

Metrics = {
    "": void, 
    CoverageFraction.__name__: CoverageFraction, 
    AllSBC.__name__: AllSBC, 
    "LC2ST": LC2ST
}
