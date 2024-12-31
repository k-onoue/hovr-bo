import torch
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler

from ._base_sampler import RelativeSampler
from ._utils import get_acquisition_function


class DAPL2Sampler(RelativeSampler):
    pass


class DAPARTLSampler(RelativeSampler):
    pass