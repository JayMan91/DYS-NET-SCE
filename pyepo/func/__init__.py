#!/usr/bin/env python
# coding: utf-8
"""
Pytorch autograd function for end-to-end training
"""

from pyepo.func.spoplus import SPOPlus
from pyepo.func.dys import DYSOpt, DYSOpt_OTF
from pyepo.func.cvx import CVXOpt
from pyepo.func.sfge import SFGEOpt
from pyepo.func.blackbox import blackboxOpt, negativeIdentity
from pyepo.func.perturbed import perturbedOpt, perturbedFenchelYoung
from pyepo.func.contrastive import NCE, contrastiveMAP, contrastiveMAP_linear
# from pyepo.func.full_contrastive import SCE_Full, SCELinear_Full, SCELinearAlternative_Full
from pyepo.func.rank import listwiseLTR, pairwiseLTR, pointwiseLTR
from pyepo.func.cave import innerConeAlignedCosine

