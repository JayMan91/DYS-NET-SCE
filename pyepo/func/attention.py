#!/usr/bin/env python
# coding: utf-8
"""
Noise contrastive estimation loss function
"""

import numpy as np
import torch

from pyepo import EPO
from pyepo.func.abcmodule import optModule
from pyepo.data.dataset import optDataset
from pyepo.func.utlis import _solveWithObj4Par, _solve_in_pass, _cache_in_pass




