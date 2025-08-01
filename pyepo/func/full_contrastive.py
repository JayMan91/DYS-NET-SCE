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



class SCE_Full(optModule):
    """
    An autograd module for Maximum A Posterior contrastive estimation as
    surrogate loss functions, which is a efficient self-contrastive algorithm.

    For the MAP, the cost vector needs to be predicted from contextual data and
    maximizes the separation of the probability of the optimal solution.

    Thus, allows us to design an algorithm based on stochastic gradient descent.

    Reference: <https://www.ijcai.org/proceedings/2021/390>
    """

    def __init__(self, optmodel, processes=1):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (None/optDataset): the training data, usually this is simply the training set
        """
        super().__init__(optmodel, processes,)


    def forward(self, pred_cost, true_cost, true_sol, reduction="mean"):
        """
        Forward pass
        """
        # get device
        device = pred_cost.device
        # convert tensor
        cp = pred_cost.detach().to("cpu").numpy()
        c = true_cost.detach().to("cpu").numpy()
        sol, _ = _solve_in_pass(cp, self.optmodel, self.processes, self.pool)
        sol = np.array(sol)
        pred_sol = torch.FloatTensor(sol).to(device)


        if self.optmodel.modelSense == EPO.MINIMIZE:
            loss = ((pred_cost )*(true_sol - pred_sol)).sum(axis =1 )
        if self.optmodel.modelSense == EPO.MAXIMIZE:
            loss = -((pred_cost )*(true_sol - pred_sol)).sum(axis =1 )
        # reduction
    
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
        elif reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(reduction))
        return loss

class SCELinear_Full(optModule):
    """
    An autograd module for Maximum A Posterior contrastive estimation as
    surrogate loss functions, which is a efficient self-contrastive algorithm.

    For the MAP, the cost vector needs to be predicted from contextual data and
    maximizes the separation of the probability of the optimal solution.

    Thus, allows us to design an algorithm based on stochastic gradient descent.

    Reference: <https://www.ijcai.org/proceedings/2021/390>
    """

    def __init__(self, optmodel, processes=1):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (None/optDataset): the training data, usually this is simply the training set
        """
        super().__init__(optmodel, processes,)


    def forward(self, pred_cost, true_cost, true_sol, reduction="mean"):
        """
        Forward pass
        """
        # get device
        device = pred_cost.device
        # convert tensor
        cp = pred_cost.detach().to("cpu").numpy()
        c = true_cost.detach().to("cpu").numpy()
        sol, _ = _solve_in_pass(cp, self.optmodel, self.processes, self.pool)
        sol = np.array(sol)
        pred_sol = torch.FloatTensor(sol).to(device)

        if self.optmodel.modelSense == EPO.MINIMIZE:
            loss = ((pred_cost - true_cost)*(true_sol - pred_sol)).sum(axis =1 )
        if self.optmodel.modelSense == EPO.MAXIMIZE:
            loss = -((pred_cost - true_cost)*(true_sol - pred_sol)).sum(axis =1 )
        # reduction
    
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
        elif reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(reduction))
        return loss

class SCELinearAlternative_Full(optModule):
    """
    An autograd module for Maximum A Posterior contrastive estimation as
    surrogate loss functions, which is a efficient self-contrastive algorithm.

    For the MAP, the cost vector needs to be predicted from contextual data and
    maximizes the separation of the probability of the optimal solution.

    Thus, allows us to design an algorithm based on stochastic gradient descent.

    Reference: <https://www.ijcai.org/proceedings/2021/390>
    """

    def __init__(self, optmodel, processes=1):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (None/optDataset): the training data, usually this is simply the training set
        """
        super().__init__(optmodel, processes,)


    def forward(self, pred_cost, true_cost, true_sol, reduction="mean"):
        """
        Forward pass
        """
        # get device
        device = pred_cost.device
        # convert tensor
        cp = pred_cost.detach().to("cpu").numpy()
        c = true_cost.detach().to("cpu").numpy()
        sol, _ = _solve_in_pass(cp -c, self.optmodel, self.processes, self.pool)
        sol = np.array(sol)
        pred_sol = torch.FloatTensor(sol).to(device)

        if self.optmodel.modelSense == EPO.MINIMIZE:
            loss = ((pred_cost - true_cost)*(true_sol - pred_sol)).sum(axis =1 )
        if self.optmodel.modelSense == EPO.MAXIMIZE:
            loss = -((pred_cost - true_cost)*(true_sol - pred_sol)).sum(axis =1 )
        # reduction
    
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
        elif reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(reduction))
        return loss
