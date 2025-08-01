#!/usr/bin/env python
# coding: utf-8
"""
Differentiable Black-box optimization function
"""

import numpy as np
import torch
from torch.autograd import Function
from torch import nn
from pyepo.func.abcmodule import optModule
from pyepo import EPO
from pyepo.func.utlis import _solveWithObj4Par, _solve_in_pass, _cache_in_pass, regret_loss
from torch.distributions import Normal

class SFGEOpt(optModule):
    """


    Reference: <https://arxiv.org/abs/2307.05213>
    """

    def __init__(self, optmodel, n_samples= 1, std=1., processes=1, solve_ratio=1, entropy_lambda=0., dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            lambd (float): a hyperparameter for differentiable block-box to contral interpolation degree
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (None/optDataset): the training data
        """
        super().__init__(optmodel, processes, solve_ratio, dataset)
        self.std =  std
        self.n_samples = n_samples
        self.entropy_lambda = entropy_lambda


    def forward(self, cost_mean, true_cost, sol_true):
        """
        Forward pass
        """
        std  = self.std
        solve_ratio = self.solve_ratio
        device = cost_mean.device
        batch_size, input_shape = cost_mean.shape
        n_samples = self.n_samples


        dist = Normal(cost_mean, std)
        cp_sample = dist.sample(sample_shape=[n_samples])
        log_prob = dist.log_prob(cp_sample)
        # sols_sample = self.sfge.apply(features, self.std, self.optmodel,
        #                       self.processes, self.pool, self.solve_ratio, self)
        

        rand_sigma = np.random.uniform()
        cp = cp_sample.detach().to("cpu").numpy()
        cp = cp.reshape(batch_size*n_samples,input_shape)

        if rand_sigma <= solve_ratio:
            sol, _ = _solve_in_pass(cp, self.optmodel, self.processes, self.pool)
            if solve_ratio < 1:
                # add into solpool
                self.solpool = np.concatenate((self.solpool, sol))
                # remove duplicate
                self.solpool = np.unique(self.solpool, axis=0)
        else:
            sol, _ = _cache_in_pass(cp, self.optmodel, self.solpool)
        # convert to tensor
        sol = np.array(sol)
        sols_sample = torch.FloatTensor(sol).to(device)
        sols_sample =  sols_sample.view (n_samples, batch_size,input_shape)
        # sols_sample =  sols_sample.mean(dim=0) ### Average of n_samples
        ### No need of averaging, element by element operation does the reshaping itself
        training_loss  = (sols_sample*true_cost - sol_true*true_cost)*self.optmodel.modelSense

        prob = torch.exp(log_prob)
        entropy = -torch.sum(prob*log_prob) 
    
        training_loss = torch.mean(log_prob * training_loss - self.entropy_lambda* entropy)

        return training_loss


# class SFGEOptFunc(Function):
#     """
#     A autograd function for differentiable black-box optimizer
#     """

#     @staticmethod
#     def forward(ctx, c_mean, std, optmodel, processes, pool, solve_ratio, module):
#         """
#         Forward pass for DBB

#         Args:
#             pred_cost (torch.tensor): a batch of predicted values of the cost
#             lambd (float): a hyperparameter for differentiable block-box to contral interpolation degree
#             optmodel (optModel): an PyEPO optimization model
#             processes (int): number of processors, 1 for single-core, 0 for all of cores
#             pool (ProcessPool): process pool object
#             solve_ratio (float): the ratio of new solutions computed during training
#             module (optModule): blackboxOpt module

#         Returns:
#             torch.tensor: predicted solutions
#         """
#         # get device
        

#         dist = Normal(c_mean, std)
#         cp_sample = dist.sample()
#         log_prob = dist.log_prob(cp_sample)
        
#         # convert tenstor
#         device = cp_sample.device
#         cp = cp_sample.detach().to("cpu").numpy()
#         # solve
#         rand_sigma = np.random.uniform()
#         if rand_sigma <= solve_ratio:
#             sol, _ = _solve_in_pass(cp, optmodel, processes, pool)
#             if solve_ratio < 1:
#                 # add into solpool
#                 module.solpool = np.concatenate((module.solpool, sol))
#                 # remove duplicate
#                 module.solpool = np.unique(module.solpool, axis=0)
#         else:
#             sol, _ = _cache_in_pass(cp, optmodel, module.solpool)
#         # convert to tensor
#         sol = np.array(sol)
#         pred_sol = torch.FloatTensor(sol).to(device)
#         return pred_sol, log_prob
#     #     # save
#     #     ctx.save_for_backward(pred_cost, pred_sol)
#     #     # add other objects to ctx
#     #     ctx.lambd = lambd
#     #     ctx.optmodel = optmodel
#     #     ctx.processes = processes
#     #     ctx.pool = pool
#     #     ctx.solve_ratio = solve_ratio
#     #     if solve_ratio < 1:
#     #         ctx.module = module
#     #     ctx.rand_sigma = rand_sigma
#     #     return pred_sol

#     # @staticmethod
#     # def backward(ctx, grad_output):
#     #     """
#     #     Backward pass for DBB
#     #     """
#     #     pred_cost, pred_sol = ctx.saved_tensors
#     #     lambd = ctx.lambd
#     #     optmodel = ctx.optmodel
#     #     processes = ctx.processes
#     #     pool = ctx.pool
#     #     solve_ratio = ctx.solve_ratio
#     #     rand_sigma = ctx.rand_sigma
#     #     if solve_ratio < 1:
#     #         module = ctx.module
#     #     # get device
#     #     device = pred_cost.device
#     #     # convert tenstor
#     #     cp = pred_cost.detach().to("cpu").numpy()
#     #     wp = pred_sol.detach().to("cpu").numpy()
#     #     dl = grad_output.detach().to("cpu").numpy()
#     #     # perturbed costs
#     #     cq = cp + lambd * dl
#     #     # solve
#     #     if rand_sigma <= solve_ratio:
#     #         sol, _ = _solve_in_pass(cq, optmodel, processes, pool)
#     #         if solve_ratio < 1:
#     #             # add into solpool
#     #             module.solpool = np.concatenate((module.solpool, sol))
#     #             # remove duplicate
#     #             module.solpool = np.unique(module.solpool, axis=0)
#     #     else:
#     #         sol, _ = _cache_in_pass(cq, optmodel, module.solpool)
#     #     # get gradient
#     #     grad = []
#     #     for i in range(len(sol)):
#     #         grad.append((sol[i] - wp[i]) / lambd)
#     #     # convert to tensor
#     #     grad = np.array(grad)
#     #     grad = torch.FloatTensor(grad).to(device)
#     #     return grad, None, None, None, None, None, None


