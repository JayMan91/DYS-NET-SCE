#!/usr/bin/env python
# coding: utf-8
"""
Differentiable DYS
"""

import numpy as np
import torch
from torch.autograd import Function
from torch import nn
from pyepo.func.abcmodule import optModule
from pyepo import EPO
from pyepo.func.utlis import _solveWithObj4Par, _solve_in_pass, _cache_in_pass
from pyepo.func.dys_util import standardizeLP, get_AbCd
from pyepo.func.dys_presolve import presolve
from copy import deepcopy
import cvxpy as cp
import cvxpylayers
from cvxpylayers.torch import CvxpyLayer

class CVXOpt(optModule):
    """
    Reference: <https://github.com/cvxgrp/cvxpylayers>
    """

    def __init__(self,  build_from_optmodel , load_cvxmodel ,  optmodel, cvxobj  = None, tau =1., barrier=False,  processes=1,
                  solve_ratio=1, dataset=None, QP= False, 
                  regret_withTransformC= False, cost_transform = None, sol_transform = None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (None/optDataset): the training data
            cvxobj: a tuple (problem, parameters, variables)
        """
        super().__init__(optmodel, processes, solve_ratio, dataset)
        # smoothing parameter
        self.tau = tau
        self.load_cvxmodel =  load_cvxmodel
        self.build_from_optmodel = build_from_optmodel
        self.cvxobj =  cvxobj
        self.cost_transform = cost_transform
        self.sol_transform = sol_transform
        self.regret_withTransformC = regret_withTransformC
        if load_cvxmodel:
            
            problem, parameters, variables = self.cvxobj.getModel(tau = tau)
            self.layer = CvxpyLayer(problem, parameters= parameters, variables= variables)

        elif build_from_optmodel:

            A,b, C, d = get_AbCd (self.optmodel)
            assert A.shape[1]==C.shape[1]
            num_var = A.shape[1]
            
            x = cp.Variable(num_var)
            # constraints = [x >= 0,A @ x == b, C @ x <= d]
            constraints = [x >=0 ]
            if A.shape[0]>0:
                constraints.append (A @ x == b)
            if C.shape[0]>0:
                constraints.append (C @ x <= d)
            if not QP:
                c = cp.Parameter(num_var)
                objective = cp.Minimize(c @ x + tau*cp.pnorm(x, p=2)) 
            if barrier:
                c = cp.Parameter(num_var)
                objective = cp.Minimize(c @ x - tau*cp.log(x).sum() -  tau*cp.log(C @ x).sum()  ) 

            if QP:
                c = cp.Parameter(num_var) #cp.Parameter((num_var, num_var) )
                # c = c.T @ c
                # source:  https://locuslab.github.io/2019-10-28-cvxpylayers/ 
                objective =    cp.Maximize (  cp.sum_squares(c@x)  ) # cp.quad_form(x, c) 


            problem = cp.Problem(objective, constraints)
            self.layer = CvxpyLayer(problem, parameters=[c], variables=[x])
            self.QP = QP
            self.num_var = num_var
        else:
            raise NotImplementedError
        
    def forward(self, pred_cost):
        """
        Forward pass
        """
        if self.cost_transform is not None:
            pred_cost =  self.cost_transform.apply (pred_cost)
            # print ("cost after tranform", pred_cost)
            
        if self.optmodel.modelSense == EPO.MAXIMIZE:
            pred_cost =  -1 *pred_cost 
        
        if self.load_cvxmodel:
            sol = self.layer(pred_cost)
            x = self.cvxobj.extract_sol(sol)
        elif self.build_from_optmodel:
            x, = self.layer(pred_cost)
        # if self.QP:
        #     print (sol.shape)
        #     sol  = sol @ sol.T
        #     sol = sol.flatten()
        #     print (sol.shape)
        # print ("solution", x[:, :20])
        if (self.sol_transform is  None) or (self.regret_withTransformC):
            return x
    
        x_final = self.sol_transform.apply (x)
        return x_final #, x


