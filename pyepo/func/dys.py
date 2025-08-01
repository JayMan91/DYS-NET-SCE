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
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool

@torch.jit.script
def proj1(sol: torch.Tensor, A: torch.Tensor, b: torch.Tensor, A_pseudoinv: torch.Tensor ) -> torch.Tensor:
    return sol -   torch.einsum('ij,bj->bi', A_pseudoinv  , (  torch.einsum('ij,bj->bi', A, sol)-b )) 
  
@torch.jit.script
def proj2(sol: torch.Tensor)  -> torch.Tensor:
    return torch.clamp(sol, min = 0. )

@torch.jit.script
def DY_split (sol: torch.Tensor, pred_cost: torch.Tensor, A: torch.Tensor, b: torch.Tensor, A_pseudoinv: torch.Tensor,
               alpha: float = 0.1, tau: float = 1e-1 )  -> torch.Tensor:
    return sol - proj2(sol) + proj1( (2 -  tau*alpha)* proj2(sol)- sol - alpha*pred_cost,  A, b, A_pseudoinv )

@torch.jit.script
def _call_DYS (c_inputi: torch.Tensor, Asi: torch.Tensor,   bsi: torch.Tensor, Asi_inv: torch.Tensor = None, 
        alpha: float = 0.1, num_iter: int = 100, tau: float = 1e-1, decay_param: float =10,  inv_provided: bool = False) -> torch.Tensor:
    
    Aeq,   b = Asi,  bsi
    # non_zero_mask = Aeq.any(dim=1)
    # Aeq = Aeq [non_zero_mask]
    # b = b [non_zero_mask]

    n_cons, n_var =  Aeq.size()
    A = torch.cat ( (Aeq, torch.eye(n_cons)), dim=1) 
    # print ("Shape of Asi", Asi.shape, "cost shape", c_inputi.shape)
    if inv_provided:
        A_pseudoinv = Asi_inv
    else:
        A_pseudoinv = torch.pinverse( A )

    c_inputi = c_inputi.view (1, -1)

    sol_ = torch.zeros_like(c_inputi).float()
    # alpha = deepcopy( self.alpha )
 

    for i in range (num_iter):
        
        alpha_ = alpha* ( (1-(i/num_iter) )**(1/decay_param) )
        # print ("iteration: ", i, "Alpha: ",alpha_)
        sol_ = DY_split(sol_, c_inputi,   A, b, A_pseudoinv, alpha_, tau)

    sol =  proj2(sol_)
    # print ("SOL", sol)
    return sol

class DYSOpt_OTF(nn.Module):
    def __init__(self, num_iter = 50, alpha = 0.05, tau = 1., decay_param = 10, processes=1, inv_provided= False):

        """
        Args:
            Ab: A tuple of two , For Ax=b
        """
        super().__init__()
        if processes not in range(mp.cpu_count()+1):
            raise ValueError("Invalid processors number {}, only {} cores.".
                format(processes, mp.cpu_count()))
        self.processes = mp.cpu_count() if not processes else processes  
        if processes == 1:
            self.pool = None
        # multi-core
        else:
            self.pool = ProcessingPool(processes)

        self.num_iter = num_iter
        self.alpha = alpha
        self.tau = tau
        self.decay_param = decay_param
        self.inv_provided = inv_provided

    def proj1(self, sol, A, b, A_pseudoinv ):
        return sol -   torch.einsum('ij,bj->bi', A_pseudoinv  , (  torch.einsum('ij,bj->bi', A, sol)-b ))   

    def proj2(self, sol):
        return nn.ReLU()(sol)
    def DY_split (self, sol, pred_cost, alpha, tau,  A, b, A_pseudoinv):
        return sol - self.proj2(sol) + self.proj1( (2 -  tau*alpha)* self.proj2(sol)- sol - alpha*pred_cost,  A, b, A_pseudoinv )
    def forward(self, c_input, As , bs, As_inv = None):
        """
        Forward pass
        """
        device = c_input.device
        if self.processes == 1:
            wp = torch.empty(c_input.shape).to(device)
            for bi, _ in enumerate(zip(c_input, As,   bs)):
                if  self.inv_provided:
                    # wp[bi] =  self._call_DYS( c_input[bi],  As[bi], bs[bi], As_inv[bi] )
                    wp[bi] =  _call_DYS( c_input[bi],  As[bi], bs[bi], As_inv[bi] ,
                                    alpha = self.alpha, num_iter = self.num_iter, tau = self.tau,
                                     decay_param = self.decay_param,  inv_provided = self.inv_provided)
                else:
                    # wp[bi] =  self._call_DYS( c_input[bi],  As[bi], bs[bi] )
                    wp[bi] = _call_DYS( c_input[bi],  As[bi], bs[bi],
                                       alpha = self.alpha, num_iter = self.num_iter, tau = self.tau,
                                     decay_param = self.decay_param,  inv_provided = self.inv_provided )
            
        # wp = self.dysopt(cp)
        else:
            res = self.pool.amap(self._call_DYS, c_input , As,  bs, As_inv).get()
            wp, _ = zip(*res)
            wp = torch.stack(wp, dim=0).to(device)
        
        return wp
  
    # def _call_DYS (self, c_inputi, Asi,   bsi, Asi_inv = None):
        
    #     Aeq,   b = Asi,  bsi
    #     n_cons, n_var =  Aeq.size()
    #     A = torch.cat ( (Aeq, torch.eye(n_cons)), dim=1) 
    #     # print ("Shape of Asi", Asi.shape, "cost shape", c_inputi.shape)
    #     if self.inv_provided:
    #         A_pseudoinv = Asi_inv
    #     else:
    #         A_pseudoinv = torch.pinverse( A )

    #     c_inputi = c_inputi.view (1, -1)
    #     num_iter = self.num_iter
    
    #     tau = self.tau
    #     decay_param = self.decay_param
    #     sol_ = torch.zeros_like(c_inputi).float()
    #     alpha = deepcopy( self.alpha )
    #     tau = self.tau  
    #     decay_param = self.decay_param
    #     for i in range (self.num_iter):
            
    #         alpha_ = alpha* ( (1-(i/self.num_iter) )**(1/decay_param) )
    #         # print ("iteration: ", i, "Alpha: ",alpha_)
    #         sol_ = self.DY_split(sol_, c_inputi, alpha_, tau,  A, b, A_pseudoinv)

    #     sol =  self.proj2(sol_)

    #     return sol

## Below function is assumed Ainvs is p[recomputed ]
# class DYS_OTF(nn.Module):
#     def __init__(self, num_iter = 50, alpha = 0.05, tau = 1., decay_param = 10,    processes=1):

#         """
#         Args:
#             Ab: A tuple of two , For Ax=b
#         """
#         super().__init__()
#         if processes not in range(mp.cpu_count()+1):
#             raise ValueError("Invalid processors number {}, only {} cores.".
#                 format(processes, mp.cpu_count()))
#         self.processes = mp.cpu_count() if not processes else processes  
#         if processes == 1:
#             self.pool = None
#         # multi-core
#         else:
#             self.pool = ProcessingPool(processes)

#         self.num_iter = num_iter
#         self.alpha = alpha
#         self.tau = tau
#         self.decay_param = decay_param

#     def proj1(self, sol, A, b, A_pseudoinv ):
        

#         return sol -   torch.einsum('ij,bj->bi', A_pseudoinv  , (  torch.einsum('ij,bj->bi', A, sol)-b ))   

#     def proj2(self, sol):
#         return nn.ReLU()(sol)
#     def DY_split (self, sol, pred_cost, alpha, tau,  A, b, A_pseudoinv):
#         return sol - self.proj2(sol) + self.proj1( (2 -  tau*alpha)* self.proj2(sol)- sol - alpha*pred_cost,  A, b, A_pseudoinv )
#     def forward(self, c_input, As, Ainvs,  bs):
#         """
#         Forward pass
#         """
#         device = c_input.device
#         if self.processes == 1:
#             wp = torch.empty(c_input.shape).to(device)
#             for bi, _ in enumerate(zip(c_input, As, Ainvs,  bs)):
#                 wp[bi] =  self._call_DYS( c_input[bi],  As[bi], Ainvs[bi],  bs[bi] )
            
#         # wp = self.dysopt(cp)
#         else:
#             # calculate projections with pool
#             res = self.pool.amap(self._call_DYS, c_input , As, Ainvs, bs).get()
#             # the projection
#             wp, _ = zip(*res)
#             wp = torch.stack(wp, dim=0).to(device)
        
#         return wp
  
#     def _call_DYS (self, c_inputi, Asi, Ainvsi,  bsi):
#         # print ("Shape of Asi", Asi.shape)
#         A,  A_pseudoinv, b = Asi, Ainvsi,  bsi
#         # print ("BS: ", bsi)
#         # print ("As: ", Asi)
#         c_inputi = c_inputi.view (1, -1)
#         num_iter = self.num_iter
    
#         tau = self.tau
#         decay_param = self.decay_param
#         sol_ = torch.zeros_like(c_inputi).float()
#         alpha = deepcopy( self.alpha )
#         tau = self.tau  
#         decay_param = self.decay_param
#         for i in range (self.num_iter):
            
#             alpha_ = alpha* ( (1-(i/self.num_iter) )**(1/decay_param) )
#             # print ("iteration: ", i, "Alpha: ",alpha_)
#             sol_ = self.DY_split(sol_, c_inputi, alpha_, tau,  A, b, A_pseudoinv)


#         sol =  self.proj2(sol_)

#         return sol

class DYSOpt(optModule):
    """
    Reference: <https://arxiv.org/pdf/2301.13395>
    """

    def __init__(self, optmodel,  num_iter = 50, alpha = 0.05, tau = 1., decay_param = 10, 
                  dopresolve=False, doQR= False, doScale =False,
                 processes=1, solve_ratio=1, AbCd_provided= None, dataset=None, 
                 cost_transform = None, sol_transform = None, regret_withTransformC= False,
                 verbose=False):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (None/optDataset): the training data
            AbCD_provided: A tuple of four , For Ax=b and Cx<= d 
        """
        super().__init__(optmodel, processes, solve_ratio, dataset)
        # smoothing parameter

        self.num_iter = num_iter
        self.alpha = alpha
        self.tau = tau
        self.decay_param = decay_param

        if AbCd_provided is None:
            A,b, C, d = get_AbCd (self.optmodel)
        else:
            A,b, C, d = AbCd_provided
        print ("Shape Check: ", A.shape, C.shape)
        

        if dopresolve:
            presolver = presolve (C, d, A,b)
            (C, d, A,b) = presolver.transform ()

    
        self.standardizer = standardizeLP (C, d, A,b)
        A_np, b_np = self.standardizer.getAb()
        print ("Shape Check: ", A_np.shape, b_np.shape)
   

        if doScale:

            row_max = A_np.max(axis=1)

            # Avoid division by zero or negative 
            row_max[row_max <= 0] = 1.

            # Divide each row
            A_scaled = A_np / row_max[:, None]  # shape: (m, n)
            b_scaled = b_np / row_max           # shape: (m,)

            A_np , b_np = A_scaled , b_scaled


        if doQR:
            print ("Shape of A matrix before QR: ", A_np.shape)
            Q, A_np = np.linalg.qr(A_np)
            b_np = np.dot(Q.T, b_np)
            print ("Shape of A matrix after QR: ", A_np.shape)
        
        
            
        U, S, Vh = np.linalg.svd(A_np, full_matrices=False)
        ### This change I had done for TSP, other experimentes were run without this, 
        # For some reason I had to convert as array only for TSP
        U = np.asarray(U)
        Vh = np.asarray(Vh)
        # print ("S", S)
        S_chosen = S[S>1e-3]
        # S_chosen = S[S>5]
        n_chosen = len(S_chosen)
        # print (n_chosen)


        U_chosen = U[:,:n_chosen]
        Vh_chosen = Vh[:n_chosen,:]


        A_pseudoinv = np.dot(Vh_chosen.T * (1/S_chosen), U_chosen.T)

       
        self.A_pseudoinv = torch.from_numpy(A_pseudoinv).float()
        self.A, self.b = torch.from_numpy (A_np).float() ,   torch.from_numpy (b_np).float()
        if verbose:
            print ("Matmul", torch.matmul (self.A, self.A_pseudoinv)  )
            print ("Matmul Inverse", torch.matmul (self.A_pseudoinv, self.A)  )
        self.cost_transform = cost_transform
        self.sol_transform = sol_transform
        self.regret_withTransformC = regret_withTransformC
        self.doScale = doScale
        self.verbose = verbose


    def proj1(self, sol):
        A, b, A_pseudoinv = self.A, self.b, self.A_pseudoinv

        return sol -   torch.einsum('ij,bj->bi', A_pseudoinv  , (  torch.einsum('ij,bj->bi', A, sol)-b ))   

    def proj2(self, sol):
        return torch.clamp( sol, min=0)
    def DY_split (self, sol, c_hat, alpha, tau):
        A, b, A_pseudoinv = self.A, self.b, self.A_pseudoinv
        if self.verbose:
            print ("Sol",sol )
            print ("alpha * c_hat", alpha * (c_hat + ( 1e-6 + tau )* sol) )
        update =    (
                        sol - self.proj2(sol) 
                        + self.proj1(
                        (2 - tau * alpha) * self.proj2(sol)
                        - sol - alpha * ( c_hat + ( 1e-6 + tau )* sol ) #  + ( 1e-6 + tau )* sol
                                )
                    )
        return  update
    def forward(self, pred_cost, init_sol = None):
        """
        Forward pass
        """
        if self.cost_transform is not None:
            pred_cost =  self.cost_transform.apply (pred_cost)
            # pred_cost = nn.functional.normalize(pred_cost, p=1,dim = 1)
            # print ("cost after tranform", pred_cost)
        if self.optmodel.modelSense == EPO.MAXIMIZE:
            pred_cost =  -1 *pred_cost 
        true_length = pred_cost.shape[1] 
        c_ = self.standardizer.transformC(pred_cost).float()
        changed_length = c_.shape[1]
        c_orig = c_.clone()
        
        if self.doScale:
            c_ = nn.functional.normalize(c_, p=1, dim = 1)
            # c_truncated = c_[:, :true_length]
            # c_truncated = nn.functional.normalize(c_truncated, p=1, dim = 1)
            # c_ = torch.cat([c_truncated, 
            #             torch.zeros(c_.shape[0], changed_length - true_length, 
            #                         device=c_.device)], dim=1)
        if self.verbose:
            print ("C before starting itertaion: ", c_)

        if init_sol is None:
            sol_ =  torch.randn_like (c_).float() # torch.zeros_like (c_).float()
        else:
            sol_ = init_sol.clone().detach()
            num_nu_columns = changed_length - true_length

            sol_ = torch.cat([sol_, torch.zeros( len(sol_), num_nu_columns)], dim=1)  
            sol_ = sol_.float()
        alpha = deepcopy( self.alpha )
        tau = self.tau  
        decay_param = self.decay_param
        # with torch.no_grad():
        for i in range (self.num_iter ):
        
            alpha_ = alpha * ( (1-(i/self.num_iter) )**(1/decay_param) )
            if self.verbose:
                print     ("iteration: ", i, "Alpha: ",alpha_)
                # print (sol_[:, :20])
            sol_ = self.DY_split(sol_, c_, alpha_, tau)
        
        # for _ in range (2):
        #     sol_ =  self.proj2(sol_)
        #     sol_ =  self.proj1(sol_)

        # sol_ =  self.proj1(sol_)

        sol_ = self.DY_split(sol_, c_, alpha, tau)
        # if self.doScale:
        #     sol_ = nn.functional.normalize(sol_, p=1, dim = 1)
        sol_ =  self.proj2(sol_)

        sol = sol_
        if self.verbose:
            print ("solution", sol[:, :20]) 
        sol = self.standardizer.transformsolution (sol).float()

        if (self.sol_transform is  None) or (self.regret_withTransformC):
            return sol
    
        sol_final = self.sol_transform.apply (sol)
        return sol_final
