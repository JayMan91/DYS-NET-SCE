#!/usr/bin/env python
# coding: utf-8
"""
optDataset class to obtain tight constraints
"""

import time

from gurobipy import GRB
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from pyepo.model.opt import optModel
from pyepo.data.dataset import optDataset


class optDatasetConstrs(optDataset):
    """
    This class is Torch Dataset for optimization problems with binding constraints.

    Attributes:
        model (optModel): Optimization models
        feats (np.ndarray): Data features
        costs (np.ndarray): Cost vectors
        sols (np.ndarray): Optimal solutions
        ctrs (list(np.ndarray)): active constraints
    """
    def __init__(self, model, feats, costs, skip_infeas=True):
        """
        A method to create a optDataset from optModel

        Args:
            model (optModel): an instance of optModel
            feats (np.ndarray): data features
            costs (np.ndarray): costs of objective function
            skip_infeas (bool): if True, skip infeasible data points
        """
        if not isinstance(model, optModel):
            raise TypeError("arg model is not an optModel")
        self.model = model
        # drop infeasibe or get error
        self.skip_infeas = skip_infeas
        # data
        self.feats = feats
        self.costs = costs
        self.sols, self.objs, self.ctrs = self._getSols()

    def _getSols(self):
        """
        A method to get optimal solutions for all cost vectors
        """
        sols, objs, ctrs, valid_ind = [], [], [], []
        print("Optimizing for optDataset...")
        time.sleep(1)
        tbar = tqdm(self.costs)
        for i, c in enumerate(tbar):
            try:
                # solve
                sol, obj, model = self._solve(c)
                # get binding constrs
                constrs = self._getBindingConstrs(model)
                # print ( "Length of Constrs", len(constrs))
            except AttributeError as e:
                # infeasibe
                if self.skip_infeas:
                    # skip this data point
                    tbar.write("No feasible solution! Drop instance {}.".format(i))
                    continue
                else:
                    # raise the exception
                    raise ValueError("No feasible solution!")
            sols.append(sol)
            objs.append([obj])
            ctrs.append(np.array(constrs))
            valid_ind.append(i)
        # update feats and costs to keep only valid entries
        self.feats = self.feats[valid_ind]
        self.costs = self.costs[valid_ind]
        
        # print ( "Shape of CTRS", len(ctrs)  )
        return np.array(sols), np.array(objs), ctrs

    def _solve(self, cost):
        """
        A method to solve optimization problem to get an optimal solution with given cost

        Args:
            cost (np.ndarray): cost of objective function

        Returns:
            tuple: optimal solution (np.ndarray) and objective value (float)
        """
        # copy model
        # model = self.model.copy()  # This line is causing error
        ### Probaly because opying a whole object is not working
        # set obj
        # model.setObj(cost)
        # optimize
        # sol, obj = model.solve() 

        self.model.setObj(cost)
        sol, obj = self.model.solve()

        return sol, obj, self.model

    def _getBindingConstrs(self, model):
        """
        A method to get tight constraints with current solution

        Args:
            model (optModel): optimization models

        Returns:
            np.ndarray: normal vector of constraints
        """
        xs = model._model.getVars()
        constrs = []
        # if there is lazy constraints
        if hasattr(model, "lazy_constrs"):
            # add lazy constrs to model
            for constr in model.lazy_constrs:
                model._model.addConstr(constr)
            # fix the variables to the optimal
            for var in model._model.getVars():
                var.start = int(var.x)
            # update model
            model._model.update()
            # solve
            model.solve()
        # iterate all constraints
        for constr in model._model.getConstrs():
            # check binding constraints A x == b
            if abs(constr.Slack) < 1e-5:
                t_constr = []
                # get coefficients
                for x in xs:
                    t_constr.append(model._model.getCoeff(constr, x))
                # get coefficients with correct direction
                if constr.sense == GRB.LESS_EQUAL:
                    # <=
                    constrs.append(t_constr)
                elif constr.sense == GRB.GREATER_EQUAL:
                    # >=
                    constrs.append([- coef for coef in t_constr])
                elif constr.sense == GRB.EQUAL:
                    # ==
                    constrs.append(t_constr)
                    constrs.append([- coef for coef in t_constr])
                else:
                    # invalid sense
                    raise ValueError("Invalid constraint sense.")
        # iterate all variables to check bounds
        for i, x in enumerate(xs):
            t_constr = [0] * len(xs)
            # add tight bounds as cosnrtaints
            if x.x <= 1e-5:
                # x_i >= 0
                t_constr[i] = - 1
                constrs.append(t_constr)
            elif x.x >= 1 - 1e-5:
                # x_i <= 1
                t_constr[i] = 1
                constrs.append(t_constr)
        return constrs

    def __len__(self):
        """
        A method to get data size

        Returns:
            int: the number of optimization problems
        """
        return len(self.feats)

    def __getitem__(self, index):
        """
        A method to retrieve data

        Args:
            index (int): data index

        Returns:
            tuple: data features (torch.tensor),
                   costs (torch.tensor),
                   optimal solutions (torch.tensor),
                   objective values (torch.tensor)
        """
        return (
                torch.FloatTensor(self.feats[index]),
                torch.FloatTensor(self.costs[index]),
                torch.FloatTensor(self.sols[index]),
                torch.FloatTensor(self.objs[index]),
                torch.FloatTensor(self.ctrs[index])
            )
### When active constraints was not matchingng, I tried this
# def collate_fn(batch):
#     """
#     A custom collate function for PyTorch DataLoader.
#     """
#     # seperate batch data
#     x, c, w, z, t_ctrs = zip(*batch)
#     # stack lists of x, c, and w into new batch tensors
#     x = torch.stack(x, dim=0)
#     c = torch.stack(c, dim=0)
#     w = torch.stack(w, dim=0)
#     z = torch.stack(z, dim=0)
#     # pad t_ctrs with 0 to make all sequences have the same length.
#     # the number of binding constraints are different.
#     ctrs_padded = pad_sequence(t_ctrs, batch_first=True, padding_value=0)

#     cost_dim = c.size(1)
#     ctrs_dim = ctrs_padded.size (-1)
#     if cost_dim==ctrs_dim:
#         return x, c, w, z, ctrs_padded
#     else:
#         excess_dim = ctrs_dim -  cost_dim
#         ctrs_padded1 = ctrs_padded [:, :, :excess_dim]
#         ctrs_padded2 = ctrs_padded [:, :, -excess_dim:]
#         concat_ctrs_padded = ctrs_padded1 + ctrs_padded2
#         ctrs_padded3  = ctrs_padded [:, :, excess_dim: -excess_dim]
#         new_ctrs_padded = torch.cat((concat_ctrs_padded, ctrs_padded3 ), dim=-1)  

#         return x, c, w, z, new_ctrs_padded

def collate_fn(batch):
    """
    A custom collate function for PyTorch DataLoader.
    """
    # seperate batch data
    x, c, w, z, t_ctrs = zip(*batch)
    # stack lists of x, c, and w into new batch tensors
    x = torch.stack(x, dim=0)
    c = torch.stack(c, dim=0)
    w = torch.stack(w, dim=0)
    z = torch.stack(z, dim=0)
    # pad t_ctrs with 0 to make all sequences have the same length.
    # the number of binding constraints are different.
    ctrs_padded = pad_sequence(t_ctrs, batch_first=True, padding_value=0)
    return x, c, w, z, ctrs_padded


class optDatasetDYS(optDataset):
    """
    This class is Torch Dataset for optimization problems with binding constraints.

    Attributes:
        model (optModel): Optimization models
        feats (np.ndarray): Data features
        costs (np.ndarray): Cost vectors
        sols (np.ndarray): Optimal solutions
        ctrs (list(np.ndarray)): active constraints
    """
    def __init__(self, model, feats, costs, inv_provided = False, skip_infeas=False, allConstraints =  False):
        """
        A method to create a optDataset from optModel

        Args:
            model (optModel): an instance of optModel
            feats (np.ndarray): data features
            costs (np.ndarray): costs of objective function
            skip_infeas (bool): if True, skip infeasible data points
            allConstraints (bool): if True,  all constraints added
        """
        if not isinstance(model, optModel):
            raise TypeError("arg model is not an optModel")
        self.model = model
        # drop infeasibe or get error
        self.skip_infeas = skip_infeas
        # data
        self.feats = feats
        self.costs = costs
        self.inv_provided = inv_provided
        self.allConstraints = allConstraints
        if inv_provided:
            self.sols, self.objs, self.As, self.Ainvs = self._getSols()
        else:
            self.sols, self.objs, self.As = self._getSols()


    def _getSols(self):
        """
        A method to get optimal solutions for all cost vectors
        """
        sols, objs, As, Ainvs,  valid_ind = [], [], [], [], []
        print("Optimizing for optDataset...")
        time.sleep(1)
        tbar = tqdm(self.costs)
        for i, c in enumerate(tbar):
            try:
                # solve
                sol, obj, model = self._solve(c)
                # get binding constrs
                constrs = self._getBindingConstrs(model)
            except AttributeError as e:
                # infeasibe
                if self.skip_infeas:
                    # skip this data point
                    tbar.write("No feasible solution! Drop instance {}.".format(i))
                    continue
                else:
                    # raise the exception
                    raise ValueError("No feasible solution!")
            sols.append(sol)
            objs.append([obj])
            A = np.array(constrs)
            cost_dim = len (c)
            A_dim = A.shape [-1]
            
            if A_dim > cost_dim:
                excess_dim = A_dim -  cost_dim
                A1 = A [:, :, :excess_dim]
                A2 = A [:, :, -excess_dim:]
                A3 = A1 #+ A2
                A4  = A [:, :, excess_dim: -excess_dim]
                newA = torch.cat((A3, A4 ), dim=-1)  
                As.append (newA)
            else:
                # print ("In Else LOOP")
                As.append (A)
                print ("Shape of A", A.shape)
                # print ("## AA ##")
                # print (A.round(decimals=2))
            if self.inv_provided:
 
                At = torch.from_numpy(As[-1]).float()
                A_concatenated = torch.cat ( (At, torch.eye(At.size(0))), dim=1)
                A_inv = torch.pinverse(A_concatenated).numpy()
                Ainvs.append ( np.transpose(A_inv) )
                # print ("Shape of a A and Ainv: ", A.shape, A_inv.shape)
            valid_ind.append(i)
        # update feats and costs to keep only valid entries
        self.feats = self.feats[valid_ind]
        self.costs = self.costs[valid_ind]
        if self.inv_provided:
            return np.array(sols), np.array(objs), As, Ainvs
        return np.array(sols), np.array(objs), As#, Ainvs

    def _solve(self, cost):
        """
        A method to solve optimization problem to get an optimal solution with given cost

        Args:
            cost (np.ndarray): cost of objective function

        Returns:
            tuple: optimal solution (np.ndarray) and objective value (float)
        """
        # copy model
        model = self.model.copy()
        # set obj
        model.setObj(cost)
        # optimize
        sol, obj = model.solve()
        return sol, obj, model

    def _getBindingConstrs(self, model):
        """
        A method to get tight constraints with current solution

        Args:
            model (optModel): optimization models

        Returns:
            np.ndarray: normal vector of constraints
        """
        xs = model._model.getVars()
        constrs = []
        # if there is lazy constraints
        if hasattr(model, "lazy_constrs"):
            # add lazy constrs to model
            for constr in model.lazy_constrs:
                model._model.addConstr(constr)
            # fix the variables to the optimal
            for var in model._model.getVars():
                var.start = int(var.x)
            # update model
            model._model.update()
            # solve
            model.solve()
        # iterate all constraints
        for constr in model._model.getConstrs():
            # check binding constraints A x == b
            
            if (abs(constr.Slack) < 1e-5) or (self.allConstraints):
                # if (abs(constr.Slack) > 1e-5):
                #     print  ("Constraint is being added although not active")

                t_constr = []
                # get coefficients
                for x in xs:
                    t_constr.append(model._model.getCoeff(constr, x))
                # get coefficients with correct direction
                if constr.sense == GRB.LESS_EQUAL:
                    # <=
                    constrs.append(t_constr)
                elif constr.sense == GRB.GREATER_EQUAL:
                    # >=
                    constrs.append([- coef for coef in t_constr])
                elif constr.sense == GRB.EQUAL:
                    # ==
                    constrs.append(t_constr)
                    constrs.append([- coef for coef in t_constr])
                else:
                    # invalid sense
                    raise ValueError("Invalid constraint sense.")
        # iterate all variables to check bounds
        ### NOTE: Not including variable assignmeents as constraints, if variable is 0
        for i, x in enumerate(xs):
            t_constr = [0] * len(xs)
            # add tight bounds as cosnrtaints
            # if x.x <= 1e-5:
            #     # x_i >= 0
            #     t_constr[i] = - 1
            #     constrs.append(t_constr)
            if x.x >= 1 - 1e-5:
                # x_i <= 1
                t_constr[i] = 1
                constrs.append(t_constr)
        return constrs

    def __len__(self):
        """
        A method to get data size

        Returns:
            int: the number of optimization problems
        """
        return len(self.feats)

    def __getitem__(self, index):
        """
        A method to retrieve data

        Args:
            index (int): data index

        Returns:
            tuple: data features (torch.tensor),
                   costs (torch.tensor),
                   optimal solutions (torch.tensor),
                   objective values (torch.tensor)
        """
        if self.inv_provided:
            return (
                    torch.FloatTensor(self.feats[index]),
                    torch.FloatTensor(self.costs[index]),
                    torch.FloatTensor(self.sols[index]),
                    torch.FloatTensor(self.objs[index]),
                    torch.FloatTensor(self.As[index]) ,
                    torch.FloatTensor(self.Ainvs[index])
                )
        else:
            return (
                    torch.FloatTensor(self.feats[index]),
                    torch.FloatTensor(self.costs[index]),
                    torch.FloatTensor(self.sols[index]),
                    torch.FloatTensor(self.objs[index]),
                    torch.FloatTensor(self.As[index]) 
                    )


def DYS_collate_fn(batch, inv_provided = False):
    """
    A custom collate function for PyTorch DataLoader.
    """
    # seperate batch data
    # x, c, w, z, A, Ainv = zip(*batch)
    if inv_provided:
        x, c, w, z, A , Ainv = zip(*batch)
    else:
        x, c, w, z, A = zip(*batch)
    # stack lists of x, c, and w into new batch tensors
    x = torch.stack(x, dim=0)
    c = torch.stack(c, dim=0)
    w = torch.stack(w, dim=0)
    z = torch.stack(z, dim=0)
    # A = torch.stack(A, dim=0)
    A = pad_sequence(A, batch_first=True, padding_value=0)
    cost_dim = c.size(1)
    A_dim = A.size (-1)

    
    if inv_provided:
        # Ainv =  torch.stack(Ainv , dim=0)
        Ainv = pad_sequence(Ainv, batch_first=True, padding_value=0)
        return x, c, w, z, A , Ainv
    return x, c, w, z, A 

class optDatasetDYS_AGG(optDataset):
    """
    This class is Torch Dataset for optimization problems with binding constraints.

    Attributes:
        model (optModel): Optimization models
        feats (np.ndarray): Data features
        costs (np.ndarray): Cost vectors
        sols (np.ndarray): Optimal solutions
        ctrs (list(np.ndarray)): active constraints
    """
    def __init__(self, model, feats, costs, inv_provided = False, skip_infeas=False, allConstraints =  False):
        """
        A method to create a optDataset from optModel

        Args:
            model (optModel): an instance of optModel
            feats (np.ndarray): data features
            costs (np.ndarray): costs of objective function
            skip_infeas (bool): if True, skip infeasible data points
        """
        if not isinstance(model, optModel):
            raise TypeError("arg model is not an optModel")
        self.model = model
        # drop infeasibe or get error
        self.skip_infeas = skip_infeas
        # data
        self.feats = feats
        self.costs = costs
        self.inv_provided = inv_provided
        self.allConstraints = allConstraints
        # if inv_provided:
        #     self.sols, self.objs, self.As, self.Ainvs = self._getSols()
        # else:
        #     self.sols, self.objs, self.As = self._getSols()
        self.sols, self.objs, self.As , self.bs= self._getSols()
    
    def return_activeconstraint_matrix (self):
        A = np.concatenate (self.As ,  dtype = float)
        b =  np.concatenate(self.bs,  dtype = float)

        A, indices = np.unique (A, axis=0, return_index=True)
        b = b[indices]

        x = self.sols[2]
        rhs = np.dot(A, x) - b
        print ("Positive in RHS", np.sum(rhs>1e-3))
        
        # At = torch.from_numpy(A).float()
        # A_concatenated = torch.cat ( (At, torch.eye(At.size(0))), dim=1)
        # A_inv = torch.pinverse(A_concatenated).numpy()
        return A, b#, A_inv


    def _getSols(self):
        """
        A method to get optimal solutions for all cost vectors
        """
        sols, objs, As, bs,  valid_ind = [], [], [], [], []
        print("Optimizing for optDataset...")
        time.sleep(1)
        tbar = tqdm(self.costs)
        for i, c in enumerate(tbar):
            try:
                # solve
                sol, obj, model = self._solve(c)
                # get binding constrs
                constrs, vals = self._getBindingConstrs(model)
            except AttributeError as e:
                # infeasibe
                if self.skip_infeas:
                    # skip this data point
                    tbar.write("No feasible solution! Drop instance {}.".format(i))
                    continue
                else:
                    # raise the exception
                    raise ValueError("No feasible solution!")
            sols.append(sol)
            objs.append([obj])
            A = np.array(constrs)
            cost_dim = len (c)
            A_dim = A.shape [-1]

            if A_dim > cost_dim:
                print ("Dimension of A", A_dim, "Cost dimension", cost_dim)
                excess_dim = A_dim -  cost_dim
                A1 = A [ :, :excess_dim]
                A2 = A [:, -excess_dim:]
                # print ("A1 is same as A2", A1==A2)
                A3 = A1 #np.where(A1 == A2, A1, A1 + A2) 
                ## Explain: coeffcient of (0,t) and (t,0) should be same in a constraint
                ## Only differe for the bound x[0,t]
                A4  = A [ :, excess_dim: -excess_dim]
                newA =  np.concatenate ( (A3, A4), axis = -1 )  # torch.cat((A3, A4 ), dim=-1)  
                As.append (newA)
            
            else:
                As.append (A)
            # if self.inv_provided:
 
            #     At = torch.from_numpy(As[-1]).float()
            #     A_concatenated = torch.cat ( (At, torch.eye(At.size(0))), dim=1)
            #     A_inv = torch.pinverse(A_concatenated).numpy()
            #     Ainvs.append (A_inv)
            bs.append ( np.array(vals) )
            valid_ind.append(i)
        # update feats and costs to keep only valid entries
        self.feats = self.feats[valid_ind]
        self.costs = self.costs[valid_ind]
        if self.inv_provided:
            return np.array(sols), np.array(objs), As, Ainvs
        return np.array(sols), np.array(objs), As, bs#, Ainvs

    def _solve(self, cost):
        """
        A method to solve optimization problem to get an optimal solution with given cost

        Args:
            cost (np.ndarray): cost of objective function

        Returns:
            tuple: optimal solution (np.ndarray) and objective value (float)
        """
        # copy model
        model = self.model.copy()
        # set obj
        model.setObj(cost)
        # optimize
        sol, obj = model.solve()
        return sol, obj, model

    def _getBindingConstrs(self, model):
        """
        A method to get tight constraints with current solution

        Args:
            model (optModel): optimization models

        Returns:
            np.ndarray: normal vector of constraints
        """
        xs = model._model.getVars()
        constrs = []
        vals = []
        # if there is lazy constraints
        if hasattr(model, "lazy_constrs"):
            # add lazy constrs to model
            for constr in model.lazy_constrs:
                model._model.addConstr(constr)
            # fix the variables to the optimal
            for var in model._model.getVars():
                var.start = int(var.x)
            # update model
            model._model.update()
            # solve
            model.solve()
        # iterate all constraints
        for constr in model._model.getConstrs():
            # check binding constraints A x == b
            if (abs(constr.Slack) < 1e-5) or (self.allConstraints):
                t_constr = []
                # get coefficients
                for x in xs:
                    t_constr.append(model._model.getCoeff(constr, x))
                # get coefficients with correct direction
                if constr.sense == GRB.LESS_EQUAL:
                    # <=
                    constrs.append(t_constr)
                    vals.append (constr.RHS)
                elif constr.sense == GRB.GREATER_EQUAL:
                    # >=
                    constrs.append([- coef for coef in t_constr])
                    vals.append (-constr.RHS)
                elif constr.sense == GRB.EQUAL:
                    # ==
                    constrs.append(t_constr)
                    constrs.append([- coef for coef in t_constr])
                    vals.append (constr.RHS)
                    vals.append (-constr.RHS)
                else:
                    # invalid sense
                    raise ValueError("Invalid constraint sense.")
        # iterate all variables to check bounds
        ### NOTE: Not including variable assignmeents as constraints, if variable is 0
        for i, x in enumerate(xs):
            t_constr = [0] * len(xs)
            # add tight bounds as cosnrtaints
            # if x.x <= 1e-5:
            #     # x_i >= 0
            #     t_constr[i] = - 1
            #     constrs.append(t_constr)
            # if x.x >= 1 - 1e-5:
            #     # x_i <= 1
            #     t_constr[i] = 1
            #     constrs.append(t_constr)
            #     vals.append (x.x)
        return constrs, vals

    def __len__(self):
        """
        A method to get data size

        Returns:
            int: the number of optimization problems
        """
        return len(self.feats)

    def __getitem__(self, index):
        """
        A method to retrieve data

        Args:
            index (int): data index

        Returns:
            tuple: data features (torch.tensor),
                   costs (torch.tensor),
                   optimal solutions (torch.tensor),
                   objective values (torch.tensor)
        """
        return (
                torch.FloatTensor(self.feats[index]),
                torch.FloatTensor(self.costs[index]),
                torch.FloatTensor(self.sols[index]),
                torch.FloatTensor(self.objs[index]),
            )
