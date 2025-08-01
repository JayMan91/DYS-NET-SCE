#!/usr/bin/env python
# coding: utf-8
"""
Knapsack problem
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from pyepo.model.grb.grbmodel import optGrbModel


class knapsackModel(optGrbModel):
    """
    This class is optimization model for knapsack problem

    Attributes:
        _model (GurobiPy model): Gurobi model
        weights (np.ndarray / list): Weights of items
        capacity (np.ndarray / listy): Total capacity
        items (list): List of item index
    """

    def __init__(self, weights, capacity, relax= False):
        """
        Args:
            weights (np.ndarray / list): weights of items
            capacity (np.ndarray / list): total capacity
        """
        

        self.weights, self.capacity = np.array(weights), np.array(capacity)
        self.relax = relax
        self.items = list(range(self.weights.shape[1]))
        num_items = len(self.items)

        G = np.array(weights)
        h = np.array(capacity)
        ubG = np.eye(num_items)
        lbG = -np.eye(num_items)
        ubh = np.ones (num_items)
        lbh = np.zeros (num_items)
        G = np.concatenate( (G,ubG),axis =0 )
        h = np.concatenate ( (h, ubh), axis=0)

        self.A, self.b = None, None
        self.C, self.d = G, h
        super().__init__()

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        m = gp.Model("knapsack")
        # varibles
        if self.relax:
            x = m.addVars(self.items, name="x", ub=1, lb=0)
        else:
            x = m.addVars(self.items, name="x", vtype=GRB.BINARY)
        # sense
        m.modelSense = GRB.MAXIMIZE
        # constraints
        for i in range(len(self.capacity)):
            m.addConstr(gp.quicksum(self.weights[i,j] * x[j]
                        for j in self.items) <= self.capacity[i])
        return m, x

    def relax(self):
        """
        A method to get linear relaxation model
        """
        # copy
        model_rel = knapsackModelRel(self.weights, self.capacity)
        return model_rel


class knapsackModelRel(knapsackModel):
    """
    This class is relaxed optimization model for knapsack problem.
    """

    def _getModel(self):
        """
        A method to build Gurobi
        """
        # ceate a model
        m = gp.Model("knapsack")
        # turn off output
        m.Params.outputFlag = 0
        # varibles
        x = m.addVars(self.items, name="x", ub=1)
        # sense
        m.modelSense = GRB.MAXIMIZE
        # constraints
        for i in range(len(self.capacity)):
            m.addConstr(gp.quicksum(self.weights[i,j] * x[j]
                        for j in self.items) <= self.capacity[i])
        return m, x

    def relax(self):
        """
        A forbidden method to relax MIP model
        """
        raise RuntimeError("Model has already been relaxed.")


class qpknapsackModel(optGrbModel):
    """
    This class is optimization model for knapsack problem

    Attributes:
        _model (GurobiPy model): Gurobi model
        weights (np.ndarray / list): Weights of items
        capacity (np.ndarray / listy): Total capacity
        items (list): List of item index
    """

    def __init__(self, weights, capacity, relax= False):
        """
        Args:
            weights (np.ndarray / list): weights of items
            capacity (np.ndarray / list): total capacity
        """
        

        self.weights, self.capacity = np.array(weights), np.array(capacity)
        self.relax = relax
        self.items = list(range(self.weights.shape[1]))
        num_items = len(self.items)

        G = np.array(weights)
        h = np.array(capacity)
        ubG = np.eye(num_items)
        lbG = -np.eye(num_items)
        ubh = np.ones (num_items)
        lbh = np.zeros (num_items)
        G = np.concatenate( (G,ubG),axis =0 )
        h = np.concatenate ( (h, ubh), axis=0)

        self.A, self.b = None, None
        self.C, self.d = G, h
        super().__init__()

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        m = gp.Model("knapsack")
        # varibles
        if self.relax:
            x = m.addVars(self.items, name="x", ub=1, lb=0)
        else:
            x = m.addVars(self.items, name="x", vtype=GRB.BINARY)
        # sense
        m.modelSense = GRB.MAXIMIZE
        m.Params.NonConvex = 2
        # constraints
        for i in range(len(self.capacity)):
            m.addConstr(gp.quicksum(self.weights[i,j] * x[j]
                        for j in self.items) <= self.capacity[i])
        return m, x

    def setObj(self, c):
        """
        A method to set objective function

        Args:
            c (np.ndarray / list): cost of objective function
        """
        # each cost is a vector of dimension n_items x n_items
        num_items = len(self.items)
        c_reshaped = c.reshape (self.num_cost , self.num_cost )
        # print (c_reshaped, c_reshaped.min(), np.linalg.det(c_reshaped))

        obj = gp.quicksum(c_reshaped [i,j] * self.x[i] * self.x[j] for i, k1 in enumerate(self.x) for j, k1 in enumerate(self.x))
        self._model.setObjective(obj)

    def solve(self):
        """
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        self._model.update()
        self._model.optimize()
        sol = np.array(  [self.x[k].x for k in self.x]  )
        sol = np.outer( sol, sol  ).flatten()
        return sol, self._model.objVal


    def relax(self):
        """
        A method to get linear relaxation model
        """
        # copy
        model_rel = knapsackModelRel(self.weights, self.capacity)
        return model_rel