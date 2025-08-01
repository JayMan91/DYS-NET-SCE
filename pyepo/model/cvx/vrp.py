import cvxpy as cp
import cvxpylayers
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import torch
class vrpModel:
    def __init__(self, num_nodes, demands, capacity, num_vehicle):

        self.num_nodes =  num_nodes + 1
        self.nodes = list(range(self.num_nodes))
        self.edges = [(i, j) for i in self.nodes
                    for j in self.nodes if i < j]
        self.directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        self.num_cost = len(self.edges)
        self.demands = demands
        self.capacity = capacity
        self.num_vehicle = num_vehicle   
        
    
    def _buildModel(self, tau = 0.):
        num_cost = self.num_cost
        num_nodes = self.num_nodes
        edges =  self.edges

        c = cp.Parameter (num_cost)
        X = cp.Variable( (num_nodes, num_nodes) ) # , boolean=True
        ### X should be a boolean, relaxing the variable
        u = cp.Variable(num_nodes)
        obj = 0
        for k, (i,j) in enumerate(edges):
            obj += c[k]* (X[i,j] +  X[j,i]  ) + tau * ( cp.square(X[i,j] )  +   cp.square(X[j,i])   )


        objective = cp.Minimize (obj)
        ones = np.ones((num_nodes,1))

        # Defining the constraint
        constraints = [X >=0, X<=1] # first contraint to bound x between 0 and 1
        constraints += [X[1:,:] @ ones == ones[1:,:]  ]
        constraints += [X[:, 1:].T @ ones == ones[1:,:] ]

        constraints += [X[0, :]@ ones <= self.num_vehicle ]
        constraints += [X[:,0].T @ ones <= self.num_vehicle ]

        constraints += [cp.diag(X) == 0]
        # constraints += [u[1:] >= 2]
        # constraints += [u[1:] <= num_nodes]
        # constraints += [u[0] == 1]

        for i in range (num_nodes):
            constraints += [ u[i]<= self.capacity ]
            if i ==0:
                constraints +=  [ u[i] >= 0 ]
            else:
                constraints +=  [ u[i] >= self.demands[i -1] ]
        for i in range(1, num_nodes):
            for j in range(1, num_nodes):
                if i != j:
                    constraints += [ u[i] - u[j] + self.demands[j-1]  <= self.capacity * (1 - X[i, j]) ]

        # Solving the problem
        prob = cp.Problem(objective, constraints)

        self.problem, self.parameters, self.variables = prob, [c], [X, u ] 
    
    def getModel(self, tau = 0.):
        self._buildModel(tau = tau)
        return self.problem, self.parameters, self.variables
    
    def extract_sol(self , sol):
        X_val , u_val = sol
        b, r, c = X_val.shape
        I = torch.ones ( r,c)
        indices = (torch.triu( I, diagonal=1 ) == 1).expand(b, -1, -1)
        sol = X_val[indices ] + (X_val.transpose(1, 2))[indices]

        return sol.view(b, -1)