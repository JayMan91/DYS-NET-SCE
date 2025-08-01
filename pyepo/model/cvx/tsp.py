import cvxpy as cp
import cvxpylayers
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import torch
class tspMTZcvx:
    def __init__(self, num_nodes):


        self.nodes = list(range(num_nodes))
        self.edges = [(i, j) for i in self.nodes
                    for j in self.nodes if i < j]
        self.directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        self.num_cost = len(self.edges)
        self.num_nodes =  num_nodes
        
    
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
        constraints += [X @ ones == ones]
        constraints += [X.T @ ones == ones]
        constraints += [cp.diag(X) == 0]
        constraints += [u[1:] >= 2]
        constraints += [u[1:] <= num_nodes]
        constraints += [u[0] == 1]

        for i in range(1, num_nodes):
            for j in range(1, num_nodes):
                if i != j:
                    constraints += [ u[i] - u[j] + 1  <= (num_nodes - 1) * (1 - X[i, j]) ]

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



    



# def tspMTZcvx(num_nodes):

#     nodes = list(range(num_nodes))
#     edges = [(i, j) for i in nodes
#                 for j in nodes if i < j]
#     directed_edges = edges + [(j, i) for (i, j) in edges]
#     num_cost = len(edges)


#     c = cp.Parameter (num_cost)
#     X = cp.Variable( (num_nodes, num_nodes) ) # , boolean=True
#     u = cp.Variable(num_nodes)
#     obj = 0
#     for k, (i,j) in enumerate(edges):
#         obj += c[k]* (X[i,j] +  X[j,i]  )


#     objective = cp.Minimize (obj)
#     ones = np.ones((num_nodes,1))

#     # Defining the constraint
#     constraints = [X >=0, X<=1]
#     constraints += [X @ ones == ones]
#     constraints += [X.T @ ones == ones]
#     constraints += [cp.diag(X) == 0]
#     constraints += [u[1:] >= 2]
#     constraints += [u[1:] <= num_nodes]
#     constraints += [u[0] == 1]

#     for i in range(1, num_nodes):
#         for j in range(1, num_nodes):
#             if i != j:
#                 constraints += [ u[i] - u[j] + 1  <= (num_nodes - 1) * (1 - X[i, j]) ]

#     # Solving the problem
#     prob = cp.Problem(objective, constraints)
#     return prob, [c], [X, u ] 