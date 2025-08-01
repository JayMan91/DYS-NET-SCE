import cvxpy as cp
import cvxpylayers
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import torch
class portfolioModel:
    def __init__(self, num_assets, covariance, gamma=2.25):
        self.num_assets = num_assets
        self.covariance = covariance
        self.risk_level = self._getRiskLevel(gamma)
        super().__init__()

    def _getRiskLevel(self, gamma):
        """
        A method to calculate the risk level

        Returns:
            float: risk level
        """
        risk_level = gamma * np.mean(self.covariance)
        return risk_level

   
    def getModel(self, tau = 0.):
        c = cp.Parameter (self.num_assets )
        x = cp.Variable( self.num_assets) 
        
        risk = cp.quad_form(x,self.covariance)
        constraints = [x >=0, x<=1, cp.sum(x) <= 1, risk <= self.risk_level ]
        objective = cp.Minimize(c @ x + tau*cp.pnorm(x, p=2))
        prob = cp.Problem(objective, constraints)
        return prob, [c], [x ] 

    
    def extract_sol(self , sol):
        x, = sol
        return x



    



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