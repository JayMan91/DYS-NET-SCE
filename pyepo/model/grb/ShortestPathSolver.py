# build optModel
from pyepo.model.grb import optGrbModel
import numpy as np
##############################   Optimization Model ######################################
########################################################################################## 

class shortestPathModel(optGrbModel):

    def __init__(self, grid):
        self.grid = grid
        self.A, self.b = self._getAb()
        super().__init__()

    def _getAb(self):
        """
        A helper method to get list of arcs for grid network

        Returns:
           tuple: A and b as numpy arrays
        """
        grid = self.grid
        A_rows = []
        n_nodes = grid[0]*grid[1]
        for i in range(grid[0]):
            # edges on rows
            for j in range(grid[1] - 1):
                col  = np.zeros(n_nodes)
                v = i * grid[1] + j
                col[v] = 1
                col[v+1] = -1
                A_rows.append(col)
            # edges in columns
            if i == grid[0] - 1:
                continue
            for j in range(grid[1]):
                col  = np.zeros(n_nodes)
                v = i * grid[1] + j
                col[v] = 1
                col[v + grid[1]] = -1
                A_rows.append(col)
                
        b = np.zeros(n_nodes)
        b[0], b[-1] = 1, -1
        num_arcs = len(A_rows)


        # ubG = np.eye(num_arcs)
        # lbG = -np.eye(num_arcs)
        # ubh = np.ones (num_arcs)
        # lbh = np.zeros (num_arcs)
        # G = np.concatenate( (ubG, lbG),axis =0 )
        # h = np.concatenate ( ( ubh, lbh), axis=0)



        return np.array(A_rows).T, b
    
    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        import gurobipy as gp
        from gurobipy import GRB
        A, b = self.A, self.b
        n_vertices, n_edges = A.shape

        m = gp.Model("shortest path")
        m.modelSense = GRB.MINIMIZE
        x = m.addVars( n_edges, name="x") # In Gurobi  Default lb= 0.0; https://www.gurobi.com/documentation/current/refman/py_model_addvars.html

        m.addConstrs(gp.quicksum(A[v, e] * x[e] for e in range(n_edges) ) == b[v] for v in range(n_vertices))

        # for i in range(n_vertices):
        #     m.addConstrs(  (x[i] <= 1 for i in range(4))  )


        # x = m.addMVar(shape=A.shape[1],  name="x")
        # m.modelSense = GRB.MINIMIZE
        # m.addConstr(A @ x == b, name="eq")
        return m, x

class shortestPathModelBinary(optGrbModel):

    def __init__(self, grid):
        self.grid = grid
        self.A, self.b = self._getAb()
        super().__init__()

    def _getAb(self):
        """
        A helper method to get list of arcs for grid network

        Returns:
           tuple: A and b as numpy arrays
        """
        grid = self.grid
        A_rows = []
        n_nodes = grid[0]*grid[1]
        for i in range(grid[0]):
            # edges on rows
            for j in range(grid[1] - 1):
                col  = np.zeros(n_nodes)
                v = i * grid[1] + j
                col[v] = 1
                col[v+1] = -1
                A_rows.append(col)
            # edges in columns
            if i == grid[0] - 1:
                continue
            for j in range(grid[1]):
                col  = np.zeros(n_nodes)
                v = i * grid[1] + j
                col[v] = 1
                col[v + grid[1]] = -1
                A_rows.append(col)
                
        b = np.zeros(n_nodes)
        b[0], b[-1] = 1, -1
        num_arcs = len(A_rows)


        # ubG = np.eye(num_arcs)
        # lbG = -np.eye(num_arcs)
        # ubh = np.ones (num_arcs)
        # lbh = np.zeros (num_arcs)
        # G = np.concatenate( (ubG, lbG),axis =0 )
        # h = np.concatenate ( ( ubh, lbh), axis=0)



        return np.array(A_rows).T, b
    
    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        import gurobipy as gp
        from gurobipy import GRB
        A, b = self.A, self.b
        n_vertices, n_edges = A.shape

        m = gp.Model("shortest path")
        m.modelSense = GRB.MINIMIZE
        x = m.addVars( n_edges, name="x", vtype=GRB.BINARY ) # In Gurobi  Default lb= 0.0; https://www.gurobi.com/documentation/current/refman/py_model_addvars.html

        m.addConstrs(gp.quicksum(A[v, e] * x[e] for e in range(n_edges) ) == b[v] for v in range(n_vertices))

        # for i in range(n_vertices):
        #     m.addConstrs(  (x[i] <= 1 for i in range(4))  )


        # x = m.addMVar(shape=A.shape[1],  name="x")
        # m.modelSense = GRB.MINIMIZE
        # m.addConstr(A @ x == b, name="eq")
        return m, x







