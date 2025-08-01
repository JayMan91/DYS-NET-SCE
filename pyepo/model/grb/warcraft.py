#!/usr/bin/env python
# coding: utf-8

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from pyepo.model.grb.grbmodel import optGrbModel

class  WarcraftshortestPathNodeModel(optGrbModel):
    """
    This class is optimization model for shortest path problem on 2D grid with 8 neighbors

    Attributes:
        _model (GurobiPy model): Gurobi model
        grid (tuple of int): Size of grid network
    """

    def __init__(self, grid):
        """
        Args:
            grid (tuple of int): size of grid network
        """
        self.grid = grid
        self.n_nodes = grid[0]*grid[1]
        self.A, self.b = self._getAb()
        super().__init__()
        
    def _add_edge(self, source_indices, dest_indices ):
        n_nodes = self.n_nodes
        grid = self.grid
        (i_source, j_source) = source_indices
        (i_dest, j_dest) = dest_indices
        col  = np.zeros( 2*n_nodes)
        col [i_source * grid[1] + j_source]= 1
        col [i_dest * grid[1] + j_dest]= -1
        return col
    def _getAb(self):
        """
        A helper method to get list of arcs for grid network

        Returns:
           tuple: A and b as numpy arrays
        """
        grid = self.grid

        A_rows = []

        for i in range(grid[0]):
            for j in range(grid[1]):
                A_rows.append( self._add_edge( (i,j), (i+grid[0],j)  ) )

        for i in range(grid[0]):
            for j in range(grid[1]):
                if i >0:
                    A_rows.append( self._add_edge( (i+grid[0],j), (i-1,j)  ) )
                    if j>0:
                        A_rows.append( self._add_edge( (i+grid[0],j), (i-1,j-1)  ) )
                    if j< grid[1]-1:
                        A_rows.append( self._add_edge( (i+grid[0],j), (i-1,j+1)  ) )             
                        
                if i < grid[0]-1:
                    A_rows.append( self._add_edge( (i+grid[0],j), (i+1,j)  ) )
                    if j>0:
                        A_rows.append( self._add_edge( (i+grid[0],j), (i+1,j-1)  ) )
                    if j< grid[1]-1:
                        A_rows.append( self._add_edge( (i+grid[0],j), (i+1,j+1)  ) )   
                if j>0:
                    A_rows.append( self._add_edge( (i+grid[0],j), (i,j-1)  ) )
                if j< grid[1]-1:
                    A_rows.append( self._add_edge( (i+grid[0],j), (i,j+1)  ) )  

        b = np.zeros( 2*self.n_nodes)
        b[0], b[-1] = 1, -1
        print ( "Shape of A: ", np.array(A_rows).shape  )
        return np.array(A_rows).T, b

        #### If we want to impose the constraint if the edge i-j is selected j-i cannot be selected
        #### Better To add Subtour elimination constraint, not adding

        # grid = self.grid

        # A_rows = []
        # edgeIndex = 0
        # edge_name= []
        # for i in range(grid[0]):
        #     for j in range(grid[1]):
        #         A_rows.append( self._add_edge( (i,j), (i+grid[0],j)  ) )
        #         edge_name.append( 'In_{}_{}.Out_{}_{}'.format(i,j,i,j) )
        #         edgeIndex +=1
        # non_overlapping_edges = []
        
        # for i in range(grid[0]):
        #     for j in range(grid[1]):
        #         # if i >0:
        #         #     A_rows.append( self._add_edge( (i+grid[0],j), (i-1,j)  ) )
        #         #     if j>0:
        #         #         A_rows.append( self._add_edge( (i+grid[0],j), (i-1,j-1)  ) )
        #         #     if j< grid[1]-1:
        #         #         A_rows.append( self._add_edge( (i+grid[0],j), (i-1,j+1)  ) )             
                        
        #         if i < grid[0]-1:
        #             A_rows.append( self._add_edge( (i+grid[0],j), (i+1,j)  ) )
        #             edgeIndex +=1
        #             edge_name.append( 'In_{}_{}.Out_{}_{}'.format(i,j,i+1,j) )
                    
        #             A_rows.append( self._add_edge( (i+1+grid[0],j), (i,j)  ) )
                    
        #             non_overlapping_edges.append( (edgeIndex, edgeIndex-1) )
        #             edgeIndex +=1
        #             edge_name.append( 'In_{}_{}.Out_{}_{}'.format(i+1,j,i,j) )

        #             if j< grid[1]-1:
        #                 A_rows.append( self._add_edge( (i+grid[0],j), (i+1,j+1)  ) )
        #                 edgeIndex +=1
        #                 edge_name.append( 'In_{}_{}.Out_{}_{}'.format(i,j,i+1,j+1) )

        #                 A_rows.append( self._add_edge( (i+1+grid[0],j+1), (i,j)  ) )   
        #                 non_overlapping_edges.append( (edgeIndex, edgeIndex-1) )
        #                 edgeIndex +=1
        #                 edge_name.append( 'In_{}_{}.Out_{}_{}'.format(i+1,j+1,i,j) )
        #             if j>0:
        #                 A_rows.append( self._add_edge( (i+grid[0],j), (i+1,j-1)  ) )
        #                 edgeIndex +=1
        #                 edge_name.append( 'In_{}_{}.Out_{}_{}'.format(i,j,i+1,j-1) )

        #                 A_rows.append( self._add_edge( (i+1+grid[0],j-1), (i,j  )  ) )
        #                 non_overlapping_edges.append( (edgeIndex, edgeIndex-1) )
        #                 edgeIndex +=1
        #                 edge_name.append( 'In_{}_{}.Out_{}_{}'.format(i+1,j-1,i,j) )
        #         if j< grid[1]-1:
        #             A_rows.append( self._add_edge( (i+grid[0],j), (i,j+1)  ) )
        #             edgeIndex +=1
        #             edge_name.append( 'In_{}_{}.Out_{}_{}'.format(i,j,i,j+1) )

        #             A_rows.append( self._add_edge( (i+grid[0],j+1), (i,j)  ) )  
        #             non_overlapping_edges.append( (edgeIndex, edgeIndex-1) )
        #             edgeIndex +=1
        #             edge_name.append( 'In_{}_{}.Out_{}_{}'.format(i,j+1,i,j) )
        # b = np.zeros( 2*self.n_nodes)
        # b[0], b[-1] = 1, -1

        # n_edges = edgeIndex
        # G_rows = []
        # for i in range(len(non_overlapping_edges)):
        #     s,t = non_overlapping_edges[i]
        #     c =  np.zeros(n_edges)
        #     c[s] = c[t] =1
        #     G_rows.append(c)
        
        # h = np.ones(len(non_overlapping_edges))
        # print (np.array(G_rows), h)
        # self.edge_name =  edge_name
        # return np.array(A_rows).T, b

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
        x = m.addVars( n_edges, ub=1, name= "x") 

        m.addConstrs(gp.quicksum(A[v, e] * x[e] for e in range(n_edges) ) == b[v] for v in range(n_vertices))
        return m, x

    def setObj(self, c):
        """
        A method to set objective function

        Args:
            c (np.ndarray): cost of objective function
        """
        # vector to matrix
        extended_cost  = np.zeros(self.A.shape[1])
        extended_cost[0:self.n_nodes] = c.flatten()

        obj = gp.quicksum(extended_cost[i] * self.x[k] for i, k in enumerate(self.x))
        self._model.setObjective(obj)

    def solve(self):
        """
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        self._model.update()
        self._model.optimize()
        solution = np.array([self.x[k].x for k in self.x])
        # f = open("solution.txt",'w')
        # for k in range (len(solution)):
        #     if self.x[k].x>0.99:
        #         print(self.x[k].VarName, self.x[k].x,  file=f)
        # f.close()
            
        if self._model.status!=2:
            print(self._model.status)
            if self._model.status==3:
                self._model.computeIIS()
                self._model.write("infreasible_nodeweightedSP.ilp")        

        return solution[0:self.n_nodes], self._model.objVal
