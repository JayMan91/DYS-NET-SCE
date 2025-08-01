
import itertools
import functools
from functools import partial
import heapq
from collections import namedtuple
from pyepo.model.opt import optModel
import numpy as np

class WarcraftdijkstraModel(optModel):
    def __init__(self, grid):
        """
        Args:
            grid (tuple of int): size of grid network
        """
        self.grid = grid
        self.n_nodes = grid[0]*grid[1]
        nodes, edges, nodes_map = self._getEdges()
        ### My Initial Implemetation, Commenting
        A = np.zeros((2 * self.n_nodes, len(edges)))
        for j,e in enumerate(edges):
            ind0 = e[0]
            ind1 = e[1]
            A[ind0,j] = -1.
            A[ind1, j] = +1.

        b = np.zeros(2 * self.n_nodes)
        b[0] = -1.
        b[-1] = 1.
        ##### Code from DYS
        # A = np.zeros((self.n_nodes, len(edges)))
        # for j,e in enumerate(edges):
        #     ind0 = e[0]
        #     ind1 = e[1]
        #     A[ind0,j] = -1.
        #     A[ind1, j] = +1.

        # b = np.zeros( self.n_nodes)
        # b[0] = -1.
        # b[-1] = 1.

        np.savetxt("A_mat.csv", A, delimiter=",")
        self.A, self.b = A, b
        super().__init__()

    def _calNode(self, x, y):
        """
        A method to calculate index of node
        """
        v = x * self.grid[1] + y 
        return v


    def _neighbourhood_fn(self, x, y, x_max, y_max):
        deltas_x = (-1, 0, 1)
        deltas_y = (-1, 0, 1)
        for (dx, dy) in itertools.product(deltas_x, deltas_y):
            x_new, y_new = x + dx, y + dy
            if 0 <= x_new < x_max and 0 <= y_new < y_max and (dx, dy) != (0, 0):
                yield x_new, y_new
    def setObj(self, c):
        self.matrix = c

    def _getModel(self):
        return None, None
        # """
        # A method to build Gurobi model

        # Returns:
        #     tuple: optimization model and variables
        # """
        # import gurobipy as gp
        # from gurobipy import GRB
        # A, b = self.A, self.b
        # n_vertices, n_edges = A.shape

        # m = gp.Model("shortest path")
        # x = m.addVars( n_edges, name="x")

        # m.addConstrs(gp.quicksum(A[v, e] * x[e] for e in range(n_edges) ) == b[v] for v in range(n_vertices))

        # return m, x

    # def _getEdges(self):
    #     """
    #     A method to get list of edges for grid network

    #     Returns:
    #         list: arcs
    #     """
    #     # init list
    #     nodes, edges = [], []
    #     # init map from coord to ind
    #     nodes_map = {}
    #     for i in range(self.grid[0]):
    #         for j in range(self.grid[1]):
    #             u = self._calNode(i, j)
    #             nodes_map[u] = (i,j)
    #             nodes.append(u)
    #             # edge to 8 neighbors
    #             # up
    #             if i != 0:
    #                 v = self._calNode(i-1, j)
    #                 edges.append((u,v))
    #                 # up-right
    #                 if j != self.grid[1] - 1:
    #                     v = self._calNode(i-1, j+1)
    #                     edges.append((u,v))
    #             # right
    #             if j != self.grid[1] - 1:
    #                 v = self._calNode(i, j+1)
    #                 edges.append((u,v))
    #                 # down-right
    #                 if i != self.grid[0] - 1:
    #                     v = self._calNode(i+1, j+1)
    #                     edges.append((u,v))
    #             # down
    #             if i != self.grid[0] - 1:
    #                 v = self._calNode(i+1, j)
    #                 edges.append((u,v))
    #                 # down-left
    #                 if j != 0:
    #                     v = self._calNode(i+1, j-1)
    #                     edges.append((u,v))
    #             # left
    #             if j != 0:
    #                 v = self._calNode(i, j-1)
    #                 edges.append((u,v))
    #                 # top-left
    #                 if i != 0:
    #                     v = self._calNode(i-1, j-1)
    #                     edges.append((u,v))
    #     return nodes, edges, nodes_map


    # This is what I Wrote:

    def _getEdges(self):
        """
        A method to get list of edges for grid network

        Returns:
            list: arcs
        """
        # init list
        nodes, edges = [], []
        n_nodes =  self.n_nodes
        # init map from coord to ind
        nodes_map = {}
        # first connect the self-loop
        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                u = self._calNode(i, j)
                edges.append((u , u + n_nodes))


        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                u = self._calNode(i, j)
                nodes_map[u] = (i,j)
                nodes.append(u)
                # edge to 8 neighbors
                # up
                if i != 0:
                    v = self._calNode(i-1, j)
                    edges.append((u + n_nodes,v   ))
                    # up-right
                    if j != self.grid[1] - 1:
                        v = self._calNode(i-1, j+1)
                        edges.append((u + n_nodes,v  ))
                # right
                if j != self.grid[1] - 1:
                    v = self._calNode(i, j+1)
                    edges.append((u+ n_nodes,v   ))
                    # down-right
                    if i != self.grid[0] - 1:
                        v = self._calNode(i+1, j+1)
                        edges.append((u  + n_nodes,v ))
                # down
                if i != self.grid[0] - 1:
                    v = self._calNode(i+1, j)
                    edges.append((u + n_nodes,v ))
                    # down-left
                    if j != 0:
                        v = self._calNode(i+1, j-1)
                        edges.append((u + n_nodes,v  ))
                # left
                if j != 0:
                    v = self._calNode(i, j-1)
                    edges.append((u + n_nodes,v  ))
                    # top-left
                    if i != 0:
                        v = self._calNode(i-1, j-1)
                        edges.append((u + n_nodes,v ))
        return nodes, edges, nodes_map

    def solve(self, request_transitions=False):
        matrix = self.matrix

        x_max, y_max = self.grid
        matrix = matrix.reshape(x_max, y_max)
        neighbors_func = partial(self._neighbourhood_fn, x_max=x_max, y_max=y_max)

        costs = np.full_like(matrix, 1.0e10)
        costs[0][0] = matrix[0][0]
        num_path = np.zeros_like(matrix)
        num_path[0][0] = 1
        priority_queue = [(matrix[0][0], (0, 0))]
        certain = set()
        transitions = dict()

        while priority_queue:
            cur_cost, (cur_x, cur_y) = heapq.heappop(priority_queue)
            if (cur_x, cur_y) in certain:
                pass

            for x, y in neighbors_func(cur_x, cur_y):
                if (x, y) not in certain:
                    if matrix[x][y] + costs[cur_x][cur_y] < costs[x][y]:
                        costs[x][y] = matrix[x][y] + costs[cur_x][cur_y]
                        heapq.heappush(priority_queue, (costs[x][y], (x, y)))
                        transitions[(x, y)] = (cur_x, cur_y)
                        num_path[x, y] = num_path[cur_x, cur_y]
                    elif matrix[x][y] + costs[cur_x][cur_y] == costs[x][y]:
                        num_path[x, y] += 1

            certain.add((cur_x, cur_y))
        # retrieve the path
        cur_x, cur_y = x_max - 1, y_max - 1
        on_path = np.zeros_like(matrix)
        on_path[-1][-1] = 1
        while (cur_x, cur_y) != (0, 0):
            cur_x, cur_y = transitions[(cur_x, cur_y)]
            on_path[cur_x, cur_y] = 1.0

        is_unique = num_path[-1, -1] == 1
        objVal = (on_path*matrix).sum()

        if request_transitions:
            return on_path.flatten(), objVal, transitions
            return DijkstraOutput(shortest_path=on_path, is_unique=is_unique, transitions=transitions)
        else:
            return on_path.flatten(), objVal
            return DijkstraOutput(shortest_path=on_path, is_unique=is_unique, transitions=None)

