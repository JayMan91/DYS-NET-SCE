"""
facility location problem solver
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from pyepo.model.grb.grbmodel import optGrbModel

class FacilityLocationModel(optGrbModel):
    """
    This class is optimization model for Facility Location Problem with Unknown Transport_costs

    Attributes:

    """
    def __init__(self,
                 demands=None,
                 capacities=None,
                 setup_costs=None,
                 relax  = False):
        """
        Args:
            num_customer (int) : 
            num_facilities (int) :
            demands (list(int)): customer demands; len num_customer
            capacities  (list(int)): facility capacities; len num_facilities
            setup_costs  (list(int)): cost of setting up facilities; len num_facilities
        """
        self.num_customers = len (demands)
        self.num_facilities = len (capacities)
        assert len(setup_costs) == self.num_facilities
        # self.cartesian_product = list(product(range(num_customers), range(num_facilities)))
        # self.d = len(self.cartesian_product)
        self.customers = list (range (self.num_customers))
        self.facilities = list (range (self.num_facilities))


        self.demands = demands
        self.capacities = capacities
        self.setup_costs = setup_costs
        ### edges between customer and facility 
        self.edges = [(c, f) for c in self.customers
                      for f in self.facilities]
        self._createAbCd()
        self.relax = relax
        
        super().__init__()

    @property
    def num_cost(self):
        return len(self.edges)

    def _createAbCd (self):
        '''
        A1 defines the eqaility constraints: \sum_f edge (c,f) = 1
        C2 defines the ineqalty constraints: \sum_c demand[c] edge [c,f] <= capacities[f]*assign [f]
        C3 defines assign [f] <= 1
        Objective is:
                \sum_c \sum_f edge (c,f) demand [c] cost[c,f] + \sum_f fixed_cost [f]* assign [f]
        '''
        A1 = np.zeros(( self.num_customers, len(self.edges) + self.num_facilities ))
        C2 = np.zeros(( self.num_facilities, len(self.edges) + self.num_facilities ))
        C3 = np.zeros(( self.num_facilities, len(self.edges) + self.num_facilities ))

        for c in self.customers:
            start_index = c * (self.num_facilities)
            A1 [ c , start_index: (start_index + self.num_facilities)] = 1
        
        for f in self.facilities:
            for c in self.customers :
                C2 [f, f + c * self.num_facilities] = self.demands[c]
            C2 [f , len(self.edges) + f ] = - self.capacities [f]
            C3 [f,   len(self.edges) + f ] = 1


        b1 = np.ones (self.num_customers)
        d2  = np.zeros (self.num_facilities)
        d3 = np.ones (self.num_facilities)

        self.C = np.concatenate((C2, C3)) 
        self.d = np.concatenate((d2, d3))

        self.A = A1
        self.b = b1

    def _getModel(self):
        """
        A method to build Gurobi model
        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        m = gp.Model("FL")
        # turn off output
        m.Params.outputFlag = 0
        # varibles
        # directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        x = m.addVars(self.edges, ub=1, vtype=gp.GRB.CONTINUOUS, name='Assign')
        if self.relax:
            self.select = m.addVars(self.num_facilities, ub=1, vtype=gp.GRB.CONTINUOUS, lb =0, name='Select')
        else:
            self.select = m.addVars(self.num_facilities, vtype=gp.GRB.BINARY, name='Select')
        # sense
        m.modelSense = GRB.MINIMIZE
        # constraints
        m.addConstrs(
            (gp.quicksum(x[c, f] * self.demands[c] for c in range(self.num_customers)) <=
             self.capacities[f] * self.select[f] for f in range(self.num_facilities)), name='Capacity')

        m.addConstrs(
            (gp.quicksum(x[c, f] for f in range(self.num_facilities)) == 1 for
             c in range(self.num_customers)), name='Demand')
        
        self.obj1 =  gp.quicksum(self.select[f] * self.setup_costs[f] for f in range(self.num_facilities))

        return m, x

    def setObj(self, cost):
        """
        A method to set objective function

        Args:
            c (list): cost vector
        """
        obj2 = gp.quicksum(cost[k] * self.demands[c] * (self.x[c,f] )
                          for k, (c,f) in enumerate(self.edges))
        obj1 =  gp.quicksum(self.select[f] * self.setup_costs[f] for f in range(self.num_facilities))
        self._model.setObjective( obj1 + obj2)
    def PrescribedObj (self,  true_cost):
        """
        A method to evalute objective function
        when solution and cost are provided
        """ 

        self._model.update()
        self._model.setParam('TimeLimit', 60)
        self._model.optimize()
        if self._model.status==9:
            print ("Time Limit reached")
            raise Exception("Time Limit reached")

        sol = np.zeros(len(self.edges), dtype=np.float32)
        for  k, (c,f) in enumerate(self.edges):
            # print ("Customer fcility", k, "Customer", c, self.x[c, f].x)
            sol[k] = self.demands [c] * self.x[c, f].x
        secondObj = 0
        for f in range(self.num_facilities):
            secondObj += self.select[f].x * self.setup_costs[f]

        return np.dot(sol, true_cost) + secondObj
        
         

    def solve(self):
        """
        A method to solve model
        """
        self._model.update()
        self._model.setParam('TimeLimit', 60)
        self._model.optimize()
        if self._model.status==9:
            print ("Time Limit reached")
            raise Exception("Time Limit reached")

        sol = np.zeros(len(self.edges), dtype=np.float32)
        for  k, (c,f) in enumerate(self.edges):
            sol[k] = self.demands [c] * self.x[c, f].x
        secondObj = 0
        for f in range(self.num_facilities):
            # print ("fcility opened:", self.select[f].x)
            secondObj += self.select[f].x * self.setup_costs[f]
        
        # print (self._model.objVal)
        # print("Second Obj", secondObj )
        # print ("____")
        return sol, self._model.objVal
