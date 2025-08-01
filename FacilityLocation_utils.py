import torch
import numpy as np
from einops import einsum
from einops import repeat

class cost_transform:
    def __init__(self, optmodel) -> None:
        demands = np.array(optmodel.demands)
        num_facilities = optmodel.num_facilities
        repeated_demands = np.repeat( demands, num_facilities )
        setupcosts= np.array ( optmodel.setup_costs  )
        self.repeated_demands =  torch.from_numpy (repeated_demands).float()
        self.setupcosts =  torch.from_numpy (setupcosts).float()

        pass
    def apply (self, pred_cost):
        # num_zeros = self.num_totaledges - pred_cost.size(1)
        op = pred_cost * self.repeated_demands
        setup_costs_expanded = repeat( self.setupcosts, 'd -> b d', b= len (pred_cost)) 
        op = torch.cat([op, setup_costs_expanded ], dim=1)
        return op

class sol_transform:
    def __init__(self, optmodel) -> None:
        demands = np.array(optmodel.demands)
        self.num_facilities = optmodel.num_facilities
        repeated_demands = np.repeat( demands, self.num_facilities )
        setupcosts= np.array ( optmodel.setup_costs  )
        self.repeated_demands =  torch.from_numpy (repeated_demands).float()
        self.setupcosts =  torch.from_numpy (setupcosts).float()
        pass
    def apply (self,  pred_sol):
        # num_zeros = self.num_totaledges - pred_cost.size(1)
        clipped_sol = pred_sol [:, :-self.num_facilities]
        op = clipped_sol * self.repeated_demands
        return op
