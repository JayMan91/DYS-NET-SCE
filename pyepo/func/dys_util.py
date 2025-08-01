import numpy as np
import scipy.sparse as sps
from warnings import warn
import torch
# from ._optimize import OptimizeWarning
# from scipy.optimize._remove_redundancy import (
#     _remove_redundancy_svd, _remove_redundancy_pivot_sparse,
#     _remove_redundancy_pivot_dense, _remove_redundancy_id
#     )
from collections import namedtuple

class standardizeLP:
    def __init__(self,A_ub, b_ub, A_eq, b_eq) -> None:
        """
        Given LP of the following form
            min c @ x
            A_ub @ x <= b_ub
            A_eq @ x == b_eq
        """
        self.A_ub, self.b_ub, self.A_eq, self.b_eq  = A_ub, b_ub, A_eq, b_eq

    def getAb(self):
        '''
        Return the (a,b) of standard form:

            A @ x == b
                x >= 0
        by adding slack variables and making variable substitutions as necessary.
        The block matrix is of the following structure:
        

        ----------------------------
                    |
         A_ub       |   I    
                    |
                    |
        ----------------------------
                    |
                    |
         A_eq       |    0
                    |
        ----------------------------

        
        '''
        A_ub, b_ub, A_eq, b_eq = self.A_ub, self.b_ub, self.A_eq, self.b_eq
        if sps.issparse(A_eq):
            self.sparse = True
            A_eq = sps.csr_matrix(A_eq)
            A_ub = sps.csr_matrix(A_ub)

            def hstack(blocks):
                return sps.hstack(blocks, format="csr")

            def vstack(blocks):
                return sps.vstack(blocks, format="csr")

            zeros = sps.csr_matrix
            eye = sps.eye
        else:
            self.sparse = False
            hstack = np.hstack
            vstack = np.vstack
            zeros = np.zeros
            eye = np.eye
        
        # # Variables lbs and ubs (see below) may be changed, which feeds back into
        # # bounds, so copy.
        # bounds = np.array(bounds, copy=True)

        # # modify problem such that all variables have only non-negativity bounds
        # lbs = bounds[:, 0]
        # ubs = bounds[:, 1]
        m_ub, n_ub = A_ub.shape
        m_eq, n_eq = A_eq.shape
        assert n_ub == n_eq
        self.m_ub, self.n_ub, self.m_eq, self.n_eq = m_ub, n_ub, m_eq, n_eq

        # lb_none = np.equal(lbs, -np.inf)
        # ub_none = np.equal(ubs, np.inf)
        # lb_some = np.logical_not(lb_none)
        # ub_some = np.logical_not(ub_none)

        # # unbounded below: substitute xi = -xi' (unbounded above)
        # # if -inf <= xi <= ub, then -ub <= -xi <= inf, so swap and invert bounds
        # l_nolb_someub = np.logical_and(lb_none, ub_some)
        # i_nolb = np.nonzero(l_nolb_someub)[0]
        # lbs[l_nolb_someub], ubs[l_nolb_someub] = (
        #     -ubs[l_nolb_someub], -lbs[l_nolb_someub])
        # lb_none = np.equal(lbs, -np.inf)
        # ub_none = np.equal(ubs, np.inf)
        # lb_some = np.logical_not(lb_none)
        # ub_some = np.logical_not(ub_none)
        # c[i_nolb] *= -1
        # if x0 is not None:
        #     x0[i_nolb] *= -1
        # if len(i_nolb) > 0:
        #     if A_ub.shape[0] > 0:  # sometimes needed for sparse arrays... weird
        #         A_ub[:, i_nolb] *= -1
        #     if A_eq.shape[0] > 0:
        #         A_eq[:, i_nolb] *= -1

        # # upper bound: add inequality constraint
        # i_newub, = ub_some.nonzero()
        # ub_newub = ubs[ub_some]
        # n_bounds = len(i_newub)
        # if n_bounds > 0:
        #     shape = (n_bounds, A_ub.shape[1])
        #     if sparse:
        #         idxs = (np.arange(n_bounds), i_newub)
        #         A_ub = vstack((A_ub, sps.csr_matrix((np.ones(n_bounds), idxs),
        #                                             shape=shape)))
        #     else:
        #         A_ub = vstack((A_ub, np.zeros(shape)))
        #         A_ub[np.arange(m_ub, A_ub.shape[0]), i_newub] = 1
        #     b_ub = np.concatenate((b_ub, np.zeros(n_bounds)))
        #     b_ub[m_ub:] = ub_newub

        A1 = vstack((A_ub, A_eq))
        b = np.concatenate((b_ub, b_eq))




        # c = np.concatenate((c, np.zeros((A_ub.shape[0],))))
        # if x0 is not None:
        #     x0 = np.concatenate((x0, np.zeros((A_ub.shape[0],))))
        # # unbounded: substitute xi = xi+ + xi-
        # l_free = np.logical_and(lb_none, ub_none)
        # i_free = np.nonzero(l_free)[0]
        # n_free = len(i_free)
        # c = np.concatenate((c, np.zeros(n_free)))
        # if x0 is not None:
        #     x0 = np.concatenate((x0, np.zeros(n_free)))
        # A1 = hstack((A1[:, :n_ub], -A1[:, i_free]))
        # c[n_ub:n_ub+n_free] = -c[i_free]
        # if x0 is not None:
        #     i_free_neg = x0[i_free] < 0
        #     x0[np.arange(n_ub, A1.shape[1])[i_free_neg]] = -x0[i_free[i_free_neg]]
        #     x0[i_free[i_free_neg]] = 0

        # add slack variables
        A2 = vstack([eye(m_ub), zeros((m_eq,m_ub))])

        A = hstack([A1, A2])

        # # lower bound: substitute xi = xi' + lb
        # # now there is a constant term in objective
        # i_shift = np.nonzero(lb_some)[0]
        # lb_shift = lbs[lb_some].astype(float)
        # c0 += np.sum(lb_shift * c[i_shift])
        # if sparse:
        #     b = b.reshape(-1, 1)
        #     A = A.tocsc()
        #     b -= (A[:, i_shift] * sps.diags(lb_shift)).sum(axis=1)
        #     b = b.ravel()
        # else:
        #     b -= (A[:, i_shift] * lb_shift).sum(axis=1)
        # if x0 is not None:
        #     x0[i_shift] -= lb_shift

        return A, b

    def transformC(self, c):
        '''
        Return the c for the standard form:
            min c @ x

            A @ x == b
                x >= 0
        
        '''
        m_ub, n_ub, m_eq, n_eq = self.m_ub, self.n_ub, self.m_eq, self.n_eq
        n_batch = len(c)

        return torch.cat ([ c, torch.zeros(n_batch, m_ub) ], -1)

        # if self.sparse:
        #     def hstack(blocks):
        #         return sps.hstack(blocks, format="csr")

        #     def vstack(blocks):
        #         return sps.vstack(blocks, format="csr")

        #     zeros = sps.csr_matrix
        #     eye = sps.eye
        # else:
        #     hstack = np.hstack
        #     vstack = np.vstack
        #     zeros = np.zeros
        #     eye = np.eye

        # return  np.concatenate((c, zeros(m_ub)))

    def transformsolution(self, x):
        '''
        Turn the x solved in for the standard form to the x of the original problem
        
        '''
        m_ub, n_ub, m_eq, n_eq = self.m_ub, self.n_ub, self.m_eq, self.n_eq
        return x[:, :n_eq]

        # if self.sparse:
        #     def hstack(blocks):
        #         return sps.hstack(blocks, format="csr")

        #     def vstack(blocks):
        #         return sps.vstack(blocks, format="csr")

        #     zeros = sps.csr_matrix
        #     eye = sps.eye
        # else:
        #     hstack = np.hstack
        #     vstack = np.vstack
        #     zeros = np.zeros
        #     eye = np.eye

        
    def transformsgradient(self, dx):
        '''
        Turn the dx solved found by differentiating the hsd to the derivative of the original problem
        
        '''
        m_ub, n_ub, m_eq, n_eq = self.m_ub, self.n_ub, self.m_eq, self.n_eq

        # if self.sparse:
        #     def hstack(blocks):
        #         return sps.hstack(blocks, format="csr")

        #     def vstack(blocks):
        #         return sps.vstack(blocks, format="csr")

        #     zeros = sps.csr_matrix
        #     eye = sps.eye
        # else:
        #     hstack = np.hstack
        #     vstack = np.vstack
        #     zeros = np.zeros
        #     eye = np.eye

        return  dx[:n_eq, :n_eq]

def get_AbCd(optmodel):
    '''
    Works if the constraint of the Optmodel is in the form of
    Ax ==b 
    Cx <= d
    Esnure that the name are 'A', 'b', 'C' and 'd'
    '''
    has_A = False
    has_C =  False

    if hasattr(optmodel, 'A'):
        if optmodel.A is not None:
            A = (optmodel.A).astype(float)
            b = (optmodel.b).astype(float)
            has_A = True


    if hasattr(optmodel, 'C'):
        if (optmodel.C is not None):
            C =  (optmodel.C).astype(float)
            d =  (optmodel.d).astype(float)
            has_C = True
    if has_A and has_C:
        m_ub, n_ub = C.shape
        m_eq, n_eq = A.shape
        if n_ub != n_eq:
            raise Exception("Number of variables in the equality constraint matrix does not match."
                            "with the number of variables in the inequality constraint matrix does not match")
        n = n_ub = n_eq
    elif not has_A:
        m_ub, n = C.shape
    
        A = np.zeros((0, n)).astype(float)
        b = np.array([], dtype=float)
        m_eq = 0    
    elif not has_C:
        m_eq, n = A.shape
        C = np.zeros((0, n)).astype(float)
        d = np.array([], dtype=float)
        m_ub = 0       
    else:
        raise Exception("The problem is (trivially) unbounded "
                    "because there are no non-trivial constraints.")


    return (A, b , C, d)





