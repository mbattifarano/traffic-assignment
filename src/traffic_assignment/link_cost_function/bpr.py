from dataclasses import dataclass
from typing import Optional

import numpy as np
from toolz import curry

from .base import LinkCostFunction
from ..utils import value_or_default, ArrayOrFloat

import numba


@dataclass(frozen=True)
class BPRLinkCostFunction(LinkCostFunction):
    free_flow_travel_time: ArrayOrFloat
    capacity: ArrayOrFloat
    alpha: ArrayOrFloat = 0.15
    beta: ArrayOrFloat = 4

    def link_cost(self, link_flow: ArrayOrFloat) -> np.ndarray:
        return _bpr(self.alpha, self.beta, self.free_flow_travel_time,
                    self.capacity, link_flow)

    def integral_link_cost(self, link_flow: ArrayOrFloat) -> np.ndarray:
        return _i_bpr(self.alpha, self.beta, self.free_flow_travel_time,
                      self.capacity, link_flow)

    def derivative_link_cost(self, link_flow: ArrayOrFloat) -> np.ndarray:
        return _d_bpr(self.alpha, self.beta, self.free_flow_travel_time,
                      self.capacity, link_flow)


@dataclass(frozen=True)
class BPRMarginalLinkCostFunction(BPRLinkCostFunction):
    fleet_link_flow: Optional[np.ndarray] = None

    def link_cost(self, link_flow: np.ndarray) -> np.ndarray:
        fleet_link_flow = value_or_default(self.fleet_link_flow, link_flow)
        return _marginal_bpr(self.alpha, self.beta, self.free_flow_travel_time,
                             self.capacity, link_flow, fleet_link_flow)
        #travel_time = _bpr(self.alpha, self.beta, self.free_flow_travel_time,
        #                   self.capacity, link_flow)
        #travel_time_gradient = _d_bpr(self.alpha, self.beta,
        #                              self.free_flow_travel_time,
        #                              self.capacity, link_flow)
        #return travel_time + fleet_link_flow * travel_time_gradient

    def integral_link_cost(self, link_flow: np.ndarray) -> np.ndarray:
        return _marginal_i_bpr(self.alpha, self.beta, self.free_flow_travel_time,
                               self.capacity, link_flow)
        #cost = _bpr(self.alpha, self.beta, self.free_flow_travel_time,
        #            self.capacity, link_flow)
        #return link_flow * cost

    def derivative_link_cost(self, link_flow: ArrayOrFloat) -> np.ndarray:
        raise NotImplementedError()


@numba.jit(nopython=True, nogil=True, parallel=True)
def _marginal_bpr(alpha, beta, free_flow, capacity, link_flow, fleet_link_flow):
    travel_time = _bpr(alpha, beta, free_flow, capacity, link_flow)
    travel_time_gradient = _d_bpr(alpha, beta, free_flow, capacity, link_flow)
    return travel_time + fleet_link_flow * travel_time_gradient


@numba.jit(nopython=True, nogil=True, parallel=True)
def _marginal_i_bpr(alpha, beta, free_flow, capacity, link_flow):
    return link_flow * _bpr(alpha, beta, free_flow, capacity, link_flow)


@numba.jit(nopython=True, nogil=True, parallel=True)
def _bpr(alpha: float, beta: float, free_flow: np.ndarray, capacity: np.ndarray,
         link_flow: np.ndarray) -> np.ndarray:
    """The Bureau of Public Road (BPR) link cost function."""
    return (
        free_flow * (1.0 + alpha * (link_flow / capacity)**beta)
    )


@numba.jit(nopython=True, nogil=True, parallel=True)
def _d_bpr(alpha: float, beta: float, free_flow: np.ndarray,
           capacity: np.ndarray, link_flow: np.ndarray) -> np.ndarray:
    """The gradient of the Bureau of Public Road (BPR) link cost function."""
    return (
        free_flow * alpha * beta * link_flow**(beta - 1) / capacity**beta
    )


@numba.jit(nopython=True, nogil=True, parallel=True)
def _i_bpr(alpha: float, beta: float, free_flow: np.ndarray,
           capacity: np.ndarray, link_flow: np.ndarray) -> np.ndarray:
    """The integral of the Bureau of Public Road (BPR) link cost function."""
    x0 = np.zeros_like(link_flow)
    Fx = _antiderivative_bpr(alpha, beta, free_flow, capacity, link_flow)
    F0 = _antiderivative_bpr(alpha, beta, free_flow, capacity, x0)
    return Fx - F0


@numba.jit(nopython=True, nogil=True, parallel=True)
def _antiderivative_bpr(alpha: float, beta: float, free_flow: np.ndarray,
                        capacity: np.ndarray, link_flow: np.ndarray) -> np.ndarray:
    """The antiderivative of the BPR."""
    return (
        free_flow * link_flow
        + (free_flow * alpha * link_flow**(beta + 1)) / ((beta+1) * capacity**beta)
    )

