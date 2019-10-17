from dataclasses import dataclass

import cvxpy as cp

from .utils import Constants, Variables
from traffic_assignment.utils import Timer
from warnings import warn


@dataclass
class UpperControlRatio:
    constants: Constants
    variables: Variables
    tolerance: float = 1e-5

    def objective(self):
        return cp.sum(self.variables.fleet_demand)

    @property
    def sense(self):
        return cp.Minimize

    def constraints(self):
        v = self.variables
        c = self.constants
        return [
            v.fleet_path_flow >= 0.0,
            v.user_path_flow >= 0.0,
            v.user_link_flow + v.fleet_link_flow == c.target_link_flow,
            c.fleet_link_path_incidence() @ v.fleet_path_flow == v.fleet_link_flow,
            c.user_link_path_incidence() @ v.user_path_flow == v.user_link_flow,
            c.fleet_path_od_incidence() @ v.fleet_path_flow == v.fleet_demand,
            c.user_path_od_incidence() @ v.user_path_flow == v.user_demand,
            v.user_demand + v.fleet_demand == c.total_demand,
        ]

    def problem(self):
        timer = Timer()
        print("Creating problem.")
        timer.start()
        p = cp.Problem(
            self.sense(self.objective()),
            self.constraints()
        )
        print(f"Problem created in {timer.time_elapsed():0.2f} (s).")
        return p


class LowerControlRatio(UpperControlRatio):
    @property
    def sense(self):
        return cp.Maximize

