from __future__ import annotations
from typing import Sequence
from .data import ProblemData
from .sgs import schedule_order
import numpy as np




class RcpspState:
    """ALNS state representation for RCPSP.

    Stores a job priority ``order`` and the corresponding schedule and resource usage computed by the SGS.
    The objective for this example is the makespan (project completion time).
    """

    def __init__(self, order: Sequence[int], data: ProblemData):
        """Create a state.

        Args:
            order: Current job sequence.
            data: Immutable problem data shared across states.
        """
        self.order = list(order)
        self.data = data
        self.starts, self.used = schedule_order(self.order, self.data)


    def copy(self):
        """Return a shallow copy of the state (shares the same :class:`ProblemData`)."""
        return RcpspState(self.order.copy(), self.data)


    def objective(self) -> float:
        """
        Makespan objective with a big penalty if any positive-duration job
        is left unscheduled (start == -1). This prevents partial states
        from appearing better than feasible full schedules.
        """
        starts = self.starts
        dur = self.data.durations

        # If any real job is unscheduled, return a large penalty
        mask = (dur > 0) & (starts < 0)
        if np.any(mask):
            # Add a tiny tie-breaker so fewer missing jobs is preferred
            return 1e12 + float(mask.sum())

        return float((starts + dur).max(initial=0))
