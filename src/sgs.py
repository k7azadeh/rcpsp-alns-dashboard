from __future__ import annotations
from typing import Sequence, Tuple
import numpy as np
from src.data import ProblemData
import logging

log = logging.getLogger(__name__)




def schedule_order(order: Sequence[int], data: ProblemData) -> Tuple[np.ndarray, np.ndarray]:
    """Serial SGS for time-varying capacities with an eligible-set check.

    The function schedules jobs in the given priority ``order`` at the earliest feasible start that respects
    precedence constraints and renewable, time-dependent capacities.

    Args:
        order: A feasible (topological) priority list of job indices.
        data: Problem data bundle with durations, needs, precedence, and capacities.

    Returns:
        (start_times, used_eff):
            start_times: integer array of size n_jobs with job start periods (-1 if unset).
            used_eff: resource-usage matrix cropped to the final makespan.
    """


    n = len(data.durations)
    T = data.cap_tm.shape[0]

    start_times = -np.ones(n, int)
    finish_times = -np.ones(n, int)
    used = np.zeros_like(data.cap_tm, int)
    scheduled = np.zeros(n, bool)
    remaining = list(order)

    present = set(order)  # only enforce precedence among jobs present in this partial sequence

    while remaining:
        idx = None
        for i, j in enumerate(remaining):
            preds_present = [p for p in data.predecessors[j] if p in present]
            if all(scheduled[p] for p in preds_present):
                idx = i
                break
        if idx is None:
            raise RuntimeError("No eligible job found – precedence cycle or invalid order (in partial order handling).")

        j = remaining.pop(idx)

        preds_present = [p for p in data.predecessors[j] if p in present]
        est = max((finish_times[p] for p in preds_present), default=0)
        if est < 0:
            est = 0

        d = int(data.durations[j])
        if d == 0:
            start_times[j] = est
            finish_times[j] = est
            scheduled[j] = True
            continue

        req = data.needs[j]
        t = est
        while True:
            if t + d > T:
                extra = t + d - T
                pad = np.repeat(data.cap_tm[-1:], extra, axis=0)
                data.cap_tm = np.vstack([data.cap_tm, pad])
                used = np.vstack([used, np.zeros_like(pad)])
                T = data.cap_tm.shape[0]

            feasible = True
            for tau in range(t, t + d):
                if np.any(used[tau] + req > data.cap_tm[tau]):
                    feasible = False
                    break
            if feasible:
                used[t:t + d] += req
                start_times[j] = t
                finish_times[j] = t + d
                scheduled[j] = True
                break
            t += 1

    makespan = int(finish_times.max(initial=0))
    used_eff = used[:makespan]
    return start_times, used_eff

