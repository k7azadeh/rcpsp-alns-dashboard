import random
from typing import Sequence
import numpy as np
from src.state import RcpspState
from src.sgs import schedule_order





def random_removal(rng: random.Random, frac: float = 0.2):
    """Randomly remove a fraction of jobs from the sequence.

    Args:
        rng: Python random generator.
        frac: Fraction of jobs to remove (0 < frac ≤ 1).

    Returns:
        A destroy operator compatible with ``alns`` that returns a new partial state.
    """

    def destroy(state: RcpspState, rng_, **_):
        n = len(state.order)
        if n == 0:
            return state

        k = max(1, int(frac * n))
        if k>n:
            k=n
        removed = set(rng_.choice(state.order, size=k, replace=False))
        keep = [j for j in state.order if j not in removed]
        return RcpspState(keep, state.data)
    return destroy


def non_peak_removal(rng: random.Random, frac: float = 0.2):
    """Remove jobs that sit in low-utilization windows relative to capacity.

    Args:
        rng: Python random generator (unused but kept for signature consistency).
        frac: Fraction of jobs to remove.

    Returns:
        A destroy operator prioritizing removal of jobs scheduled in poorly utilized windows.
    """

    def destroy(state: RcpspState, rng_, **_):
        n = len(state.order)
        if n == 0:
            return state

        starts, used = schedule_order(state.order, state.data)
        cap = state.data.cap_tm
        scores = []

        for j, s in enumerate(starts):
            d = int(state.data.durations[j])
            if d <= 0 or s < 0:
                scores.append((j, 0.0))
                continue
            win = slice(int(s), int(s) + d)
            cap_win = cap[win]
            used_win = used[win] if used.shape[0] >= int(s) + d else used
            frac_used = (used_win / np.maximum(cap_win, 1)).mean() if cap_win.size else 0.0
            scores.append((j, float(frac_used)))

        scores.sort(key=lambda x: x[1]) # lowest utilization first
        k = max(1, int(frac * n))
        k = min(k, len(scores))
        to_remove = set(j for j, _ in scores[:k])
        keep = [j for j in state.order if j not in to_remove]
        return RcpspState(keep, state.data)
    return destroy


def random_insert(rng: random.Random):
    """Randomly insert removed jobs back into the sequence and reschedule.

    Args:
        rng: Python random generator.

    Returns:
        A repair operator compatible with ``alns``.
    """

    def repair(state: RcpspState, rng_, removed: Sequence[int] | None = None, **_):
        if not removed:
            return state
        order = state.order.copy()
        for j in removed:
            pos = rng.randrange(len(order) + 1)
            order.insert(pos, j)
        return RcpspState(order, state.data)
    return repair


def justify(state: RcpspState) -> RcpspState:
    """Right-justify then reschedule to compact the sequence.

    This is a simple compaction heuristic: sort tasks by decreasing finish times and rebuild the state.

    Args:
        state: Current state.

    Returns:
        A new state with a re-ordered sequence.
    """

    starts, _ = schedule_order(state.order, state.data)
    finish = starts + state.data.durations
    perm = sorted(range(len(state.order)), key=lambda j: (finish[j], starts[j]), reverse=True)
    order_r = [state.order[j] for j in perm]
    return RcpspState(order_r, state.data)

def justify_repair(rng):
    """Wrap justify() as a repair operator compatible with ALNS."""
    def repair(state, rng_, **_):
        return justify(state)
    return repair