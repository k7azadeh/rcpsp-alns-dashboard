from __future__ import annotations
import numpy as np
from typing import List
from alns import ALNS
from alns.stop import MaxIterations
from alns.accept import SimulatedAnnealing
from alns.select import RouletteWheel
from src.state import RcpspState
from src.operators import random_removal, random_insert, non_peak_removal, justify_repair
from src.data import ProblemData




def _topological_lrp_order(data: ProblemData) -> list[int]:
    """Topological order that prioritizes nodes with largest tail (longest remaining path)."""
    n = len(data.durations)
    indeg = [len(p) for p in data.predecessors]
    # compute tail (longest distance to sink in precedence DAG, ignoring resources)
    tail = [0]*n
    # standard topo to get reverse order
    q = [i for i,d in enumerate(indeg) if d == 0]
    topo = []
    idx = 0
    while idx < len(q):
        u = q[idx]; idx += 1
        topo.append(u)
        for v in data.successors[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    # reverse DP for tail
    for u in reversed(topo):
        if data.successors[u]:
            tail[u] = int(data.durations[u]) + max(tail[v] for v in data.successors[u])
        else:
            tail[u] = int(data.durations[u])

    # rebuild indegrees for the prioritized Kahn
    indeg = [len(p) for p in data.predecessors]
    ready = [i for i,d in enumerate(indeg) if d == 0]
    order = []
    # pick the ready node with largest tail each step
    while ready:
        # argmax by tail, tiebreak by smaller id for determinism
        j = max(ready, key=lambda x: (tail[x], -x))
        ready.remove(j)
        order.append(j)
        for v in data.successors[j]:
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
    return order




def solve_with_alns(data: ProblemData, seed: int = 42, iters: int = 1000):
    """Run ALNS to minimize makespan under time-varying capacities.

    Args:
        data: Problem data.
        seed: Random seed for reproducibility.
        iters: Number of ALNS iterations.

    Returns:
        A summary dictionary with keys:
        - makespan
        - order
        - starts
        - resource_names
        - usage (time × resource matrix)
        - capacities (time × resource matrix)
        - utilization (fraction)
        - total_utilization (overall average)
        - durations
        - needs
    """
    rng = np.random.default_rng(seed)

    # --- 1. Initial feasible state
    init_state = RcpspState(_topological_lrp_order(data), data)

    # --- 2. Configure ALNS
    alns = ALNS(rng)
    op_rand = random_removal(rng, 0.2)
    op_np   = non_peak_removal(rng, 0.2)
    op_rep  = random_insert(rng)
    alns.add_destroy_operator(op_rand, name="random_removal")
    alns.add_destroy_operator(op_np,   name="non_peak_removal")
    alns.add_repair_operator(op_rep,   name="random_insert")
    alns.add_repair_operator(justify_repair(rng), name="justify")

    # --- 3. Acceptance and stopping
    accept = SimulatedAnnealing.autofit(
        init_obj=init_state.objective(),
        worse=0.05,
        accept_prob=0.8,
        num_iters=iters,
        method="exponential"
    )
    stop = MaxIterations(iters)
    select = RouletteWheel(
        scores=[1, 0.8, 0.5, 0.3],
        decay=0.8,
        num_destroy=len(alns.destroy_operators),
        num_repair=len(alns.repair_operators),
    )

    # --- 4. Run ALNS
    result = alns.iterate(
        initial_solution=init_state,
        op_select=select,
        stop=stop,
        accept=accept,
    )

    best: RcpspState = result.best_state
    starts = np.asarray(best.starts, dtype=int)
    dur = np.asarray(data.durations, dtype=int)
    makespan = int((starts + dur).max(initial=0))
    n_tasks = len(dur)
    R = len(data.resource_names)

    # --- 5. Compute usage per resource over time
    used = np.zeros((makespan, R), dtype=int)
    for j in range(1, n_tasks - 1):  # exclude dummy start/sink
        if dur[j] > 0 and starts[j] >= 0:
            used[starts[j]:starts[j] + dur[j]] += data.needs[j]

    # --- 6. Slice capacities to horizon
    caps = data.cap_tm[:makespan, :]

    # --- 7. Compute utilization stats
    utilization = np.divide(
        used, caps, out=np.zeros_like(used, dtype=float), where=(caps > 0)
    )
    total_utilization = float(np.mean(utilization[caps > 0]))

    # --- 8. Build rich summary dictionary
    summary = {
        "makespan": makespan,
        "order": best.order,
        "starts": starts.tolist(),
        "resource_names": data.resource_names,
        "usage": used.tolist(),           # time × resource usage
        "capacities": caps.tolist(),      # time × resource capacity
        "utilization": utilization.tolist(),
        "total_utilization": total_utilization,
        "durations": dur.tolist(),
        "needs": data.needs.tolist(),
    }

    return summary