from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
import logging


log = logging.getLogger(__name__)

@dataclass
class ProblemData:
    """Container for static RCPSP data.
    
    Attributes:
        durations: Array of job durations of shape (n_jobs,).
        successors: Adjacency list of successors for each job.
        predecessors: Adjacency list of predecessors for each job.
        needs: Demand matrix of shape (n_jobs, n_resources), integer units per period.
        cap_tm: Time-varying resource capacity matrix of shape (T, n_resources).
        resource_names: Names of the resources, in the same order as needs/cap_tm columns.
    """
    durations: np.ndarray
    successors: List[List[int]]
    predecessors: List[List[int]]
    needs: np.ndarray
    cap_tm: np.ndarray
    resource_names: List[str]

def has_dummies(tasks_df: pd.DataFrame) -> bool:
    """Check whether a tasks DataFrame already includes dummy start/sink nodes."""
    if "Task ID" not in tasks_df.columns or "Predecessors" not in tasks_df.columns:
        return False

    task_ids = set(tasks_df["Task ID"])
    preds = set()
    for p in tasks_df["Predecessors"].dropna():
        preds.update(int(x) for x in str(p).split(";") if x.strip().isdigit())

    has_start_dummy = (
        0 in task_ids
        and str(tasks_df.loc[tasks_df["Task ID"] == 0, "Predecessors"].iloc[0])
        .strip()
        .lower() in ["", "nan", "none"]
        and tasks_df.loc[tasks_df["Task ID"] == 0, "Duration"].iloc[0] == 0
    )

    has_sink_dummy = (
        any(tasks_df["Duration"] == 0)
        and tasks_df["Task ID"].max() == len(tasks_df) - 1
    )

    return has_start_dummy or has_sink_dummy

def load_from_csv(tasks_csv: str, caps_csv: str, add_dummies: bool = True) -> ProblemData:
    """Load a time-varying RCPSP instance from CSV files.

    Args:
        tasks_csv: Path to tasks CSV with columns: "Task ID", "Duration", "Predecessors", and one or more
            resource need columns named "Needs_<res>".
        caps_csv: Path to capacities CSV with column "Time" and matching capacity columns "Cap_<res>".

    Returns:
        A fully constructed :class:`ProblemData` instance.

    Raises:
        ValueError: If required columns are missing or resources mismatch between files.
    """

    # Read task and capacity data
    tasks_df = pd.read_csv(tasks_csv)
    caps_df = pd.read_csv(caps_csv)

    # --- Auto-detect dummies ---

    if has_dummies(tasks_df):
        add_dummies = False
        logging.getLogger("rcpsp.data").info("Detected dummy start/sink — using input as is.")
    else:
        add_dummies = True
        logging.getLogger("rcpsp.data").info("No dummies detected — adding dummy source/sink automatically.")

    # Check required columns for tasks
    req_cols = {"Task ID", "Duration", "Predecessors"}
    if not req_cols.issubset(tasks_df.columns):
        raise ValueError(f"Missing columns in tasks.csv: {req_cols - set(tasks_df.columns)}")


    # Identify resource columns
    needs_cols = [c for c in tasks_df.columns if c.startswith("Needs_")]
    if not needs_cols:
        raise ValueError("tasks.csv must include at least one 'Needs_' column")


    resource_names = [c[len("Needs_"): ] for c in needs_cols]


    # Ensure task IDs are contiguous integers starting from 0
    tasks_df = tasks_df.sort_values("Task ID").reset_index(drop=True)
    if not np.array_equal(tasks_df["Task ID"].to_numpy(), np.arange(len(tasks_df))):
        raise ValueError("Task IDs must be contiguous integers starting from 0")


    # Extract core arrays
    durations = tasks_df["Duration"].to_numpy(int)
    needs = tasks_df[needs_cols].to_numpy(int)


    # Build predecessor-successor relationships
    preds = []
    for s in tasks_df["Predecessors"].fillna(""):
        if isinstance(s, str) and s.strip():
            p = [int(x) for x in s.split(";") if x.strip()]
        else:
            p = []
        preds.append(p)


    n = len(tasks_df)
    succs = [[] for _ in range(n)]
    for i, pl in enumerate(preds):
        for p in pl:
            succs[p].append(i)


    # Validate capacities CSV
    if "Time" not in caps_df:
        raise ValueError("capacities.csv must contain a 'Time' column")


    cap_cols = [c for c in caps_df.columns if c.startswith("Cap_")]
    if not cap_cols:
        raise ValueError("capacities.csv must have at least one 'Cap_' column")


    cap_res = [c[len("Cap_"): ] for c in cap_cols]
    if set(cap_res) != set(resource_names):
        raise ValueError(f"Mismatch between task and capacity resources: {cap_res} vs {resource_names}")


    # Sort by time and normalize the capacity horizon
    caps_df = caps_df.sort_values("Time").drop_duplicates(subset=["Time"]).reset_index(drop=True)

    # Compute a safe upper bound on makespan (sum of durations of real tasks)
    safe_horizon = int(tasks_df["Duration"].sum())

    # Ensure Time starts at 0 and has no gaps; extend to max(current_horizon, safe_horizon)
    res_cols = [c for c in caps_df.columns if c.startswith("Cap_")]
    max_needed_t = max(safe_horizon, int(caps_df["Time"].max()))
    caps_df = (
        caps_df.set_index("Time")
            .reindex(range(0, max_needed_t + 1))    # fill missing rows
            .ffill()                                 # forward-fill capacities
            .reset_index()
            .rename(columns={"index": "Time"})
    )

    # Now create the capacity matrix
    cap_tm = caps_df[res_cols].to_numpy(int)
    if add_dummies:
        n_real = len(durations)
        R = needs.shape[1]

        # Initialize augmented arrays
        durations_aug = np.zeros(n_real + 2, dtype=int)
        needs_aug = np.zeros((n_real + 2, R), dtype=int)

        # Copy real task info shifted by +1
        durations_aug[1:1 + n_real] = durations
        needs_aug[1:1 + n_real, :] = needs

        # Rebuild preds/succs with shifted indices
        preds_aug = [[] for _ in range(n_real + 2)]
        succs_aug = [[] for _ in range(n_real + 2)]

        # Map real tasks 0..n-1 → 1..n
        for j_old in range(n_real):
            j_new = j_old + 1
            p_old_list = preds[j_old]
            if p_old_list:
                p_new_list = [p_old + 1 for p_old in p_old_list]
            else:
                # attach dummy start (0) as predecessor if none
                p_new_list = [0]
            preds_aug[j_new] = p_new_list
            for p_new in p_new_list:
                succs_aug[p_new].append(j_new)

        # Identify terminal tasks (no successors)
        terminals = [j_old + 1 for j_old in range(n_real)
                     if len(succs_aug[j_old + 1]) == 0]

        # Attach dummy sink (id = n_real + 1)
        sink_id = n_real + 1
        preds_aug[sink_id] = terminals
        for t in terminals:
            succs_aug[t].append(sink_id)

        # Replace originals with augmented versions
        durations = durations_aug
        needs = needs_aug
        preds = preds_aug
        succs = succs_aug

    # Basic dataset summary
    n_jobs = len(durations)
    rnames = list(resource_names)
    cap_const = (cap_tm.max(axis=0) == cap_tm.min(axis=0)).all()
    log.info("Loaded problem: %d jobs, %d renewable resources %s",
            n_jobs, len(rnames), rnames)
    log.info("Capacity horizon: T=%d (constant=%s)", cap_tm.shape[0], cap_const)

    # If you remap Task IDs here, log that too
    try:
        if "Task ID" in locals() or "tasks_df" in globals():
            pass  # ignore if your implementation differs
    except Exception:
        pass
    
    return ProblemData(durations, succs, preds, needs, cap_tm, resource_names)
