import os
import tempfile
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="RCPSP Scheduler Dashboard (ALNS)", layout="wide")

from src.data import load_from_csv, has_dummies
from src.solver import solve_with_alns

try:
    from src.visualizations_streamlit import rcpsp_streamlit_page
    _HAS_LIB_VIZ = True
except Exception:
    _HAS_LIB_VIZ = False

st.title("RCPSP Scheduler Dashboard — ALNS Optimizer")

# --------------- Sidebar inputs ---------------
with st.sidebar:
    st.header("Inputs")
    tasks_file = st.file_uploader("Upload tasks.csv", type=["csv"], key="tasks")
    caps_file = st.file_uploader("Upload capacities.csv", type=["csv"], key="caps")
    iters = st.number_input("ALNS iterations", min_value=10, max_value=20000, value=1000, step=50)
    seed = st.number_input("Random seed", min_value=0, max_value=1_000_000, value=42, step=1)
    run_btn = st.button("Run ALNS Scheduler")

# --------------- Detect and preview data ---------------
add_dummies = False
if tasks_file and caps_file:
    try:
        st.subheader("Preview of tasks")
        tasks_preview = pd.read_csv(tasks_file)
        st.dataframe(tasks_preview.head(50))

        st.subheader("Preview of capacities")
        caps_preview = pd.read_csv(caps_file)
        st.dataframe(caps_preview.head(50))

        if not has_dummies(tasks_preview):
            add_dummies = True
            st.info("No dummy tasks found — dummy source and sink will be added automatically.")
        else:
            st.success("Detected dummy start and sink — using input as is.")

    except Exception as e:
        st.error(f"Error reading input files: {e}")
        st.stop()
else:
    st.info("Upload both tasks.csv and capacities.csv to proceed.")

# --------------- Run solver on demand and cache results ---------------
def _solve_and_cache(_tasks_file, _caps_file, _iters, _seed, _add_dummies):
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks_path = os.path.join(tmpdir, "tasks.csv")
        caps_path = os.path.join(tmpdir, "caps.csv")
        with open(tasks_path, "wb") as f:
            f.write(_tasks_file.getbuffer())
        with open(caps_path, "wb") as f:
            f.write(_caps_file.getbuffer())
        problem = load_from_csv(tasks_path, caps_path, add_dummies=_add_dummies)
    summary = solve_with_alns(problem, seed=int(_seed), iters=int(_iters))
    makespan = int(summary["makespan"])

    st.session_state["rcpsp_problem"] = problem
    st.session_state["rcpsp_summary"] = summary
    st.session_state["rcpsp_makespan"] = makespan

if run_btn and tasks_file and caps_file:
    with st.spinner("Running ALNS optimization..."):
        try:
            _solve_and_cache(tasks_file, caps_file, iters, seed, add_dummies)
            st.success(f"Optimization complete. Makespan: {st.session_state['rcpsp_makespan']}")
        except Exception as e:
            st.error(f"Scheduling failed: {e}")
            st.stop()

# --------------- Show visualization and outputs from cache ---------------
if "rcpsp_problem" in st.session_state and "rcpsp_summary" in st.session_state:
    problem = st.session_state["rcpsp_problem"]
    summary = st.session_state["rcpsp_summary"]
    makespan = st.session_state["rcpsp_makespan"]
    starts = np.asarray(summary["starts"], dtype=int)

    # Visualization
    if _HAS_LIB_VIZ:
        st.subheader("Schedule and resource utilization")
        rcpsp_streamlit_page(
            data=problem,
            starts=summary["starts"],
            usage=summary.get("usage"),
            capacities=summary.get("capacities"),
            title=f"ALNS Schedule (Makespan={makespan})",
            time_unit="t",
        )
    else:
        st.warning("Visualization module not found (src.visualizations_streamlit).")

    # KPIs
    st.subheader("Performance metrics")
    col1, col2 = st.columns(2)
    col1.metric("Makespan", f"{makespan}")
    util = summary.get("total_utilization")
    if util is not None:
        col2.metric("Average utilization", f"{util * 100:.1f}%")

    # Schedule table and download
    st.subheader("Schedule table")
    dur = np.asarray(problem.durations, dtype=int)
    sched_df = pd.DataFrame({
        "Task ID": range(len(starts)),
        "Start": starts,
        "Finish": starts + dur,
        "Duration": dur
    })
    for i, rname in enumerate(problem.resource_names):
        sched_df[f"Needs_{rname}"] = problem.needs[:, i]
    st.dataframe(sched_df)

    csv = sched_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download schedule CSV",
        data=csv,
        file_name="schedule.csv",
        mime="text/csv",
        use_container_width=True
    )
else:
    st.caption("Upload files and click Run ALNS Scheduler to generate a schedule.")

st.caption("This dashboard runs your ALNS-based RCPSP solver and visualizes the optimized schedule.")
