import os
import tempfile
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="🚀 Kaveh Scheduler", layout="wide")

from src.data import load_from_csv, has_dummies
from src.solver import solve_with_alns

try:
    from src.visualizations_streamlit import rcpsp_streamlit_page
    _HAS_LIB_VIZ = True
except Exception:
    _HAS_LIB_VIZ = False

st.title("🚀 Kaveh Scheduler")
# Choose resource names for the templates
# ---------- Helper functions ----------
def make_example_tasks_df(resource_names):
    base = {
        "Task ID": [0, 1, 2],
        "Duration": [3, 2, 4],
        "Predecessors": ["", "", "0;1"],
    }
    df = pd.DataFrame(base)
    # Simple pattern so each resource gets nonzero usage somewhere
    for i, r in enumerate(resource_names):
        if i == 0:
            df[f"Needs_{r}"] = [2, 0, 2]
        elif i == 1:
            df[f"Needs_{r}"] = [0, 1, 2]
        else:
            df[f"Needs_{r}"] = [1, 1, 1]
    return df

def make_example_caps_df(resource_names, horizon=4):
    data = {"Time": list(range(horizon))}
    for i, r in enumerate(resource_names):
        # Give first resource slightly higher capacity
        base_cap = 4 if i == 0 else 3
        data[f"Cap_{r}"] = [base_cap] * horizon
    return pd.DataFrame(data)

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")
    

# ---------- How to Use ----------
# st.subheader("📘 How to Use This Tool")
st.markdown("""
### 🧭 Overview
This tool solves **Resource-Constrained Project Scheduling Problems (RCPSP)** using a heuristic approach.

It takes two input CSV files:
1. **Tasks file** – defines each task, its duration, resource needs, and dependencies.  
2. **Capacities file** – defines available capacity for each resource over time.

After uploading both files, the tool:
- Runs the scheduling algorithm to find an optimal (or near-optimal) plan  
- Displays a **Gantt Chart** of the scheduled tasks  
- Shows **Resource Utilization** over time  
- Displays the **Final Schedule**, which can be downloaded as a CSV file
""") 

with st.expander(" 📂 Input Files Description", expanded=False):
    st.markdown("""
    ### Input Format

    #### 🧩 Tasks file
    The **tasks file** defines all project activities and their resource requirements.

    It must be a CSV file with the following columns:

    | Column | Description | Example |
    |:--------|:-------------|:---------|
    | `Task ID` | Unique integer starting from 0 | `0`, `1`, `2` |
    | `Duration` | Task duration in time units | `3` |
    | `Predecessors` | IDs of predecessor tasks separated by semicolons (leave blank if none) | `0;1` |
    | `Needs_<resource>` | Resource requirement for each resource type | `Needs_Dev`, `Needs_Ops` |


    **Example:**
    ```
    Task ID,Duration,Predecessors,Needs_Dev,Needs_Ops
    0,3,,2,0
    1,2,,0,1
    2,4,0;1,2,2
    ```
    **Notes:**
    - `Task ID`s must be contiguous integers starting from 0.  
    - Leave the `Predecessors` cell empty if a task has no dependencies.  
    - The `Needs_<resource>` column names must match the resource names defined in the capacities file.


    #### ⚙️ Capacities file
    The **capacities file** defines the available capacity for each resource at each time period.

    It must be a CSV file with the following columns:

    | Column | Description | Example |
    |:--------|:-------------|:---------|
    | `Time` | Time period index (starting from 0, without gaps) | `0`, `1`, `2`, ... |
    | `Cap_<resource>` | Capacity available for each resource type | `Cap_Dev`, `Cap_Ops` |

    **Example:**
    ```
    Time,Cap_Dev,Cap_Ops
    0,4,3
    1,4,3
    2,4,3
    3,4,3
    ```
    **Rules and Behavior:**
    1. Provide capacity values **up to the moment they stop changing**.  
    2. From that point onward, you **do not need to extend** the file — the tool automatically assumes capacities stay **constant at their last known values**.  
    3. The app will **fill any missing time periods** and **forward-fill** the last provided capacity values to cover the full planning horizon.  
    4. Resource column names (`Cap_<resource>`) must match the `Needs_<resource>` columns from the tasks file.

    """)


# ---------- Template Builder (collapsed by default) ----------
with st.expander("🧱 Template Builder", expanded=False):
    st.markdown("""
    Use this when you do not already have CSV inputs.

    What it generates
    1. tasks.csv with Task ID, Duration, Predecessors, and Needs_<resource> columns
    2. capacities.csv with Time and Cap_<resource> columns

    Capacity behavior
    Provide rows from Time = 0 until capacities stop changing. After that moment, you may omit rows, and the app assumes the last provided capacity values continue. Missing time periods are auto filled. The visualization shows usage and capacity only up to the project makespan.

    Naming rules
    Resource names must match between Needs_<resource> and Cap_<resource>.
    """)

    resources_raw = st.text_input(
        "Resource names separated by comma",
        value="Dev,Ops",
        help="Used to create Needs_<resource> in tasks and Cap_<resource> in capacities."
    )
    resource_names = [r.strip() for r in resources_raw.split(",") if r.strip()]

    horizon = st.number_input(
        "Time horizon for capacities",
        min_value=1,
        max_value=365,
        value=8,
        step=1,
        help="Number of time periods to include in the capacities template."
    )

    if len(resource_names) == 0:
        st.info("Enter at least one resource name to generate templates.")
    else:
        # Build templates
        # Assumes you already defined make_example_tasks_df, make_example_caps_df, and df_to_csv_bytes
        example_tasks_df = make_example_tasks_df(resource_names)
        example_caps_df = make_example_caps_df(resource_names, horizon=horizon)

        st.markdown("Preview of tasks.csv")
        st.dataframe(example_tasks_df, use_container_width=True)

        st.markdown("Preview of capacities.csv")
        st.dataframe(example_caps_df, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                label="Download tasks.csv",
                data=df_to_csv_bytes(example_tasks_df),
                file_name="tasks.csv",
                mime="text/csv"
            )
        with c2:
            st.download_button(
                label="Download capacities.csv",
                data=df_to_csv_bytes(example_caps_df),
                file_name="capacities.csv",
                mime="text/csv"
            )

st.divider()

# --------------- Sidebar inputs ---------------
with st.sidebar:
    with st.expander("💾 Upload Your Files", expanded=False):
        tasks_file = st.file_uploader("Upload tasks.csv", type=["csv"], key="tasks")
        caps_file = st.file_uploader("Upload capacities.csv", type=["csv"], key="caps")
    with st.expander("⚙️Optimization parameters", expanded=False):
        iters = st.number_input("ALNS iterations", min_value=10, max_value=20000, value=1000, step=50)
        seed = st.number_input("Random seed", min_value=0, max_value=1_000_000, value=42, step=1)
    run_btn = st.button("🧠 Run Kaveh Scheduler")

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
    st.caption("Upload files and click Run Kaveh Scheduler to generate a schedule.")

st.caption("This dashboard runs an ALNS-based (adaptive large neighborhood search) RCPSP solver and visualizes the optimized schedule.")
