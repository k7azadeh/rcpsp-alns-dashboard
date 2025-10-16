# RCPSP Scheduler Dashboard (ALNS)

An interactive **Streamlit** dashboard (and optional CLI) for solving the **Resource-Constrained Project Scheduling Problem (RCPSP)** using an ALNS solver, with clean, zoomable **Plotly** visuals.

---

## Highlights

- **Plotly-only** visualization (no Matplotlib).
- **One figure**: Gantt on top, per-resource utilization subplots below.
- **Natural zoom/pan** (no sidebar sliders).
- **Dynamic ticks** that adapt as you zoom.
- **Gantt y-axis reversed** so Task 0 is at the top.
- **Resource names** on each utilization subplot (left y-axis).
- Zoom state persists in Streamlit via `uirevision`.

---

## Quick Start

### 1) Create & activate a virtual environment
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

> Optional for static PNG export from Plotly:
> ```bash
> pip install -U kaleido
> ```

### 3) Run the Streamlit app
```bash
streamlit run app.py
```

Upload `tasks.csv` and `capacities.csv`, set ALNS parameters, click **Run ALNS Scheduler**, then explore results with Plotly’s zoom/pan.

---

## Project Layout

```
.
├── app.py                          # Streamlit dashboard (uses session_state)
├── src/
│   ├── __init__.py
│   ├── data.py                     # load_from_csv(...), has_dummies(...)
│   ├── solver.py                   # solve_with_alns(...)
│   ├── visualizations_streamlit.py # Plotly/Streamlit visualization (this version)
│   └── visualization.py            # (optional) legacy/CLI helpers
├── requirements.txt
├── README.md
└── .gitignore
```

> Ensure `src/__init__.py` exists (can be empty) so `src.*` imports work.

---

## Visualization Details

- **Gantt**: uniform bar color; **Task 0 at the top**.
- **Utilization**: one subplot per resource.  
  - **Usage**: step line with translucent fill  
  - **Capacity**: step line with dotted style
- **Ticks**: dynamic (`tickmode="auto"`, `nticks≈12`) so they thin when zoomed out and densify when zoomed in.
- **Zoom state** persists across reruns with `uirevision="rcpsp-zoom"`.

**Tips**
- Drag to pan, scroll to zoom; double-click an axis to reset.
- Use the Plotly toolbar (top-right) for extra zoom/reset tools.

---

## Optional: CLI

CLI visualization is not perfect!!! needs to be updated if needed.

If your repo also includes a CLI, you can run:

```bash
python -m rcpsp.cli   --tasks path/to/tasks.csv   --caps path/to/capacities.csv   --iters 1000   --seed 42   --out-schedule schedule_out.csv   --out-plot schedule.html
```

- `schedule_out.csv` contains Task ID, Start, Finish, Duration, and resource needs.
- `schedule.html` is interactive Plotly (open in a browser).
- For PNG output instead of HTML, pass a `.png` path and install **kaleido**.

---

## Troubleshooting

- **Import error for `src.visualizations_streamlit`**  
  Run `streamlit run app.py` from the **repo root** and confirm `src/__init__.py` exists.

- **Monochrome HTML export**  
  This version sets explicit colors and `template="plotly_white"`, so exported HTML matches on-screen colors.

- **PNG export fails**  
  Install `kaleido`: `pip install -U kaleido`.

---

## License

