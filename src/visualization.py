import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_schedule_and_usage(
    data,
    starts,
    lb=None,
    ub=None,
    title="ALNS Schedule",
    usage=None,          # (T, R)
    capacities=None,     # (T, R)
    time_unit="t",
    task_labels=None,
    resample_factor=None,
    gantt_page_size=30,  # rows shown when slicing Y
    x_window=None,       # width of X window for the slicer; default = 20% of span (min 20)
    x_step=5             # step for X slicer (time units)
):
    durations = np.asarray(getattr(data, "durations"))
    starts = np.asarray(starts)
    ends = starts + durations
    n_tasks = len(durations)
    task_labels = task_labels or [f"T{i}" for i in range(n_tasks)]

    # --------- utilization data ----------
    U = np.asarray(usage) if usage is not None else None
    C = np.asarray(capacities) if capacities is not None else None
    R = U.shape[1] if U is not None and U.ndim == 2 else 0
    T = U.shape[0] if U is not None else 0
    res_names = getattr(data, "resource_names", [f"R{j}" for j in range(R)])

    if U is not None and resample_factor and resample_factor > 1 and T > 0:
        s = int(resample_factor)
        new_len = T // s
        if new_len > 0:
            U = U[:new_len * s].reshape(new_len, s, R).mean(axis=1)
            if C is not None:
                C = C[:new_len * s].reshape(new_len, s, R).mean(axis=1)
            T = new_len

    t_axis = np.arange(T) if T else np.array([])

    # --------- figure: Gantt + resources ----------
    rows_total = 1 + max(R, 1)
    row_heights = [0.55] + [0.45 / max(R, 1)] * max(R, 1)

    fig = make_subplots(
        rows=rows_total,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
        subplot_titles=(None,) * rows_total
    )

    # --------- Gantt (row 1) ----------
    y_positions = np.arange(n_tasks)
    for i in range(n_tasks):
        fig.add_trace(
            go.Bar(
                x=[durations[i]],
                y=[y_positions[i]],
                base=[starts[i]],
                orientation="h",
                hovertemplate=(
                    f"Task: {task_labels[i]}<br>"
                    f"Start: {starts[i]} {time_unit}<br>"
                    f"Finish: {ends[i]} {time_unit}<br>"
                    f"Duration: {durations[i]} {time_unit}<extra></extra>"
                ),
                showlegend=False
            ),
            row=1, col=1
        )

    x_max_tasks = int(max(ends)) if n_tasks else 0
    if lb is not None:
        fig.add_vrect(x0=0, x1=lb, fillcolor="LightGreen", opacity=0.2, line_width=0, layer="below", row=1, col=1)
    if ub is not None:
        fig.add_vrect(x0=ub, x1=x_max_tasks, fillcolor="LightCoral", opacity=0.15, line_width=0, layer="below", row=1, col=1)

    # default: show all tasks
    fig.update_yaxes(
        title_text="Tasks",
        tickmode="array",
        tickvals=y_positions,
        ticktext=task_labels,
        range=[-0.5, n_tasks - 0.5 if n_tasks else 9.5],
        row=1, col=1
    )

    # --------- Utilization (rows 2..R+1) ----------
    for j in range(max(R, 1)):
        rname = res_names[j] if R else "R0"
        fig.add_trace(
            go.Scatter(
                x=t_axis,
                y=(U[:, j] if T else []),
                mode="lines",
                line=dict(shape="hv"),
                fill="tozeroy",
                name=f"Usage {rname}",
                legendgroup=rname
            ),
            row=j + 2, col=1
        )
        if C is not None and T:
            fig.add_trace(
                go.Scatter(
                    x=t_axis,
                    y=C[:, j],
                    mode="lines",
                    line=dict(shape="hv"),
                    name=f"Cap {rname}",
                    legendgroup=rname
                ),
                row=j + 2, col=1
            )
        fig.update_yaxes(title_text=rname, row=j + 2, col=1)

    # --------- axes & ticks ---------
    fig.update_xaxes(matches="x", tickmode="linear", dtick=1)
    fig.update_xaxes(title_text=f"Time [{time_unit}]", row=rows_total, col=1)

    # --------- build "clean" sliders (no thumbnails) ---------
    # X slicer: default = show ALL
    x_span = max(x_max_tasks, int(t_axis[-1]) if T > 0 else 0)
    if x_span <= 0:
        x_span = 1
    if x_window is None:
        x_window = max(20, int(round(0.2 * x_span)))
    x_step = max(1, int(x_step))
    x_max_start = max(0, x_span - x_window)
    x_starts = list(range(0, x_max_start + 1, x_step)) or [0]

    x_steps = [{
        "label": "All",
        "method": "relayout",
        "args": [{"xaxis.autorange": True}]  # default show everything
    }]
    for s in x_starts:
        x_steps.append(dict(
            label=str(s),
            method="relayout",
            args=[{"xaxis.range": [s, s + x_window]}]
        ))

    # Y slicer (Gantt): default = show ALL; then 1-row increments for smooth slicing
    y_steps = [{
        "label": "All",
        "method": "relayout",
        "args": [{"yaxis.range": [-0.5, n_tasks - 0.5]}]
    }]
    if n_tasks > gantt_page_size:
        for start in range(0, n_tasks - gantt_page_size + 1, 1):  # dense
            end = start + gantt_page_size - 1
            y_steps.append(dict(
                label=f"{start+1}-{min(end+1, n_tasks)}",
                method="relayout",
                args=[{"yaxis.range": [start - 0.5, end + 0.5]}]
            ))

    sliders = []
    # Y slider: place just under the title (but not inside it)
    sliders.append(dict(
        active=0,
        currentvalue=dict(prefix="Rows: "),
        pad=dict(t=5, b=5),
        steps=y_steps,
        x=0.5, xanchor="center",
        y=0.975, yanchor="top"
    ))
    # X slider: place below the figure, outside the last subplot
    sliders.append(dict(
        active=0,
        currentvalue=dict(prefix="t0: "),
        pad=dict(t=5, b=5),
        steps=x_steps,
        x=0.5, xanchor="center",
        y=-0.12, yanchor="bottom"   # push it below the plotting area
    ))

    # --------- layout spacing (no overlaps) ---------
    visible_rows = min(gantt_page_size, n_tasks) if n_tasks else 10
    fig.update_layout(
        title=dict(text=title, y=0.94),
        hovermode="x unified",
        legend_title_text="",
        bargap=0.15,
        margin=dict(l=90, r=30, t=110, b=160),  # extra bottom for the X slider
        height=max(750, 260 + 24 * max(visible_rows, 10) + 220 * max(R, 1)),
        sliders=sliders
    )

    return fig
