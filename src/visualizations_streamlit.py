import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc
import streamlit as st

def build_rcpsp_figure(
    data,
    starts,
    usage=None,
    capacities=None,
    time_unit="t",
    task_labels=None,
):
    durations = np.asarray(getattr(data, "durations"))
    starts = np.asarray(starts)
    ends = starts + durations
    makespan = int(ends.max()) if len(ends) else 0
    n_tasks = len(durations)
    task_labels = task_labels or [f"T{i}" for i in range(n_tasks)]

    U = np.asarray(usage) if usage is not None else None
    C = np.asarray(capacities) if capacities is not None else None
    Tu = U.shape[0] if (U is not None and U.ndim == 2) else 0
    Tc = C.shape[0] if (C is not None and C.ndim == 2) else 0
    R  = (U.shape[1] if (U is not None and U.ndim == 2) else
          C.shape[1] if (C is not None and C.ndim == 2) else 0)
    T = max(Tu, Tc, makespan + 1)
    res_names = getattr(data, "resource_names", [f"R{j}" for j in range(R)])
    t_axis = np.arange(T)
    # pad arrays to (T, R)
    def pad2(arr, pad_mode="constant"):
        if arr is None:
            return np.zeros((T, R), dtype=float)
        a = arr
        if a.shape[0] < T:
            a = np.pad(a, ((0, T - a.shape[0]), (0, 0)), mode=("edge" if pad_mode == "edge" else "constant"))
        if a.shape[1] < R:
            a = np.pad(a, ((0, 0), (0, R - a.shape[1])), mode=("edge" if pad_mode == "edge" else "constant"))
        return a

    U = pad2(U, pad_mode="constant")   # usage: zeros beyond data
    C = pad2(C, pad_mode="edge")       # capacity: hold last value beyond data

    rows_total = 1 + max(R, 1)
    row_heights = [0.55] + [0.45 / max(R, 1)] * max(R, 1)

    fig = make_subplots(
        rows=rows_total,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
    )

    # Gantt (numeric y -> lets Plotly zoom smoothly)
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
                    f"Finish: {starts[i]+durations[i]} {time_unit}<br>"
                    f"Duration: {durations[i]} {time_unit}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1, col=1
        )

    # Gantt axis: show all task labels by default; reverse so Task 0 is at top
    if n_tasks:
        fig.update_yaxes(
            title_text="Tasks",
            tickmode="array",
            tickvals=y_positions,
            ticktext=task_labels,
            autorange="reversed",
            row=1, col=1
        )
    else:
        fig.update_yaxes(title_text="Tasks", autorange="reversed", row=1, col=1)

    # Utilization subplots (one per resource), step-shaped lines
    show_global_legend = False
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
                legendgroup=rname,
                showlegend=show_global_legend
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
                    legendgroup=rname,
                    showlegend=show_global_legend
                ),
                row=j + 2, col=1
            )
        fig.update_yaxes(title_text=rname, row=j + 2, col=1)

    # Dynamic ticks: let Plotly pick nice ticks and adapt on zoom
    # Hint: integers, outside ticks, ~10-12 ticks max
    for r in range(1, rows_total + 1):
        fig.update_xaxes(
            tickmode="auto",   # dynamic tick density
            nticks=12,         # hint max number of ticks
            tickformat=",d",   # integer formatting
            ticks="outside",
            showticklabels=True,   # <- force labels on every row
            showline=True,
            mirror=True,           # draw axis line on both sides; makes ticks clear
            range=[0, makespan], row=r, col=1
        )
    fig.update_xaxes(title_text=f"Time [{time_unit}]", row=rows_total, col=1)

    # Keep user zoom/pan when Streamlit reruns
    fig.update_layout(
        hovermode="x unified",
        showlegend=False,
        bargap=0.15,
        margin=dict(l=90, r=30, t=60, b=60),
        height=max(700, 280 + 22 * max(n_tasks if n_tasks else 10, 10) + 220 * max(R, 1)),
        uirevision="rcpsp-zoom"   # stable UI state across reruns
    )
    return fig


def rcpsp_streamlit_page(
    data,
    starts,
    usage=None,
    capacities=None,
    title="ALNS Schedule",
    time_unit="t",
):
    """Render-only: no sidebar controls. Use Plotly's zoom/pan."""
    fig = build_rcpsp_figure(
        data=data,
        starts=starts,
        usage=usage,
        capacities=capacities,
        time_unit=time_unit,
    )
    fig.update_layout(title=title)
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    # # Downloads
    # html_bytes = fig.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
    # st.download_button(
    #     label="Download interactive HTML",
    #     data=html_bytes,
    #     file_name="schedule.html",
    #     mime="text/html",
    #     use_container_width=True,
    # )
    # try:
    #     import plotly.io as pio
    #     png_bytes = pio.to_image(fig, format="png", scale=2)
    #     st.download_button(
    #         label="Download PNG (static)",
    #         data=png_bytes,
    #         file_name="schedule.png",
    #         mime="image/png",
    #         use_container_width=True,
    #     )
    # except Exception:
    #     pass


