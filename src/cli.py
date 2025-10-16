import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import webbrowser
import os
import plotly.io as pio

from src.data import load_from_csv
from src.solver import solve_with_alns
from src.visualization import plot_schedule_and_usage


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RCPSP with ALNS and time-varying capacities")
    p.add_argument("--tasks", required=True, help="Path to tasks.csv")
    p.add_argument("--caps", required=True, help="Path to capacities.csv")
    p.add_argument("--iters", type=int, default=1000, help="Number of ALNS iterations")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    p.add_argument("--out-schedule", default="schedule_out.csv", help="Output schedule CSV path")
    p.add_argument("--out-plot", default="schedule.html", help="Output plot path (HTML or image if using Kaleido)")
    p.add_argument("--no-plot", action="store_true", help="Skip generating the plot")
    p.add_argument("--show", action="store_true", help="Open the plot after saving")

    p.add_argument("--log-level", default="INFO",
                   choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
                   help="Logging verbosity")
    p.add_argument("--log-file", default=None, help="Optional log file path (default: stderr)")
    return p


def main():
    args = build_arg_parser().parse_args()

    handlers = [logging.StreamHandler(sys.stderr)] if args.log_file is None \
        else [logging.FileHandler(args.log_file, mode="w", encoding="utf-8")]
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
    log = logging.getLogger("rcpsp.cli")

    try:
        tasks_path = Path(args.tasks)
        caps_path = Path(args.caps)
        out_sched = Path(args.out_schedule)
        out_plot = Path(args.out_plot)

        log.info("Loading data from %s and %s", tasks_path, caps_path)
        data = load_from_csv(str(tasks_path), str(caps_path))

        log.info("Running solver. iters=%d seed=%d", args.iters, args.seed)
        summary = solve_with_alns(data, seed=args.seed, iters=args.iters)

        starts = summary["starts"]
        makespan = summary["makespan"]
        total_util = summary.get("total_utilization")

        log.info("Best makespan: %d", makespan)
        print(f"\nBest makespan: {makespan}")

        if total_util is not None:
            log.info("Average total utilization: %.2f%%", total_util * 100)
            print(f"Average utilization: {total_util * 100:.2f}%")

        # --- Write schedule CSV
        sched_df = pd.DataFrame({
            "Task ID": range(len(starts)),
            "Start": starts,
            "Finish": [starts[i] + data.durations[i] for i in range(len(starts))],
            "Duration": data.durations,
        })
        for i, res_name in enumerate(data.resource_names):
            sched_df[f"Needs_{res_name}"] = data.needs[:, i]

        out_sched.parent.mkdir(parents=True, exist_ok=True)
        sched_df.to_csv(out_sched, index=False)
        log.info("Schedule written to %s", out_sched)

        # --- Plot using Plotly
        if not args.no_plot:
            fig = plot_schedule_and_usage(
                data,
                starts,
                usage=summary.get("usage"),
                capacities=summary.get("capacities"),
                title=f"ALNS Schedule (Makespan={makespan})"
            )

            out_plot.parent.mkdir(parents=True, exist_ok=True)

            def save_plotly(fig, path):
                ext = path.suffix.lower()
                if ext in {".png", ".jpg", ".jpeg", ".pdf", ".svg", ".webp"}:
                    import plotly.io as pio
                    pio.write_image(fig, str(path))  # needs kaleido
                    return str(path)
                else:
                    html_path = path if ext == ".html" else path.with_suffix(".html")
                    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
                    return str(html_path)

            saved_gantt = save_plotly(fig, out_plot)  # e.g., schedule.html
            

            if args.show:
                import webbrowser, os
                for p in filter(None, [saved_gantt]):
                    webbrowser.open(f"file://{os.path.abspath(p)}")

        return 0

    except Exception as e:
        log.exception("CLI failed: %s", e)
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
