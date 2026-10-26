"""Write an offline interactive chart of every tested concentration prefix."""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pod_platform.application.configuration import config_path, load_config


def main():
    config = load_config()
    state = config_path(config["application"]["state_dir"])
    report_path = Path(
        json.loads((state / "latest_edge_scan.json").read_text())["report"]
    )
    report = json.loads(report_path.read_text())
    curves = pd.read_csv(report_path.parent / "edge_curves.csv")
    pods = ["fundamentals", "events", "valuation", "dynamics"]
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[p.title() for p in pods],
        vertical_spacing=0.16,
    )
    colors = ["#1766a7", "#008578", "#ca6d14"]
    for number, pod in enumerate(pods):
        row, column = number // 2 + 1, number % 2 + 1
        identities = [i for i in report["plans"] if i.startswith(pod + ":")]
        best = max(
            identities,
            key=lambda i: report["plans"][i]["rank_ic"]["mean"] or -1,
        )
        for index, identity in enumerate(identities):
            curve = curves.loc[
                (curves.strategy_id == identity) & (curves.months >= 24)
            ]
            x = curve.names.to_list()
            mean = (curve["mean"] * 1200).to_list()
            if identity == best:
                upper = ((curve["mean"] + 1.96 * curve.hac_se) * 1200).to_list()
                lower = ((curve["mean"] - 1.96 * curve.hac_se) * 1200).to_list()
                figure.add_trace(
                    go.Scatter(
                        x=x + x[::-1],
                        y=upper + lower[::-1],
                        fill="toself",
                        fillcolor="rgba(70,90,120,0.10)",
                        line={"width": 0},
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=row,
                    col=column,
                )
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=mean,
                    mode="lines",
                    name=identity,
                    line={"color": colors[index % len(colors)], "width": 2},
                    hovertemplate="%{x} total names<br>Controlled annualized spread: %{y:.2f}%<extra>%{fullData.name}</extra>",
                ),
                row=row,
                col=column,
            )
        figure.add_hline(
            y=0, line_dash="dash", line_color="#888", row=row, col=column
        )
        figure.update_xaxes(
            type="log",
            title="Names in the diagnostic long/short prefix",
            row=row,
            col=column,
        )
        figure.update_yaxes(
            title="Annualized arithmetic spread (%)",
            range=[-40, 40],
            row=row,
            col=column,
        )
    figure.update_layout(
        title="Training signal edge across the full concentration curve",
        height=930,
        template="plotly_white",
        legend={"orientation": "h", "y": -0.15},
        margin={"b": 160},
    )
    rows = []
    for pod in pods:
        evidence = report["pods"][pod]
        all_ic = evidence["frozen_signal_rank_ic"]["mean"]
        short_ic = evidence["frozen_shortable_rank_ic"]["mean"]
        rows.append(
            f"<tr><td>{html.escape(pod.title())}</td><td>{all_ic:.4f}</td><td>{short_ic:.4f}</td></tr>"
        )
    caption = """<p>All eligible supplied stocks are ranked. No dollar capital or fixed name count enters this test.
The curves show one-month gross return spreads after benchmarking the declared risk controls; they are not executable NAV returns.
One unit is assigned to each leg. Extreme prefixes may violate eventual portfolio constraints. All prefixes are training research.
The band is a pointwise 95% monthly HAC interval for the strongest full-training rank-IC signal; it is not corrected for the project's multiple searches.
Zoom or click a legend entry to explore the curves. The final combined fund must separately satisfy 200–500 holdings.</p>
<h2>2018–2020 ranking information from annually frozen signal choices</h2>
<p>Rank correlation measures ordering information, not a return percentage or demonstrated after-cost alpha.</p>"""
    table = (
        "<table><tr><th>Pod</th><th>All eligible stocks</th><th>Stocks meeting the $500m short floor</th></tr>"
        + "".join(rows)
        + "</table>"
    )
    chart = figure.to_html(full_html=False, include_plotlyjs=True)
    document = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Training edge curves</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:20px;color:#203044}p{max-width:1100px;line-height:1.5}td,th{text-align:left;padding:10px 22px;border-bottom:1px solid #dde3e9}</style></head><body><h1>Signal edge before capital</h1>"
        + caption
        + table
        + chart
        + "</body></html>"
    )
    destination = report_path.parent / "edge_curves.html"
    destination.write_text(document)
    print(destination)


if __name__ == "__main__":
    main()
