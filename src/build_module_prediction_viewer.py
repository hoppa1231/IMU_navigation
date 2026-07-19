#!/usr/bin/env python3
"""Build a lightweight trajectory viewer for module route holdout predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=Path, default=Path("derived/predictions/module_window_baselines_all_routes"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/generated/module_predictions/index.html"))
    parser.add_argument("--models", nargs="+", default=["ridge"], help="Prediction model stems, without `_pred.csv`.")
    return parser.parse_args()


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return math.nan


def read_case(path: Path) -> dict[str, object] | None:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return None
    selected: list[dict[str, str]] = []
    next_time = -math.inf
    for row in sorted(rows, key=lambda item: number(item, "time_s")):
        if number(row, "time_s") + 1e-9 >= next_time:
            selected.append(row)
            next_time = number(row, "future_time_s")
    if not selected:
        return None
    truth_e = truth_n = pred_e = pred_n = 0.0
    points: list[dict[str, float]] = [{"te": 0.0, "tn": 0.0, "pe": 0.0, "pn": 0.0}]
    errors: list[float] = []
    for row in selected:
        truth_e += number(row, "true_dx_east_m")
        truth_n += number(row, "true_dy_north_m")
        pred_e += number(row, "pred_dx_east_m")
        pred_n += number(row, "pred_dy_north_m")
        errors.append(math.hypot(pred_e - truth_e, pred_n - truth_n))
        points.append({"te": truth_e, "tn": truth_n, "pe": pred_e, "pn": pred_n})
    return {
        "id": str(path.parent.parent.name + "/" + path.parent.name),
        "label": f"{rows[0]['flight_id']} — {path.parent.parent.name} / {path.stem.removesuffix('_pred')}",
        "steps": len(selected),
        "finalError": errors[-1],
        "meanError": sum(errors) / len(errors),
        "points": points,
    }


def main() -> None:
    args = parse_args()
    paths = [path for model in args.models for path in args.pred_dir.glob(f"*/route_holdout_*/{model}_pred.csv")]
    if args.models == ["ridge"]:
        paths = list(args.pred_dir.glob("*/all_routes_holdout_*/ridge_pred.csv"))
    cases = [case for path in sorted(paths) if (case := read_case(path))]
    if not cases:
        raise ValueError(f"No ridge predictions found under {args.pred_dir}")
    payload = json.dumps(cases, ensure_ascii=False)
    page = f'''<!doctype html><meta charset="utf-8"><title>Module route predictions</title>
<style>body{{margin:0;font:14px system-ui;background:#f4f6f8;color:#17202a}}header{{padding:14px 18px;background:#fff;border-bottom:1px solid #d8dde5;display:flex;gap:16px;align-items:center}}select{{font:inherit;padding:7px;min-width:360px}}main{{display:grid;grid-template-columns:1fr 250px;height:calc(100vh - 58px)}}svg{{width:100%;height:100%;background:#eef1f5}}aside{{padding:16px;background:#fff;border-left:1px solid #d8dde5}}.k{{margin:12px 0}}.line{{display:inline-block;width:24px;height:3px;margin-right:7px;vertical-align:middle}}</style>
<header><strong>GNSS-free route-holdout rollout</strong><select id="pick"></select></header><main><svg id="plot"></svg><aside><div class="k">Steps: <b id="steps"></b></div><div class="k">Mean horizontal error: <b id="mean"></b></div><div class="k">Final horizontal error: <b id="final"></b></div><p><span class="line" style="background:#2563eb"></span>GPS target increments</p><p><span class="line" style="background:#dc2626"></span>Accumulated ridge prediction</p><small>Each point is a non-overlapping prediction window. GPS is used only as target/reference.</small></aside></main>
<script>const cases={payload},pick=document.querySelector('#pick'),svg=document.querySelector('#plot');cases.forEach((c,i)=>pick.add(new Option(c.label,i)));function draw(){{const c=cases[pick.value||0],p=c.points,x=p.flatMap(q=>[q.te,q.pe]),y=p.flatMap(q=>[q.tn,q.pn]),r=svg.getBoundingClientRect(),pad=30,minx=Math.min(...x),maxx=Math.max(...x),miny=Math.min(...y),maxy=Math.max(...y),sx=v=>pad+(v-minx)/Math.max(maxx-minx,1)*(r.width-2*pad),sy=v=>r.height-pad-(v-miny)/Math.max(maxy-miny,1)*(r.height-2*pad),path=(a,b)=>p.map((q,i)=>`${{i?'L':'M'}}${{sx(q[a]).toFixed(1)}},${{sy(q[b]).toFixed(1)}}`).join('');svg.setAttribute('viewBox',`0 0 ${{r.width}} ${{r.height}}`);svg.innerHTML=`<path d="${{path('te','tn')}}" fill="none" stroke="#2563eb" stroke-width="2.5"/><path d="${{path('pe','pn')}}" fill="none" stroke="#dc2626" stroke-width="2.5"/>`;steps.textContent=c.steps;mean.textContent=c.meanError.toFixed(2)+' m';final.textContent=c.finalError.toFixed(2)+' m'}}pick.onchange=draw;onresize=draw;draw()</script>'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    print(f"Wrote {args.output} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
