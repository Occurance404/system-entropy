from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _fmt_num(v: float | int) -> str:
    if pd.isna(v):
        return "-"
    if isinstance(v, (int,)) or float(v).is_integer():
        return f"{int(v):,}"
    return f"{v:,.3f}"


def _table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df.empty:
        return "<p><em>No data</em></p>"
    show = df.head(max_rows).copy()
    return show.to_html(index=False, classes="tbl", border=0)


def _img_grid(title: str, figure_paths: list[str]) -> str:
    cards = []
    for p in figure_paths:
        cards.append(
            f"""
            <figure class="fig-card">
              <img src="{p}" alt="{Path(p).name}" loading="lazy" />
              <figcaption>{Path(p).name}</figcaption>
            </figure>
            """
        )
    body = "\n".join(cards) if cards else "<p><em>No figures found</em></p>"
    return f"""
    <section>
      <h2>{title}</h2>
      <div class="fig-grid">
        {body}
      </div>
    </section>
    """


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", default="20260225_231718")
    ap.add_argument("--two-model-tag", default="20260222_123759")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    run_tag = args.run_tag
    two_tag = args.two_model_tag

    base = Path("data/results")
    out = Path(args.out) if args.out else base / f"campaign_dashboard_{run_tag}.html"

    merged = _read_csv(base / f"benchmark_swepolybench_models_{run_tag}.csv")
    summary_all = _read_csv(base / f"swepolybench_models_{run_tag}_model_summary_all.csv")
    summary_clean = _read_csv(base / f"swepolybench_models_{run_tag}_model_summary_clean_noninfra.csv")
    probe_summary = _read_csv(base / f"swepolybench_models_{run_tag}_probe_scr_summary_by_model.csv")
    failure_all = _read_csv(base / f"swepolybench_models_{run_tag}_failure_composition_all.csv")
    error_counts = _read_csv(base / f"swepolybench_models_{run_tag}_error_category_counts.csv")
    hard_instances = _read_csv(base / f"swepolybench_models_{run_tag}_hard_instances_noninfra.csv")
    two_model_summary = _read_csv(base / f"swepolybench_two_models_{two_tag}_model_summary.csv")

    total_runs = len(merged) if not merged.empty else 0
    successes = int(merged["validation_passed"].astype(str).str.lower().isin(["1", "true", "yes"]).sum()) if total_runs else 0
    success_rate = (successes / total_runs) if total_runs else 0.0

    core_fig_dir = base / f"figures_models_{run_tag}_final"
    forensics_fig_dir = base / f"figures_models_{run_tag}_forensics"
    two_fig_dir = base / f"figures_two_models_{two_tag}_final"

    core_figs = sorted([str(p.relative_to(base)) for p in core_fig_dir.glob("*.png")]) if core_fig_dir.exists() else []
    forensics_figs = sorted([str(p.relative_to(base)) for p in forensics_fig_dir.glob("*.png")]) if forensics_fig_dir.exists() else []
    two_figs = sorted([str(p.relative_to(base)) for p in two_fig_dir.glob("*.png")]) if two_fig_dir.exists() else []

    artifacts = [
        f"benchmark_swepolybench_models_{run_tag}.csv",
        f"swepolybench_models_{run_tag}_model_summary_all.csv",
        f"swepolybench_models_{run_tag}_model_summary_clean_noninfra.csv",
        f"swepolybench_models_{run_tag}_probe_scr_summary_by_model.csv",
        f"swepolybench_models_{run_tag}_failure_composition_all.csv",
        f"swepolybench_models_{run_tag}_error_category_counts.csv",
        f"swepolybench_models_{run_tag}_CLOSEOUT_REPORT.md",
        f"figures_models_{run_tag}_final/",
        f"figures_models_{run_tag}_forensics/",
    ]

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SWE-PolyBench Campaign Dashboard ({run_tag})</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --card: #ffffff;
      --ink: #1f2937;
      --muted: #6b7280;
      --line: #e5e7eb;
      --accent: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", Roboto, sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, #f9fafb 0%, #eef2ff 100%);
    }}
    .wrap {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 6px; font-size: 2rem; }}
    .sub {{ color: var(--muted); margin-bottom: 18px; }}
    section {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
      margin: 14px 0;
      box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      background: #fbfffe;
    }}
    .k {{ font-size: 0.85rem; color: var(--muted); }}
    .v {{ font-size: 1.35rem; font-weight: 700; color: var(--accent); }}
    .tbl {{
      width: 100%;
      border-collapse: collapse;
      overflow-x: auto;
      display: block;
    }}
    .tbl th, .tbl td {{
      border: 1px solid var(--line);
      padding: 6px 8px;
      font-size: 0.9rem;
      white-space: nowrap;
    }}
    .tbl th {{ background: #f3f4f6; text-align: left; }}
    .fig-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 14px;
    }}
    .fig-card {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
      background: #fff;
    }}
    .fig-card img {{
      width: 100%;
      height: auto;
      display: block;
      background: #fff;
    }}
    .fig-card figcaption {{
      font-size: 0.82rem;
      color: var(--muted);
      padding: 8px 10px;
      border-top: 1px solid var(--line);
    }}
    ul.artifacts {{ margin: 0; padding-left: 18px; }}
    code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>SWE-PolyBench Campaign Dashboard</h1>
    <div class="sub">Run tag <code>{run_tag}</code> | Generated {now}</div>

    <section>
      <h2>Executive Snapshot</h2>
      <div class="cards">
        <div class="card"><div class="k">Total Runs</div><div class="v">{_fmt_num(total_runs)}</div></div>
        <div class="card"><div class="k">Successful Runs</div><div class="v">{_fmt_num(successes)}</div></div>
        <div class="card"><div class="k">Success Rate</div><div class="v">{success_rate:.3f}</div></div>
        <div class="card"><div class="k">Models in Campaign</div><div class="v">{_fmt_num(summary_all.shape[0])}</div></div>
      </div>
      <p>
        Reading rule for the paper: use <strong>clean non-infra tables</strong> for behavior claims,
        and use all-rows tables/forensics for operational reliability discussion.
      </p>
    </section>

    <section>
      <h2>Model Summary (All Rows)</h2>
      {_table(summary_all)}
    </section>

    <section>
      <h2>Model Summary (Non-Infra Rows)</h2>
      {_table(summary_clean)}
    </section>

    <section>
      <h2>Probe Summary by Model</h2>
      {_table(probe_summary)}
    </section>

    <section>
      <h2>Failure Composition (All Rows)</h2>
      {_table(failure_all)}
    </section>

    <section>
      <h2>Error Categories</h2>
      {_table(error_counts)}
    </section>

    <section>
      <h2>Hard Instances (Non-Infra)</h2>
      {_table(hard_instances, max_rows=30)}
    </section>

    {_img_grid("Core Figures (5-Model Campaign)", core_figs)}
    {_img_grid("Failure Forensics Figures", forensics_figs)}
    {_img_grid(f"Reference Figures (2-Model Run {two_tag})", two_figs)}

    <section>
      <h2>Artifact Index</h2>
      <ul class="artifacts">
        {"".join(f'<li><code>{a}</code></li>' for a in artifacts)}
      </ul>
    </section>

    <section>
      <h2>How to Regenerate</h2>
      <p><code>./.venv/bin/python analysis/closeout_swepolybench_campaign.py --run-tag {run_tag} --models-json benchmarks/models.openrouter.remaining5.json</code></p>
      <p><code>./.venv/bin/python analysis/failure_forensics_swepolybench.py --run-tag {run_tag}</code></p>
    </section>
  </div>
</body>
</html>
"""

    _ensure_dir(out.parent)
    out.write_text(html, encoding="utf-8")
    print(f"[ok] dashboard: {out}")


if __name__ == "__main__":
    main()
