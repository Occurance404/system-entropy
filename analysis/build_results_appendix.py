from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _fmt(v: object, ndigits: int = 3) -> str:
    if pd.isna(v):
        return "-"
    if isinstance(v, (int,)):
        return f"{v:,}"
    if isinstance(v, float):
        if float(v).is_integer():
            return f"{int(v):,}"
        s = f"{v:,.{ndigits}f}"
        return s.rstrip("0").rstrip(".")
    return str(v)


def _as_int(series: pd.Series) -> int:
    return int(pd.to_numeric(series, errors="coerce").fillna(0).sum())


def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No data available._\n"
    show = df.head(max_rows).copy()
    header = "| " + " | ".join(show.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(show.columns)) + " |"
    rows = []
    for _, row in show.iterrows():
        vals = [_fmt(row[c]) for c in show.columns]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *rows]) + "\n"


def _html_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "<p><em>No data available.</em></p>"
    return df.head(max_rows).to_html(index=False, classes="tbl", border=0)


def _fig_if_exists(root: Path, rel_path: str, caption: str) -> tuple[str, str] | None:
    p = root / rel_path
    if not p.exists():
        return None
    return (rel_path, caption)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", default="20260225_231718")
    ap.add_argument("--two-model-tag", default="20260222_123759")
    args = ap.parse_args()

    run_tag = args.run_tag
    two_tag = args.two_model_tag

    base = Path("data/results")
    paper_dir = Path("paper")
    md_out = paper_dir / f"RESULTS_APPENDIX_{run_tag}.md"
    html_out = base / f"results_appendix_{run_tag}.html"

    merged = _read_csv(base / f"benchmark_swepolybench_models_{run_tag}.csv")
    summary_all = _read_csv(base / f"swepolybench_models_{run_tag}_model_summary_all.csv")
    summary_clean = _read_csv(base / f"swepolybench_models_{run_tag}_model_summary_clean_noninfra.csv")
    probe_summary = _read_csv(base / f"swepolybench_models_{run_tag}_probe_scr_summary_by_model.csv")
    failure_all = _read_csv(base / f"swepolybench_models_{run_tag}_failure_composition_all.csv")
    error_counts = _read_csv(base / f"swepolybench_models_{run_tag}_error_category_counts.csv")
    for_cat = _read_csv(base / f"figures_models_{run_tag}_forensics/tbl_failure_category_counts.csv")
    for_repo = _read_csv(base / f"figures_models_{run_tag}_forensics/tbl_repo_failure_counts_noninfra.csv")
    for_model_cat = _read_csv(base / f"figures_models_{run_tag}_forensics/tbl_failure_category_by_model.csv")
    hard_instances = _read_csv(base / f"swepolybench_models_{run_tag}_hard_instances_noninfra.csv")
    two_model_summary = _read_csv(base / f"swepolybench_two_models_{two_tag}_model_summary.csv")

    total_runs = len(merged)
    total_success = 0
    if total_runs and "validation_passed" in merged.columns:
        total_success = int(
            merged["validation_passed"]
            .astype(str)
            .str.lower()
            .isin(["1", "true", "yes"])
            .sum()
        )
    total_failure = max(total_runs - total_success, 0)
    overall_success_rate = (total_success / total_runs) if total_runs else 0.0

    clean_runs = _as_int(summary_clean["runs"]) if not summary_clean.empty else 0
    clean_success = _as_int(summary_clean["successes"]) if not summary_clean.empty else 0
    clean_success_rate = (clean_success / clean_runs) if clean_runs else 0.0

    all_rank = pd.DataFrame()
    clean_rank = pd.DataFrame()
    probe_rank = pd.DataFrame()

    if not summary_all.empty:
        cols = [
            "model_name",
            "runs",
            "successes",
            "success_rate",
            "zero_step_rate",
            "infra_prefetch_failure_rate",
            "median_total_tokens_incl_probes",
            "successes_per_million_tokens",
        ]
        all_rank = (
            summary_all[[c for c in cols if c in summary_all.columns]]
            .sort_values("success_rate", ascending=False)
            .reset_index(drop=True)
        )

    if not summary_clean.empty:
        cols = [
            "model_name",
            "runs",
            "successes",
            "success_rate",
            "zero_step_rate",
            "median_total_tokens_incl_probes",
            "successes_per_million_tokens",
        ]
        clean_rank = (
            summary_clean[[c for c in cols if c in summary_clean.columns]]
            .sort_values("success_rate", ascending=False)
            .reset_index(drop=True)
        )

    if not probe_summary.empty:
        cols = [
            "model_name",
            "probe_log_coverage",
            "runs_with_probe_logs",
            "runs_with_probe_scr",
            "median_probe_scr_median",
            "median_probe_scr_max",
            "median_probe_events",
        ]
        probe_rank = probe_summary[[c for c in cols if c in probe_summary.columns]].copy()
        probe_rank = probe_rank.sort_values("probe_log_coverage", ascending=False).reset_index(drop=True)

    fail_pivot = pd.DataFrame()
    if not failure_all.empty:
        fail_pivot = (
            failure_all.pivot(index="model_name", columns="failure_class", values="rate")
            .fillna(0.0)
            .reset_index()
        )

    if not for_model_cat.empty:
        for_model_cat = for_model_cat.sort_values(["model_name", "count"], ascending=[True, False])

    core_figure_specs = [
        ("figures_models_%s_final/fig_workload_summary_panel_all.png" % run_tag, "Figure A1. Workload-scale summary on all rows. This is the operational view, including infrastructure confounds."),
        ("figures_models_%s_final/fig_workload_summary_panel_clean_noninfra.png" % run_tag, "Figure A2. Workload-scale summary after excluding infrastructure-prefetch failures. This is the behavior-comparison view."),
        ("figures_models_%s_final/fig_failure_composition_all.png" % run_tag, "Figure A3. Failure composition by model on all rows, separating success, zero-step failures, infra-prefetch failures, and agent-executed failures."),
        ("figures_models_%s_final/fig_failure_composition_clean_noninfra.png" % run_tag, "Figure A4. Failure composition by model in the clean non-infra subset."),
        ("figures_models_%s_final/fig_scr_probe_distribution.png" % run_tag, "Figure A5. Distribution of per-run SCR_probe medians with probe-log coverage annotations."),
        ("figures_models_%s_final/fig_probe_log_coverage.png" % run_tag, "Figure A6. Probe-log coverage by model, used to bound interpretability of SCR_probe comparisons."),
        ("figures_models_%s_final/fig_token_breakdown_task_vs_probe.png" % run_tag, "Figure A7. Task tokens versus probe tokens, showing the observability overhead profile."),
        ("figures_models_%s_final/fig_successes_per_million_tokens.png" % run_tag, "Figure A8. Cost-normalized throughput (successes per million tokens including probes)."),
        ("figures_models_%s_final/fig_infra_prefetch_failure_rate.png" % run_tag, "Figure A9. Infrastructure-prefetch failure rate by model; this is a confound diagnostic."),
        ("figures_models_%s_final/fig_chunk_completion_timeline.png" % run_tag, "Figure A10. Chunk-completion timeline for campaign operations."),
    ]
    forensics_figure_specs = [
        ("figures_models_%s_forensics/fig_forensics_failure_categories_overall.png" % run_tag, "Figure A11. Overall failure-category composition across the campaign."),
        ("figures_models_%s_forensics/fig_forensics_failure_categories_by_model.png" % run_tag, "Figure A12. Failure-category composition split by model."),
        ("figures_models_%s_forensics/fig_forensics_repo_failures_noninfra_top15.png" % run_tag, "Figure A13. Top repositories by non-infrastructure failure count."),
        ("figures_models_%s_forensics/fig_forensics_hard_instances_heatmap_noninfra.png" % run_tag, "Figure A14. Hard-instance heatmap in the non-infrastructure slice."),
    ]
    two_model_specs = [
        ("figures_two_models_%s_final/fig_swepoly_workload_summary_panel.png" % two_tag, "Figure A15. Two-model reference workload panel from run %s." % two_tag),
        ("figures_two_models_%s_final/fig_swepoly_scr_probe_distribution.png" % two_tag, "Figure A16. Two-model reference SCR_probe distribution from run %s." % two_tag),
        ("figures_two_models_%s_final/fig_swepoly_failure_composition.png" % two_tag, "Figure A17. Two-model reference failure composition from run %s." % two_tag),
    ]

    core_figs = [x for x in (_fig_if_exists(base, rel, cap) for rel, cap in core_figure_specs) if x]
    forensics_figs = [x for x in (_fig_if_exists(base, rel, cap) for rel, cap in forensics_figure_specs) if x]
    two_figs = [x for x in (_fig_if_exists(base, rel, cap) for rel, cap in two_model_specs) if x]

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md_parts: list[str] = []
    md_parts.append(f"# Results Appendix: SWE-PolyBench Campaign {run_tag}\n")
    md_parts.append(f"_Generated on {now}._\n")
    md_parts.append(
        "## A1. Scope, Execution Integrity, and Reporting Boundary\n\n"
        f"This appendix documents the workload-scale campaign run `{run_tag}` over five models with a nominal budget of 100 tasks per model, yielding `{total_runs}` total task-runs. "
        f"The campaign produced `{total_success}` validated successes and `{total_failure}` failures, corresponding to an aggregate success rate of `{overall_success_rate:.3f}`. "
        "Claims in the main text should separate operational reliability effects from agent behavior effects. "
        f"For behavior-focused comparisons, the clean non-infrastructure subset contains `{clean_runs}` runs with `{clean_success}` successes (`{clean_success_rate:.3f}`).\n"
    )

    md_parts.append("## A2. Model Outcomes on All Rows\n")
    md_parts.append(
        "Table A1 reports full-campaign model outcomes. These values include infrastructure-prefetch failures and should be interpreted as operational outcomes under full stack conditions.\n"
    )
    md_parts.append(_md_table(all_rank, max_rows=10))

    md_parts.append("## A3. Model Outcomes on the Clean Non-Infrastructure Slice\n")
    md_parts.append(
        "Table A2 restricts to rows without infrastructure-prefetch failure. This table is the primary source for behavior-centric comparisons under this campaign.\n"
    )
    md_parts.append(_md_table(clean_rank, max_rows=10))

    md_parts.append("## A4. Probe Observability (`SCR_probe`) and Coverage\n")
    md_parts.append(
        "Table A3 summarizes probe availability and per-run `SCR_probe` statistics by model. "
        "Coverage must be read jointly with divergence estimates: low-coverage models support weaker comparative claims.\n"
    )
    md_parts.append(_md_table(probe_rank, max_rows=10))

    md_parts.append("## A5. Failure Composition and Forensics\n")
    md_parts.append(
        "Table A4 reports failure-class rates by model. Table A5 and Table A6 provide category-level and repository-level failure concentration to support failure forensics discussion.\n"
    )
    md_parts.append(_md_table(fail_pivot, max_rows=10))
    md_parts.append(_md_table(for_cat, max_rows=10))
    md_parts.append(_md_table(for_repo, max_rows=15))

    md_parts.append("## A6. Error Categories and Hard Instances\n")
    md_parts.append(
        "Table A7 reports recorded error categories. Table A8 lists the top hard instances in the non-infrastructure slice, useful for qualitative trace follow-up.\n"
    )
    md_parts.append(_md_table(error_counts, max_rows=15))
    md_parts.append(_md_table(hard_instances, max_rows=20))

    md_parts.append("## A7. Figure Pack\n")
    md_parts.append(
        "The figures below are appendix-ready and referenced with stable numbering to support manuscript integration and rebuttal workflows.\n"
    )
    for rel, cap in [*core_figs, *forensics_figs, *two_figs]:
        md_parts.append(f"![{cap}](../data/results/{rel})\n")
        md_parts.append(f"*{cap}*\n")

    md_parts.append("## A8. Artifact Manifest and Regeneration\n")
    md_parts.append(
        "All primary inputs for this appendix are located under `data/results/` and were generated by closeout and forensics scripts bound to this run tag. "
        "The following commands regenerate the appendix inputs and this appendix artifact.\n\n"
        "```bash\n"
        f"./.venv/bin/python analysis/closeout_swepolybench_campaign.py --run-tag {run_tag} --models-json benchmarks/models.openrouter.remaining5.json\n"
        f"./.venv/bin/python analysis/failure_forensics_swepolybench.py --run-tag {run_tag}\n"
        f"./.venv/bin/python analysis/build_results_appendix.py --run-tag {run_tag} --two-model-tag {two_tag}\n"
        "```\n"
    )

    md_out.write_text("\n".join(md_parts), encoding="utf-8")

    def _fig_block(rel: str, cap: str) -> str:
        return (
            "<figure class=\"fig\">"
            f"<img src=\"{rel}\" alt=\"{Path(rel).name}\" loading=\"lazy\" />"
            f"<figcaption>{cap}</figcaption>"
            "</figure>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Results Appendix ({run_tag})</title>
  <style>
    :root {{
      --bg: #f5f7fb;
      --card: #ffffff;
      --ink: #1f2937;
      --muted: #6b7280;
      --line: #e5e7eb;
      --accent: #0b6bcb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(1300px 800px at 10% -10%, #dbeafe 0%, rgba(219,234,254,0) 55%),
        radial-gradient(1200px 700px at 90% 0%, #e0f2fe 0%, rgba(224,242,254,0) 50%),
        var(--bg);
      font-family: "Source Serif 4", "Georgia", "Times New Roman", serif;
      line-height: 1.55;
    }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    header {{ margin-bottom: 16px; }}
    h1 {{
      margin: 0 0 4px;
      font-size: 2rem;
      letter-spacing: 0.2px;
    }}
    .sub {{ color: var(--muted); font-size: 0.95rem; }}
    section {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
      margin: 14px 0;
      box-shadow: 0 2px 10px rgba(2, 6, 23, 0.04);
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 1.25rem;
      color: #0f172a;
      border-left: 4px solid var(--accent);
      padding-left: 10px;
    }}
    p {{ margin: 0 0 10px; }}
    .tbl {{
      width: 100%;
      border-collapse: collapse;
      overflow-x: auto;
      display: block;
      background: #fff;
    }}
    .tbl th, .tbl td {{
      border: 1px solid var(--line);
      padding: 6px 8px;
      font-size: 0.9rem;
      white-space: nowrap;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    }}
    .tbl th {{ background: #f8fafc; text-align: left; }}
    .fig-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 14px;
    }}
    .fig {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
      background: #fff;
    }}
    .fig img {{
      width: 100%;
      height: auto;
      display: block;
      background: #fff;
    }}
    .fig figcaption {{
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      font-size: 0.84rem;
      padding: 8px 10px;
      color: #374151;
      border-top: 1px solid var(--line);
    }}
    code {{
      font-family: "IBM Plex Mono", "Consolas", monospace;
      background: #f1f5f9;
      padding: 1px 4px;
      border-radius: 4px;
    }}
    pre {{
      font-family: "IBM Plex Mono", "Consolas", monospace;
      background: #0b1220;
      color: #dbeafe;
      padding: 10px 12px;
      border-radius: 10px;
      overflow-x: auto;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>Results Appendix: SWE-PolyBench Campaign {run_tag}</h1>
      <div class="sub">Generated {now}</div>
    </header>

    <section id="a1">
      <h2>A1. Scope, Execution Integrity, and Reporting Boundary</h2>
      <p>This appendix documents campaign <code>{run_tag}</code> over five models and {total_runs} task-runs. The campaign produced {total_success} validated successes and {total_failure} failures, for aggregate success rate {overall_success_rate:.3f}.</p>
      <p>Behavior-focused model comparisons in the manuscript should rely on the clean non-infrastructure slice ({clean_runs} runs, {clean_success} successes, success rate {clean_success_rate:.3f}), while all-row tables support operational reliability claims.</p>
    </section>

    <section id="a2">
      <h2>A2. Model Outcomes on All Rows</h2>
      {_html_table(all_rank)}
    </section>

    <section id="a3">
      <h2>A3. Model Outcomes on the Clean Non-Infrastructure Slice</h2>
      {_html_table(clean_rank)}
    </section>

    <section id="a4">
      <h2>A4. Probe Observability (SCR_probe) and Coverage</h2>
      {_html_table(probe_rank)}
    </section>

    <section id="a5">
      <h2>A5. Failure Composition and Forensics</h2>
      {_html_table(fail_pivot)}
      {_html_table(for_cat)}
      {_html_table(for_repo)}
    </section>

    <section id="a6">
      <h2>A6. Error Categories and Hard Instances</h2>
      {_html_table(error_counts)}
      {_html_table(hard_instances, max_rows=30)}
    </section>

    <section id="a7">
      <h2>A7. Figure Pack</h2>
      <div class="fig-grid">
        {"".join(_fig_block(rel, cap) for rel, cap in [*core_figs, *forensics_figs, *two_figs])}
      </div>
    </section>

    <section id="a8">
      <h2>A8. Artifact Manifest and Regeneration</h2>
      <p>Regenerate closeout, forensics, and appendix artifacts with:</p>
      <pre>./.venv/bin/python analysis/closeout_swepolybench_campaign.py --run-tag {run_tag} --models-json benchmarks/models.openrouter.remaining5.json
./.venv/bin/python analysis/failure_forensics_swepolybench.py --run-tag {run_tag}
./.venv/bin/python analysis/build_results_appendix.py --run-tag {run_tag} --two-model-tag {two_tag}</pre>
    </section>
  </div>
</body>
</html>
"""
    html_out.write_text(html, encoding="utf-8")

    print(f"[ok] markdown appendix: {md_out}")
    print(f"[ok] html appendix: {html_out}")


if __name__ == "__main__":
    main()
