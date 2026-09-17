---
name: find-query-niches
description: Select query niches in NicheQueryBenchmark by size, composition complexity, prevalence, and pairwise non-overlap. Generate per-niche CSV, PNG, and HTML reports only when explicitly requested. Use when asked to find benchmark query center cells on a source slice.
---

# Find query niches

Default target: select exactly **k mutually compatible niches**, with `k=3` unless the user requests another value, aiming for large, simple niches represented by distinct center cell IDs on **C57BL6J-638850.28**. If fewer than k valid niches survive the filters or compatibility checks, report the achieved count and shortfall rather than relaxing constraints. Use another slice or dimension pair when explicitly requested. Creating or editing this skill alone does not mean running the selection workflow.

Work in `/home/sheryl/niche_query/NicheQueryBenchmark` using `.venv/bin/python`. Read `experiment/manifests/query_niche_dimensions.json` for the actual dimension rules. Currently large means at least 300 cells; simple means 1–2 nonzero parcellations and a dominant fraction in [0.85, 1.0]. Preserve cell IDs as strings throughout.

## Paths and inputs

Set these variables in the repository root for the default target:

```bash
cd /home/sheryl/niche_query/NicheQueryBenchmark
source_slice=C57BL6J-638850.28
dim1=large
dim2=simple
k=3
metrics_root=experiment/query_niches
dimension_candidates="$metrics_root/filtered_query_niches_on_dimensions/$dim1/$dim2/${source_slice}.csv"
final_candidates="$metrics_root/final_query_niche_candidates/$dim1/$dim2/${source_slice}.csv"
report_root="$metrics_root/selected_query_niches/agent_proposed/$dim1/$dim2"
selected_centers_csv="$report_root/${source_slice}_selected_centers.csv"
selection_checks_json="$report_root/${source_slice}_selection_checks.json"
```

All exported niches for a slice are in `$metrics_root/$dim1/$source_slice.csv`. This is a CSV, not a directory. Other slice CSVs in the same size directory provide the comparison populations. Shared fraction-vector IDs are in `$metrics_root/$dim1/parcellation_ids.json`; legacy rows can instead contain `parcellation_ids`.

Use the plural `filtered_query_niches_on_dimensions` consistently. The actual dimension filter is `filter_query_niches_on_dimensions.py`, not the obsolete `filter_query_niches.py`. Check that required metrics, scripts, and source H5AD exist before starting expensive work. Do not regenerate or change the raw metrics to satisfy selection criteria.

## 1. Filter by dimensions

```bash
.venv/bin/python experiment/filter_query_niches_on_dimensions.py \
  --source-slice "$source_slice" \
  --dimension "$dim1" "$dim2" \
  --metrics-dir "$metrics_root" \
  --output-file "$dimension_candidates"
```

The output contains candidate `center_cell_name` values. A header-only output means no dimension-qualified candidates.

## 2. Filter by prevalence

```bash
.venv/bin/python experiment/filter_query_niche_on_prevalence.py \
  --input-file "$dimension_candidates" \
  --source-slice "$source_slice" \
  --metrics-dir "$metrics_root" \
  --output-file "$final_candidates"
```

No dimension argument is needed: size is inferred from the `$dim1/$dim2` input directories. Each candidate must have **>100** similar niches in the source slice and **>=30 in at least 6 other available slice CSVs** of that size (6 out of the current 11 other slices; up to 5 may fail). This is a minimum of 6 qualifying target slices, not a percentage. Counts use all eligible exported niches, not only dimension-filtered candidates.

Similarity follows `experiment/visualize_query_niche_v2.ipynb`: every parcellation fraction differs strictly by <0.05 in the source and <0.25 in other slices. Align by parcellation ID; absent IDs mean zero. Equality within 1e-12 is excluded. Source counts include the selected niche and overlapping neighborhoods. A target CSV with no eligible niches contributes no qualifying matches; slices without CSVs are not compared. Fewer than 6 available target slice CSVs means no candidate can qualify. The final candidate CSV records qualifying target slice IDs and their similar-niche counts in `target_slices` and `target_similar_niche_counts`.

## 3. Select exactly k mutually compatible niches

Use `experiment/select_non_overlapping_niches.py` for selection:

```bash
.venv/bin/python experiment/select_non_overlapping_niches.py -k "$k" \
  --candidates "$final_candidates" \
  --preprocessed "experiment/query_niches/preprocessed/$dim1/${source_slice}.csv" \
  --output "$selection_checks_json"
```

The script enforces both compatibility rules: selected niches must have disjoint member-cell IDs and disjoint positively represented parcellation IDs. It returns up to `k` niches in candidate CSV order and reports `requested_k`, `achieved_k`, `shortfall`, `center_cell_names`, and `parcellation_ids`.

Always save `$selection_checks_json` and `$selected_centers_csv`, no matter whether the user requested reports. Keep the full final-candidate CSV intact. Derive the selected-centers CSV from the JSON and source preprocessed metrics with one row per selected niche, possibly zero rows, and columns `source_slice`, `center_cell_name`, `niche_dimension`, `composition_complexity`, `niche_cell_count`, and `parcellation_ids` as a sorted JSON array.

Default selection artifact paths:

```text
experiment/query_niches/selected_query_niches/agent_proposed/large/simple/C57BL6J-638850.28_selected_centers.csv
experiment/query_niches/selected_query_niches/agent_proposed/large/simple/C57BL6J-638850.28_selection_checks.json
```

If `achieved_k` is less than `requested_k`, report the achieved count and shortfall. Never relax prevalence or overlap requirements silently.

## 4. Generate reports only when explicitly requested

Only run `experiment/visualize_query_niche.py` when the user explicitly requests reports, HTML files, PNG visualizations, or other per-niche report artifacts.

If the task is only to find or select query niches, stop after saving:

- `$selected_centers_csv`
- `$selection_checks_json`

### Report generation

For each selected center, set `center_cell_name` to the exact center-cell string ID and run:

```bash
.venv/bin/python experiment/visualize_query_niche.py \
  --source-slice "$source_slice" \
  --cell-id "$center_cell_name" \
  --all-niche-candidates-dir "$metrics_root/$dim1" \
  --output-dir "$report_root/${source_slice}_${center_cell_name}"
```

`--all-niche-candidates-dir` must point to the directory containing **all slice metrics for the selected size category** (`$metrics_root/$dim1`). Do not pass either filtered-candidate directory.

The visualization script infers the niche size from this directory and uses `data/20260601_225717` as the default source H5AD directory. Pass `--data-dir` only when the user specifies a different data location.

### Verify each generated report

For every selected center, confirm that all nine expected artifacts exist:

- `metrics.csv`
- `composition_summary.csv`
- `center_coordinates.csv`
- `members.csv`
- `composition.csv`
- `source_composition_similarity.csv`
- `target_slice_composition.csv`
- `niche.png`
- `report.html`

Then verify that:

- the source slice has more than 100 similar niches;
- at least 6 target slices each contain at least 30 similar niches;
- counts for **all** target slices remain in the report, including slices with fewer than 30 similar niches;
- reported memberships and positive-fraction parcellation IDs still satisfy all pairwise non-overlap constraints;
- `niche.png` contains the full-slice and close-up panels, highlighted niche membership, and the center-cell marker.

Do not require every target slice to meet the `>=30` threshold.

Preserve the notebook's existing calculations and report format.

### Final output

If reports were explicitly requested, return:

- every selected center ID;
- the parcellation IDs represented by each center;
- a link or path to each report directory or `report.html`;
- the requested number of niches versus the number successfully selected;
- the saved `$selected_centers_csv`;
- the saved `$selection_checks_json`.

If any report generation or verification step fails, clearly identify the failure and do not state that the report workflow completed successfully.

If reports were **not** explicitly requested, return:

- every selected center ID;
- the parcellation IDs represented by each center;
- the requested number of niches versus the number successfully selected;
- the saved `$selected_centers_csv`;
- the saved `$selection_checks_json`.

Also state that report generation was skipped because it was not explicitly requested.
