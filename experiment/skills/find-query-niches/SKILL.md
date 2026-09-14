---
name: find-query-niches
description: Select query niches in NicheQueryBenchmark by size, composition complexity, prevalence, and pairwise non-overlap, then generate per-niche CSV, PNG, and HTML reports. Use when asked to find benchmark query center cells on a source slice.
---

# Find query niches

Default target: exactly **3 large, simple niches**, represented by **3 distinct center cell IDs**, on **C57BL6J-638850.28**. Use another slice or dimension pair when explicitly requested. Creating or editing this skill alone does not mean running the selection workflow.

Work in `/home/sheryl/niche_query/NicheQueryBenchmark` using `.venv/bin/python`. Read `experiment/manifests/query_niche_dimensions.json` for the actual dimension rules. Currently large means at least 300 cells; simple means 1–2 nonzero parcellations and a dominant fraction in [0.85, 1.0]. Preserve cell IDs as strings throughout.

## Paths and inputs

Set these variables in the repository root for the default target:

```bash
cd /home/sheryl/niche_query/NicheQueryBenchmark
source_slice=C57BL6J-638850.28
dim1=large
dim2=simple
metrics_root=experiment/query_niche_metrics
dimension_candidates="$metrics_root/filtered_query_niches_on_dimensions/$dim1/$dim2/${source_slice}.csv"
final_candidates="$metrics_root/final_query_niche_candidates/$dim1/$dim2/${source_slice}.csv"
report_root="$metrics_root/niche_visualizations/agent_proposed/$dim1/$dim2"
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

No dimension argument is needed: size is inferred from the `$dim1/$dim2` input directories. Each candidate must have **>100** similar niches in the source slice and **>=30 in every other available slice CSV** of that size. Counts use all eligible exported niches, not only dimension-filtered candidates.

Similarity follows `experiment/visualize_query_niche_v2.ipynb`: every parcellation fraction differs strictly by <0.05 in the source and <0.25 in other slices. Align by parcellation ID; absent IDs mean zero. Equality within 1e-12 is excluded. Source counts include the selected niche and overlapping neighborhoods. A target CSV with no eligible niches rejects every candidate; slices without CSVs are not compared. Disclose if no other slice CSVs exist, since cross-slice prevalence then cannot be demonstrated.

## 3. Select exactly three mutually compatible niches

Read the final candidate CSV, then stream the source metrics CSV to recover the full rows for those centers. Raise `csv.field_size_limit` to 100,000,000 before parsing large membership fields. Apply the same size eligibility checks as the filters. Reject duplicate or missing candidate records; verify each saved membership is a unique string-ID list, includes its center, and matches `niche_cell_count`.

For each candidate retain:

- Its string center ID and the set of all `niche_cell_names`.
- Its set of parcellation IDs with **strictly positive** fractions. Prefer per-row IDs over the shared ID order; validate vector length and fractions before alignment. Compare IDs, never array positions. Zero fractions do not imply representation.

For every selected pair A, B, both conditions must hold:

```python
members[A].isdisjoint(members[B])
parcellations[A].isdisjoint(parcellations[B])
```

These constraints refer to the three niches on the same source slice. Distinct centers alone do not prove non-overlap. Checking only dominant parcellations is insufficient.

Search for a compatible triple in final-candidate CSV order, using backtracking or compatible-pair intersections. Prune pairs with shared members or parcellations; do not materialize every triple. A greedy choice can miss a valid triple, so backtrack before declaring failure. Stop once one valid triple is found. Do not arbitrarily truncate candidates or add undocumented ranking criteria.

If fewer than three candidates remain or an exhaustive search finds no compatible triple, report that the requested target cannot be met and identify the failing stage. If the search is interrupted, report it as incomplete, not proof of impossibility. Never relax prevalence or overlap requirements silently or present a partial selection as success.

Save the successful selection to `$report_root/${source_slice}_selected_centers.csv` with exactly three data rows and columns `source_slice`, `center_cell_name`, `niche_dimension`, `composition_complexity`, `niche_cell_count`, and `parcellation_ids` (a sorted JSON array). Save `$report_root/${source_slice}_selection_checks.json` recording the candidate-input path, source metrics path, dimension rules, and the member/parcellation intersection counts for all three pairs; all six counts must be zero. Create the report directory as needed. Keep the full final-candidate CSV intact.

## 4. Generate and verify the three reports

For each selected center, set `center_cell_name` to its exact string ID and run:

```bash
.venv/bin/python experiment/visualize_query_niche.py \
  --source-slice "$source_slice" \
  --cell-id "$center_cell_name" \
  --all-niche-candidates-dir "$metrics_root/$dim1" \
  --output-dir "$report_root/${source_slice}_${center_cell_name}"
```

The candidate-directory argument must point to **all slice metrics in the selected size directory**, not either filtered-candidate directory. The visualization script derives size from that directory and defaults to `data/20260601_225717` for the source H5AD. Use `--data-dir` if the user specifies another data location.

Verify all nine outputs for each selected center:

- `metrics.csv`, `composition_summary.csv`, `center_coordinates.csv`
- `members.csv`, `composition.csv`, `source_composition_similarity.csv`
- `target_slice_composition.csv`, `niche.png`, `report.html`

Recheck the source similar count (>100) and every target similar count (>=30) from the reports. Confirm reported memberships and positive-fraction parcellation IDs still satisfy all pairwise non-overlap checks. Inspect each PNG to confirm the full-slice and close-up panels, highlighted membership, and center marker are present. Preserve the notebook's calculations and report format.

Finish with exactly three center IDs, their represented parcellation IDs, links to their report directories or HTML files, and the saved overlap evidence. If a report or verification fails, identify it and do not claim the workflow is complete.
