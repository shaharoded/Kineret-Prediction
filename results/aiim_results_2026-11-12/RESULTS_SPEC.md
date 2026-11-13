# Imported AIIM aggregate results - 12 November 2026

Imported from `Downloads/for_export_2.zip`, nested `aiim_results.zip`.
Original archive files are retained unchanged except Jupyter checkpoint directories are omitted.
`import_manifest.json` records SHA-256 checksums of the 101 imported files.
No model weights or raw/patient-level tables were imported. CSV headers were checked for direct patient, person, sample, visit and admission identifiers.

## Observed result scope
- Primary: eight configurations; K=4; 4,508 test admissions / 3,193 people; eight retained outcome heads.
- Four-day support: training 20,994 admissions / 14,932 people; validation 4,427 / 3,168. These are not augmented training-example totals.
- Metrics: per-outcome, macro and support-weighted AUPRC/AUROC/Brier; positive-only onset MAE; length-of-stay MAE.
- Paired comparisons: KB vs sigma, KB vs SAX, SAX vs sigma, KB vs STraTS, and QA in KB/STraTS/LogReg. Person-clustered bootstrap with 2,000 draws, validity flags, centered approximate tests and explicit BH families.
- Multi-K: six non-LogReg configurations, K=2..6; fixed forecast end at day 14; paired comparisons within each K.
- LOO: KB only, six hospital folds. Bz aggregate intervals have 1,710 valid draws (85.5%) and are flagged unreliable.
- Calibration: raw and validation-temperature variants where available; raw-only LogReg variants.
- Subgroups: KB and KB+QA by age band, sex code and hospital. Some aggregate intervals are unreliable/undefined. Sex coding dictionary is absent.
- XAI: four transcoder layers; 13,013 validation tokens each; 300 forward checks; named-concept aggregate sensitivity scores and exploratory selected-concept intervals.
- Stability: 500 people; 26 eligible direct numeric states and associated trends; one derived/multivariate state excluded. Four noise/removal conditions.
- Geometry: original-space cosine summaries and two-panel UMAP export with detail panels.

## Intended recipe versus verified execution
The repository notebook specifies August 2022 through December 2025 evaluation, a 96-hour primary observation window and labels in (96,336] hours. Historical checkpoint training provenance is unchanged and can include July. The original eight heads remain fixed even though infection support in the newly filtered training subset is below 1%.

The archive omits requested_recipe.json, checkpoint/input manifests, cohort provenance, status files, completed XAI design and the executed notebook. `source_snapshot.json` fingerprints the local code consulted during manuscript preparation; it does NOT identify the exact VM execution version. Recover run-specific provenance from the secure environment before making exact training/configuration claims.

## Manuscript production
The manuscript is in `papers-drafts/KineretPrediction/AIIM2027/`.
`prepare_results.py` reads this folder and regenerates manuscript tables, figures, and a detailed audit. Imported CSVs are never rewritten by it.
The historical thesis exports remain separately in `results/full_bench_results/`.
