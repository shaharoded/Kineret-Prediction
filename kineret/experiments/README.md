# Final Kineret recipe

Run only `notebooks/benchmark.ipynb` for the planned results. All analyses
are configured and activated there; there are no separate experiment notebooks. The fixed primary task is
observe 96 hours and predict events in (96,336] hours. Training augmentation
remains K=1..13. The study starts on **2022-08-01**; existing checkpoint weights
keep their original training dates. The thesis appendix and JAMIA methods agree
on the August start. The archived thesis configuration starts in July, so the
new filtered evaluation must be distinguished from the historical thesis run.

## Activation and artifacts

| Analysis | Activation | Main exports |
|---|---|---|
| Eight-arm evaluation | `RUN['primary']` | Admission/person manifests, per-arm predictions, person-clustered CIs, status |
| Paired contrasts | `RUN['paired']` | Per-outcome and macro/weighted differences, 95% CIs, centered-bootstrap approximate p-values, explicit BH families |
| Age/sex/hospital groups | `RUN['subgroups']` | Support, prevalence, people, metrics and CIs for KB and KB+QA |
| Calibration | `RUN['calibration']` | Per-outcome ECE/Brier, reliability bin counts, optional validation-only temperature scaling |
| K=2,3,4,5,6 | `RUN['multi_k']` | Frozen-checkpoint metrics/predictions, support at each K, curves and paired contrasts at each K |
| Hospital LOO | `RUN['loo']` | One site per fold, disjoint people, fold checkpoints, paired differences and metrics |
| Trained hierarchy | `RUN['umap']` | Boundary-token UMAP and original-space cosine distances |
| Sparse transcoders | `RUN['xai']` | Held-out reconstruction/sparsity, forward equality, named-concept contributions, clinical review template |
| Mediator stability | `RUN['stability']` | Definition coverage, sampled threshold proximity, perturbed interval agreement and lost/added coverage |
| Missing primary models | `RUN['train_missing']` | Separate new model artifacts; old checkpoints or predictions prevent retraining |

The first four and missing-primary-model training are on by default. Existing primary models are reused. Expensive analyses are explicit opt-ins.
The default LOO arms are STraTS, KB, KB+QA: 18 fits for six sites. Add sigma
to test the knowledge-base contrast at unseen sites: 24 fits. Completed matching
fold models are reused; pooled checkpoints cannot answer unseen-site questions.

CSV, JSON, PNG and PDF exports live beneath the notebook's experiment root.
Private admission-level exports stay in that environment. The deployment script
ships code/configuration, excluding data, outputs and checkpoints.

## Resource and reuse rules

- CSV inference reads source tables in chunks, retaining selected admissions.
- Fold/model execution is sequential. New fold preprocessing and weights have
  separate directories; the original cohort pickle is never swapped out.
- Frozen inference uses saved vocabulary/scalers and never calls training.
- Changing evaluation dates preserves the original split and outcome heads.
- Cached exports have design/input/checkpoint manifests. A changed design does
  not authorize overwriting a fold checkpoint.
- Sparse-transcoder capture fits one layer at a time and stores at most the
  configured number of tokens per sampled admission. Predictor weights are frozen.

Old STraTS weights need their preprocessing schema recovered by replaying the
original seeded processing on the **original untouched pickle**. Compare recovered
K=4 predictions against the existing outputs before using that schema. New fits
export the schema automatically. Old LogReg runs saved coefficients, not a complete
inference bundle; existing K=4 predictions can be used, while new K inference needs
a complete bundle from a separately activated fit. A missing artifact never
silently triggers retraining.

Legacy sigma inputs fitted cutoffs on the supplied cohort. Existing weights keep
that preprocessing. New fold sigma cutoffs use training admissions only. A raw-vs-KB
contrast also contains the deliberate difference in access to non-target KB events.
Describe it as the value of the KB representation package, rather than a perfectly
isolated comparison of clinical thresholds alone.

## Claims and limits

Person bootstrap measures sampling uncertainty conditional on a trained model.
Repeated training seeds are outside scope. Report point estimates, difference CIs,
positive support and valid bootstrap draws. A CI overlapping zero is inconclusive;
it does not prove equivalence or redundancy of QA. Per-outcome BH families are
outcomes within each contrast/metric. Headline families are contrasts within each
metric/average. P-values use the bootstrap error distribution centered under zero
and a finite-sample correction; they are approximate bootstrap tests.

At different K values the eligible admissions, outcome prevalence and remaining
horizon change. Compare relative model performance at each clinical prediction
time. This is not an estimate of the isolated benefit of one extra day of history.

Attribution supports this wording: *Local attribution on the trained model identifies
named clinical states, trends and QA patterns associated with its predictions;
selected patterns can be assessed for coherence with clinical knowledge.*
Names come from the input vocabulary. Latent sparse features are not automatically
clinical concepts. Scores are local gradient×activation sensitivities, not causal
treatment effects or an exact additive explanation. Clinical coherence requires
review of the exported template; familiar names alone are insufficient. Transcoder
quality and model-forward equality are reported separately. Poor reconstruction
should temper interpretation; it must not be concealed by a plausible plot.

Mediator stability reruns the actual direct numeric state/trend implementations
inside the first 96 hours. It covers a hospital/density-stratified sample with one
admission per person, weighted to that sampling frame. It excludes derived numeric
concepts, QA patterns and downstream predictive robustness; exclusions are exported.
UMAP describes the trained input representation; cosine distances are measured in
the original embedding space, without claiming that hierarchy caused accuracy gains.

## Fixed SAX arm

`intervene_sax` is registered as an additional arm with no QA companion.
Its configuration is fixed in `kineret/sax.py` and displayed/exported by the main
notebook: per-concept population mean/std fitted only on training admissions,
four-hour segment means, seven fixed Gaussian-quantile symbols, no filling of
empty segments, and merging of contiguous identical symbols after prefix clipping.
This is a SAX-style adaptation to irregular EMR that retains differences in
absolute levels between patients. Canonical SAX uses per-series normalization:
<https://www.cs.ucr.edu/~eamonn/SAX.pdf>.

Segments become available at their ending timestamp. The segment (92,96] is
available at K=4; the segment (96,100] is excluded even though it starts at the
cutoff. Measurements exactly on a segment boundary belong to the segment ending
there. This is enforced for training augmentation and frozen multi-K inference.

SAX is included in `ARMS` with the other benchmark models.
`RUN['train_missing']` trains missing primary arms uniformly in separate experiment
folders, while existing arms keep their checkpoints. A completed SAX checkpoint
is reused on later notebook runs. Training statistics, Gaussian breakpoints in
the fixed recipe, hierarchy, atomic temporal table and design manifest accompany
its weights. Primary metrics, calibration, paired comparisons against KB/sigma,
and the configured K sensitivity include SAX. LOO remains the agreed three-arm
run unless SAX is explicitly added to `LOO_ARMS`.

SAX inherits the sigma arm's knowledge-free event access, receives static context,
and uses the same prediction heads and training schedule. The original LR/STraTS
QA controls remain in the established benchmark; no SAX+QA arm is introduced.
Multi-seed and point-event ablations remain outside scope.

## LOO progress

`LOO_ARMS` and `LOO_HOSPITALS` in the setup cell select the models and sites.
The default is KB only. Each selected model requires one fit per selected site;
completed checkpoints are reused. A narrower arm selection can reuse an existing
fold design; expanding it requires a new experiment directory.

LOO reports fold number, hospital and model before each fit. A compact live
status displays phase/epoch and batch progress, refreshed at most every five
seconds. Verbose stdout/stderr goes to `loo/<fold>/logs/<arm>.log`; evaluation
output goes to `logs/evaluation.log`. Completion/interruption is shown explicitly.
Epoch/phase ETA is local to that progress bar, not a total-study ETA.
