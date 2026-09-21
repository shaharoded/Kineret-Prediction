# Kineret-Prediction

Complication prediction on Mediator-abstracted EMR data — the closing
**transferability study** for the thesis, moving the ladder built on MIMIC and
Rambam onto the six-hospital **Kineret** cohort. Manuscript in preparation.

The repo pins the shared cohort, runs the full seven-arm comparison, and
snapshots the exact outputs quoted in the thesis under `results/`.

## Where this sits in the wider work

| Repo | Role in the pipeline |
|---|---|
| [Kineret-ETL](https://github.com/shaharoded/Kineret-ETL) | OMOP CDM → cleaned event tables (the inputs this repo consumes) |
| [Mediator](https://github.com/shaharoded/Mediator) | Temporal-abstraction engine that produces `mediator_output.csv` and the QA scores |
| [INTERVenE-Enc](https://github.com/shaharoded/INTERVenE-Enc) | Reference implementation of the interval-transformer arm — vendored under `kineret/intervene/` |
| [Medical-Transformers-Benchmark](https://github.com/shaharoded/Medical-Transformers-Benchmark) | Reference ss-STraTS baseline — vendored under `kineret/strats/` |

The ETL feeds Mediator; Mediator feeds this repo; the two model repos supply the
architectures. Everything cohort- and study-level is defined here.

---

## The task

Observe the first **K days** of an admission; predict, per complication:
whether it occurs before day 14, **when**, and how long the stay will be.

```text
inputs :  events in [0, K*24]                hours from ADMISSION
labels :  outcome occurs in (K*24, 14*24]    hours from ADMISSION
```

**K is a training-time augmentation, not a sweep.** Each training admission
contributes one sample per context window it reaches — same trajectory, cut at
a different point, label window re-derived to match. Long stays contribute
more samples; short stays teach the model without ever being scored.
**Evaluation happens at a single window**, chosen from the support table before
training. Every reported number comes from that one window.

Every arm answers the same three questions per complication:

| | Question | Head | Trained on |
|---|---|---|---|
| risk | does it occur in `(K, 14]`? | multi-label BCE | all samples |
| onset | if so, when? | per-outcome regression | positives only |
| LoS | how long is the stay? | regression | discharged only |

---

## The seven arms

| Rung | Arm | Input |
|---|---|---|
| 1 | LogReg (raw) | raw measurements, distribution-summarised — no temporal model |
| 2 | LogReg (raw + QA) | + aggregated compliance in the context vector |
| 3 | ss-STraTS (raw) | raw measurements + learned temporal attention |
| 4 | ss-STraTS (raw + QA) | + aggregated compliance in the context vector |
| 5 | INTERVenE-Enc (σ-bins) | knowledge-**free** intervals over the same raw data |
| 6 | INTERVenE-Enc (KB) | knowledge-**based** Mediator abstractions |
| 7 | INTERVenE-Enc (KB + QA) | + treatment-quality pattern tokens |

Rungs **5 → 6** isolate the knowledge base; **6 → 7** isolate the
treatment-quality signal; rungs **2 / 4** ask the same QA question of the models
that cannot represent pattern tokens, so the QA effect is separable from the
architecture that consumes it. Every number carries a 95 % interval from a
2 000-resample patient-level bootstrap.

---

## What makes the comparison valid

Everything the arms share is computed **once**, in `kineret/cohort.py`:

- **Study window** applied in one place to every source table.
- **Event support** is harmonised — the raw file and the Mediator output are
  reconciled into one canonical event table, re-injected into every arm's
  input stream. Support is identical in every arm by construction.
- **Cohort membership** is keyed on the shortest training window; val/test are
  filtered separately to the evaluation window.
- **One split, over people, not admissions**, frozen at build time.
- **One scorer + one bootstrap** read one prediction schema.

**Knowledge containment.** Non-target `*_EVENT` concepts (KB verdicts like
`AKI_EVENT`) are withheld from the raw-data arms and kept for the KB arms.
Rungs 5 → 6 are otherwise measuring a leak, not abstraction.
`unittests/test_knowledge_containment.py` asserts both directions.

**Leakage guards.** Terminus markers (`RELEASE`, `DEATH`) are stripped from every
input stream. Outcome events are harmonised and clipped to the observation
window by each arm's own truncation; the label window starts strictly after the
input window.

The wiring check fails loudly if any two arms were scored on different labels
or different test patients.

---

## Data assumptions

Four properties of the Kineret extract shape the pipeline. Each is switchable
and each reports what it did in section 2 of the notebook.

1. **The target list is authoritative.** `targets=[...]` decides what gets a
   prediction head; the alias table only says how each complication is *spelled*.
2. **The raw file's own events must be derived.** `mediator_input.csv` holds
   measurements, not complications, so `kineret/raw_events.py` rebuilds the raw
   view by applying the Mediator XML thresholds before alignment. The shipped
   `RAW_EVENT_RULES` are placeholders — tune each rule until its derived-vs-Mediator
   ratio approaches 1.00.
3. **Cross-file alignment removes small support drift.** For a target both
   files carry, an occurrence is credited only when the two files date it
   identically (within `event_align_tolerance_min`).
4. **The admission is the unit of analysis.** `mediator_input.csv` carries both
   `PatientId` (person) and `VisitId` (admission). Every temporal table is
   re-keyed onto the admission; `test_identity.py` asserts >90 % id overlap.

Full detail on each — including how they were discovered against the real
extract and how they used to fail silently — lives in the thesis.

---

## Running it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\activate
pip install -e .
```

Drop the four source tables into `data/source/` under their canonical names,
then open **`notebooks/benchmark.ipynb`** and run it top to bottom.

| Section | What it does |
|---|---|
| 1 | Study configuration — every shared decision, in one cell |
| 2 | Target support; choose the evaluation window from evidence |
| 3 | Wiring check — all seven arms, one epoch each, minutes |
| 4 | Train the ladder — all seven arms, one pass |
| 5–9 | Tables and paper figures, read back from `outputs/` |

Section 4 skips any arm that already has a `test_predictions.csv`. Delete an
arm's directory to force a retrain. All three models are PyTorch — the LogReg
included — so everything trains on the same device.

**No data yet?** `python scripts/make_synthetic_data.py --patients 400 --out
data/source_synth` generates a synthetic drop that reproduces the raw-vs-Mediator
naming split, so the alias reconciliation is exercised rather than skipped.

**Unattended runs.**

```bash
nohup jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=-1 notebooks/benchmark.ipynb > run.log 2>&1 &
```

---

## Input contract

Four tables in `data/source/`, under **exactly these names** (`.csv` or `.xlsx`):

| File | Role | Consumed by |
|---|---|---|
| `mediator_input.csv` | Mediator **input** — measurements, administrations, structural events | LogReg, ss-STraTS, σ-bin arm |
| `mediator_output.csv` | Mediator **output** — abstractions + `*_EVENT` rows | INTERVenE-Enc (KB arms); labels for every arm |
| `context_data.csv` | one row per admission, static numeric features | all arms |
| `qa_scores.csv` | Mediator QA output — temporal-shaped compliance scores | the QA arms |

`kineret/config/tak_repo_portable.json` — the concept hierarchy INTERVenE-Enc
resolves abstractions through — **ships inside the package**. Replace it with
the Kineret one before running.

Required columns and admission-framing behaviour: see `kineret/io_utils.py` and
section 2 of the notebook. Environment overrides `KINERET_RAW_TEMPORAL`,
`KINERET_ABSTRACT`, `KINERET_CONTEXT`, `KINERET_QA` point at files elsewhere.

---

## Layout

```text
Kineret-Prediction/
├── kineret/
│   ├── config/                     # paths, study defaults, TAK hierarchy
│   ├── cohort.py                   # date range + event reconciliation + labels + samples + splits + QA
│   ├── raw_events.py               # complications derived from raw thresholds
│   ├── abstraction.py              # knowledge-free sigma-bin intervals (rung 5)
│   ├── evaluation.py               # one scorer + patient-level bootstrap
│   ├── benchmark.py                # arms, wiring check, ladder runner, tables
│   ├── figures.py                  # paper figures (PNG + PDF + CSV)
│   ├── io_utils.py                 # excel/csv loading, VisitId re-keying, aliases
│   ├── intervene/                  # INTERVenE-Enc, vendored
│   ├── strats/                     # ss-STraTS, vendored
│   └── logreg/                     # non-interval reference — features + heads
├── notebooks/benchmark.ipynb       # ** start here **
├── scripts/                        # make_synthetic_data.py, package.py
├── unittests/                      # cohort, augmentation, contract, scorer, identity, containment, derivation
├── results/                        # thesis run — see below
├── data/source/                    # <- drop the four tables here
├── pyproject.toml  requirements.txt
└── README.md
```

---

## Results used in the thesis

Frozen under `results/`:

- **`results/full_bench_results/`** — the seven-arm run behind the thesis
  numbers: `benchmark_summary.csv`, per-arm predictions and metrics under
  `runs/`, the paper figures in `figures/` (PNG + PDF + CSV per figure),
  training curves under `training_curves/`, and the exact `study_config.json`
  the notebook ran with. `MANIFEST.txt` lists everything.
- **`results/transformers-results-old-with-xai/`** — earlier ss-STraTS runs
  (with and without QA) kept for their attention/XAI notebooks, superseded by
  the full ladder but preserved for reference.

Reproducing them from scratch needs the Kineret extract, which is not in the
repo.

---

## Configuration

Shared study decisions are set through one `configure_study(...)` call in
section 1 of the notebook. `kineret/config/data_config.py` holds the defaults
and the list of valid keys — an unknown key raises rather than being silently
ignored.

Per-model hyperparameters stay with their model:
`kineret/strats/config.py`, `kineret/intervene/config/model_config.py`,
`kineret/logreg/train.py::LOGREG_SETTINGS`.

---

## Tests

```bash
python -m pytest unittests/ -q
```

Suites carrying the data assumptions specifically: `test_identity.py`,
`test_knowledge_containment.py`, `test_raw_events.py`, `test_contract.py`.
They run on synthetic data and never touch the real cohort.

---

## Deployment

```bash
python scripts/package.py              # -> kineret_deploy.zip
python scripts/package.py --list       # show exactly what would ship
```

Ships an allowlist — the package, the notebook, the scripts, the tests and the
install metadata — so a stray `.csv` / `.xlsx` / `.pkl` / `.pt` / `.bin` /
`.log` can never leave the machine. `data/`, `checkpoints/` and `outputs/` are
never included.

> Re-deploying over an existing install overwrites the bundled TAK repository.
> Back the real one up first:
>
> ```bash
> cp kineret/config/tak_repo_portable.json ~/tak_backup.json
> unzip -o kineret_deploy.zip
> cp ~/tak_backup.json kineret/config/tak_repo_portable.json
> ```

---

## Citation

Manuscript in preparation. For the methodology behind the ladder, the
knowledge-containment argument, and the earlier MIMIC / Rambam stages this
study builds on, see the thesis.
