"""
Generate synthetic source tables shaped like the real Kineret drop.

    python scripts/make_synthetic_data.py --patients 300 --out data/source_synth

Use it to smoke-test the whole pipeline before the real excels are on the VM,
and to verify an environment after deployment. The generated files carry the
same canonical stems as the real drop, so the simplest way to use them is to
copy them into `data/source/` and run the notebook as usual. To keep the two
side by side, point the pipeline at the synthetic directory instead:

    python scripts/make_synthetic_data.py --out data/source_synth
    KINERET_RAW_TEMPORAL=data/source_synth/mediator_input.csv \
    KINERET_ABSTRACT=data/source_synth/mediator_output.csv \
    KINERET_CONTEXT=data/source_synth/context_data.csv \
    KINERET_QA=data/source_synth/qa_scores.csv \
      python -c "from kineret.benchmark import smoke_test; smoke_test()"

The generator deliberately reproduces the naming split that trips people up:
the raw table spells complications WITHOUT the `_EVENT` suffix (and
`CARDIOVASCULAR_DISORDER` without the hyphen), while the Mediator output uses
the canonical `<NAME>_EVENT` form. If the alias resolution in `kineret.cohort`
ever regresses, this data catches it.

Concept names are drawn from `kineret/config/tak_repo_portable.json`, because
INTERVenE's `DataProcessor` resolves every abstraction back to a raw parent
through that repository and raises on anything it does not recognise.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from kineret.config import paths

# Abstractions the generator emits, paired with the value vocabulary the
# Mediator would produce for them.
STATE_VALUES = {
    "GLUCOSE_MEASURE_STATE": ["LOW", "NORMAL", "HIGH", "VERY_HIGH"],
    "CREATININE_SERUM_MEASURE_STATE": ["NORMAL", "HIGH"],
    "HEART_RATE_MEASURE_STATE": ["LOW", "NORMAL", "HIGH"],
    "BODY_TEMPERATURE_MEASURE_STATE": ["NORMAL", "HIGH"],
}
TREND_SUFFIX = ["Increasing", "Decreasing", "Steady"]

# Raw measurements, with a plausible (mean, sd) so the numbers are not nonsense.
RAW_MEASURES = {
    "GLUCOSE_MEASURE": (160.0, 75.0),
    "CREATININE_SERUM_MEASURE": (1.4, 0.9),
    "HEART_RATE_MEASURE": (80.0, 15.0),
    "BODY_TEMPERATURE_MEASURE": (36.9, 0.6),
    "BLOOD_PRESSURE_SYSTOLIC_MEASURE": (125.0, 18.0),
    "BICARBONATE_MEASURE": (24.0, 4.0),
}
RAW_ADMIN = ["BASAL_BITZUA", "BOLUS_BITZUA", "ANTIBIOTIC_IV_BITZUA"]

# Complications the Mediator derives by thresholding a measurement, mirroring
# `data_config.RAW_EVENT_RULES`. (operator, cut-off, nth qualifying reading).
THRESHOLD_EVENTS = {
    "HYPERGLYCEMIA_EVENT":        (">=", 180.0, 2),
    "SEVERE_HYPERGLYCEMIA_EVENT": (">=", 300.0, 1),
    "HYPOGLYCEMIA_EVENT":         ("<",   70.0, 2),
    "SEVERE_HYPOGLYCEMIA_EVENT":  ("<",   54.0, 1),
}

# Canonical outcome -> the spelling the RAW table uses for the same pathology.
OUTCOME_RAW_SPELLING = {
    "INFECTION_EVENT": None,                      # Mediator-derived only
    "ACIDOSIS_EVENT": "ACIDOSIS",
    "KETOACIDOSIS_EVENT": "KETOACIDOSIS",
    "CARDIO-VASCULAR_DISORDER_EVENT": "CARDIOVASCULAR_DISORDER",
    "DEATH_EVENT": "DEATH",
}
# Per-patient probability that each outcome fires at all.
OUTCOME_RATE = {
    "INFECTION_EVENT": 0.18, "ACIDOSIS_EVENT": 0.08,
    "KETOACIDOSIS_EVENT": 0.06, "CARDIO-VASCULAR_DISORDER_EVENT": 0.09,
    "DEATH_EVENT": 0.07,
}

# `*_EVENT` concepts the Mediator derives that are NOT prediction targets.
# They exist so the KB-containment path has something to contain: these must
# reach the INTERVenE KB arms and never the raw-data arms.
KB_ONLY_EVENTS = {
    "AKI_EVENT": 0.14,
    "ELECTROLYTE_DERANGEMENT_EVENT": 0.20,
    "MYOCARDIAL_INJURY_EVENT": 0.07,
}

CONTEXT_COLUMNS = ["age_at_admission", "gender", "admission_type",
                   "has_diabetes_type1", "has_diabetes_type2", "has_hypertension",
                   "has_obesity", "has_ckd", "has_chf"]
QA_PATTERNS = ["BASAL_COMPLIANCE_PATTERN", "BOLUS_COMPLIANCE_PATTERN",
               "GLUCOSE_MONITORING_PATTERN"]


def parse_args():
    """Purpose: CLI surface for the generator."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patients", type=int, default=300)
    p.add_argument("--out", default=os.path.join(paths.DATA_DIR, "source_synth"))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--excel", action="store_true",
                   help="Write .xlsx instead of .csv, matching the real drop.")
    p.add_argument("--min-days", type=float, default=8.0,
                   help="Shortest simulated admission, in days.")
    p.add_argument("--max-days", type=float, default=16.0)
    return p.parse_args()


def _known_concepts():
    """Purpose: Restrict generated concepts to names the TAK repository knows."""
    with open(paths.TAK_REPO_PATH, encoding="utf-8") as f:
        return set(json.load(f)["taks"].keys())


def generate(n_patients, seed, min_days, max_days):
    """
    Purpose: Simulate a cohort of admissions across all four source tables.
    Method:  Each patient gets an admission timestamp, a length of stay, a
             stream of raw measurements every few hours, a matching stream of
             Mediator state/trend intervals, a set of complication events drawn
             from `OUTCOME_RATE`, and QA compliance rows. Complications are
             written to BOTH tables where a raw spelling exists, so the alias
             resolution has something to reconcile.

    Args:
        n_patients (int):   Cohort size.
        seed       (int):   RNG seed.
        min_days   (float): Shortest admission.
        max_days   (float): Longest admission.

    Returns:
        tuple[pd.DataFrame, ...]: (raw, abstract, context, qa).
    """
    rng = np.random.default_rng(seed)
    known = _known_concepts()
    base = pd.Timestamp("2024-01-01")

    raw_rows, abs_rows, ctx_rows, qa_rows = [], [], [], []

    # ~25 % of admissions are a readmission of someone already in the cohort, so
    # the person-grouped split has repeat people to keep on one side.
    person_of, people, window = {}, [], {}
    for i in range(n_patients):
        visit = 100000 + i
        if people and rng.random() < 0.25:
            person_of[visit] = int(rng.choice(people))
        else:
            person_of[visit] = 900000 + len(people)
            people.append(person_of[visit])

    for i in range(n_patients):
        pid = 100000 + i
        admit = base + pd.Timedelta(hours=float(rng.integers(0, 24 * 365)))
        los_h = float(rng.uniform(min_days, max_days)) * 24.0
        window[pid] = (admit, admit + pd.Timedelta(hours=los_h))

        def stamp(hours):
            return admit + pd.Timedelta(hours=float(hours))

        # --- structural framing ------------------------------------------
        raw_rows.append((pid, "ADMISSION", stamp(0), stamp(0), "True"))
        abs_rows.append((pid, "ADMISSION_EVENT", stamp(0), stamp(0), "True"))

        # --- raw measurement stream --------------------------------------
        # The series are kept, because the Mediator's complications are placed
        # ON the readings that cross a threshold. That is what the real engine
        # does, and it is what lets the raw-side derivation in
        # `kineret.raw_events` reproduce the same events from the same rows.
        series = {}
        for concept, (mu, sd) in RAW_MEASURES.items():
            if concept not in known:
                continue
            t = float(rng.uniform(0, 6))
            while t < los_h:
                value = float(np.round(rng.normal(mu, sd), 2))
                raw_rows.append((pid, concept, stamp(t), stamp(t), value))
                series.setdefault(concept, []).append((t, value))
                t += float(rng.uniform(3, 10))

        for concept in RAW_ADMIN:
            if concept not in known:
                continue
            for _ in range(int(rng.integers(2, 12))):
                t = float(rng.uniform(0, los_h))
                raw_rows.append((pid, concept, stamp(t), stamp(t), "True"))

        # --- Mediator abstraction intervals -------------------------------
        for concept, values in STATE_VALUES.items():
            if concept not in known:
                continue
            t = float(rng.uniform(0, 8))
            while t < los_h:
                duration = float(rng.uniform(4, 20))
                end = min(t + duration, los_h)
                abs_rows.append((pid, concept, stamp(t), stamp(end),
                                 str(rng.choice(values))))
                trend = concept.replace("_STATE", "_TREND")
                if trend in known:
                    abs_rows.append((pid, trend, stamp(t), stamp(end),
                                     str(rng.choice(TREND_SUFFIX))))
                t = end + float(rng.uniform(0, 6))

        # --- threshold complications (Mediator output only) ------------------
        # Emitted exactly as the Mediator's knowledge base defines them, so
        # `kineret.raw_events` -- which executes those same compiled rules --
        # must reproduce them occurrence for occurrence.
        #
        #   HYPERGLYCEMIA        glucose >= 250  OR  (>= 180 and previous
        #                        reading within 24 h also >= 180)
        #   HYPOGLYCEMIA         glucose <= 54   OR  (<= 70 and previous <= 70)
        #   SEVERE_HYPER/HYPO    glucose >= 250 / <= 54
        #   KIDNEY_COMPLICATION  creatinine/baseline >= 2.0  OR  >= 4.0
        glucose = [(t, v) for t, v in series.get("GLUCOSE_MEASURE", [])
                   if 20.0 <= v <= 1200.0]
        for i, (t, v) in enumerate(glucose):
            prev = glucose[i - 1] if i else None
            sustained_high = (prev is not None and v >= 180
                              and prev[1] >= 180 and (t - prev[0]) <= 24.0)
            sustained_low = (prev is not None and v <= 70
                             and prev[1] <= 70 and (t - prev[0]) <= 24.0)
            if v >= 250 or sustained_high:
                abs_rows.append((pid, "HYPERGLYCEMIA_EVENT", stamp(t), stamp(t), "True"))
            if v <= 54 or sustained_low:
                abs_rows.append((pid, "HYPOGLYCEMIA_EVENT", stamp(t), stamp(t), "True"))
            if v >= 250:
                abs_rows.append((pid, "SEVERE_HYPERGLYCEMIA_EVENT", stamp(t), stamp(t), "True"))
            if v <= 54:
                abs_rows.append((pid, "SEVERE_HYPOGLYCEMIA_EVENT", stamp(t), stamp(t), "True"))

        # Readings outside a raw concept's declared range never reach a rule.
        # The observation code: KIDNEY_COMPLICATION accepts a second
        # ConceptName, KIDNEY_COMPLICATION_OBS, and a row filed under it drives
        # the event on its own. A rule that only looks for the concept's own
        # name misses these entirely.
        if "KIDNEY_COMPLICATION_EVENT" in known and rng.random() < 0.12:
            t = float(rng.uniform(6, los_h))
            raw_rows.append((pid, "KIDNEY_COMPLICATION_OBS", stamp(t), stamp(t), "True"))
            abs_rows.append((pid, "KIDNEY_COMPLICATION_EVENT", stamp(t), stamp(t), "True"))

        creatinine = [(t, v) for t, v in series.get("CREATININE_SERUM_MEASURE", [])
                      if 0.1 <= v <= 20.0]
        if creatinine and "KIDNEY_COMPLICATION_EVENT" in known:
            baseline = creatinine[0][1]
            for i, (t, v) in enumerate(creatinine):
                # The ratio clause reads a parameterized concept whose baseline
                # is the first reading -- which the engine consumes, so row 0
                # cannot fire it. The absolute clause reads the raw concept
                # directly, so row 0 CAN fire that one.
                ratio_hit = i > 0 and baseline > 0 and v / baseline >= 2.0
                if ratio_hit or v >= 4.0:
                    abs_rows.append((pid, "KIDNEY_COMPLICATION_EVENT",
                                     stamp(t), stamp(t), "True"))

        # --- complications -------------------------------------------------
        died_at = None
        for canonical, raw_name in OUTCOME_RAW_SPELLING.items():
            if canonical not in known or rng.random() > OUTCOME_RATE[canonical]:
                continue
            n_episodes = 1 if canonical == "DEATH_EVENT" else int(rng.integers(1, 4))
            for _ in range(n_episodes):
                t = float(rng.uniform(6, los_h))
                if canonical == "DEATH_EVENT":
                    died_at = t
                abs_rows.append((pid, canonical, stamp(t), stamp(t), "True"))
                # The observation-code targets are a pass-through rule: the raw
                # concept IS the evidence, so it must exist in the raw file at
                # the same instant for the derivation to reproduce the event.
                raw_rows.append((pid, raw_name or canonical[:-len("_EVENT")],
                                 stamp(t), stamp(t), "True"))

        # --- non-target KB conclusions (Mediator output only) ---------------
        for concept, rate in KB_ONLY_EVENTS.items():
            if concept not in known or rng.random() > rate:
                continue
            t = float(rng.uniform(6, los_h))
            abs_rows.append((pid, concept, stamp(t), stamp(t), "True"))

        # --- terminus -------------------------------------------------------
        if died_at is not None:
            raw_rows.append((pid, "DEATH", stamp(died_at), stamp(died_at), "True"))
        else:
            raw_rows.append((pid, "RELEASE", stamp(los_h), stamp(los_h), "True"))
            abs_rows.append((pid, "RELEASE_EVENT", stamp(los_h), stamp(los_h), "True"))

        # --- static context --------------------------------------------------
        ctx_rows.append({
            # The Mediator carries one id column, so the temporal tables key on
            # the admission. Context keys on VisitId and names the person, who
            # may have several admissions -- exactly the real drop's shape.
            "VisitId": pid,
            "person_id": person_of[pid],
            "age_at_admission": float(np.round(rng.normal(64, 14), 1)),
            "gender": int(rng.integers(0, 2)),
            "admission_type": int(rng.integers(0, 4)),
            "has_diabetes_type1": int(rng.random() < 0.15),
            "has_diabetes_type2": int(rng.random() < 0.55),
            "has_hypertension": int(rng.random() < 0.45),
            "has_obesity": int(rng.random() < 0.25),
            "has_ckd": int(rng.random() < 0.18),
            "has_chf": int(rng.random() < 0.20),
        })

        # --- QA compliance ---------------------------------------------------
        for pattern in QA_PATTERNS:
            t = float(rng.uniform(0, 12))
            while t < los_h:
                qa_rows.append({
                    "PatientId": pid, "PatternName": pattern,
                    "StartDateTime": stamp(t), "EndDateTime": stamp(t + 1),
                    "ComplianceScore": float(np.round(rng.beta(5, 2), 3)),
                })
                t += float(rng.uniform(12, 36))

    cols = ["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value"]

    # The Mediator OUTPUT keys on the admission under the name `PatientId` --
    # the engine accepts exactly one id column.
    abstract = pd.DataFrame(abs_rows, columns=cols).sort_values(["PatientId", "StartDateTime"])

    # The Mediator INPUT carries both identities plus the window it was cut on:
    #   PatientId  the person
    #   VisitId    the admission  (what everything else keys on)
    # ...which is exactly the mismatch that collapses a cohort when the raw file
    # is joined on PatientId instead of VisitId.
    raw = pd.DataFrame(raw_rows, columns=cols).rename(columns={"PatientId": "VisitId"})
    raw["PatientId"] = raw["VisitId"].map(person_of)
    raw["AdmissionStart"] = raw["VisitId"].map(lambda v: window[v][0])
    raw["AdmissionEnd"] = raw["VisitId"].map(lambda v: window[v][1])
    # Blank on the ADMISSION row, mirroring the real export.
    raw["relevant_admission"] = np.where(raw["ConceptName"] == "ADMISSION",
                                         None, True)
    # A few rows from a NEIGHBOURING admission leak into the export and are
    # flagged False. They are appended, never drawn from the admission's own
    # series: the Mediator ran per visit and never saw them, so flipping a real
    # reading's flag would change the baseline and the previous-reading gate
    # that its rules depend on, and the derived view could not match.
    leaked = raw.sample(max(1, len(raw) // 200), random_state=int(seed)).copy()
    leaked["relevant_admission"] = False
    leaked["Value"] = leaked["Value"]
    raw = pd.concat([raw, leaked], ignore_index=True)
    raw = raw[["PatientId", "VisitId", "ConceptName", "StartDateTime",
               "EndDateTime", "Value", "AdmissionStart", "AdmissionEnd",
               "relevant_admission"]].sort_values(["VisitId", "StartDateTime"])
    return (raw.reset_index(drop=True), abstract.reset_index(drop=True),
            pd.DataFrame(ctx_rows)[["VisitId", "person_id"] + CONTEXT_COLUMNS],
            pd.DataFrame(qa_rows))


def main():
    """Purpose: Write the four synthetic source tables to disk."""
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    raw, abstract, context, qa = generate(args.patients, args.seed,
                                          args.min_days, args.max_days)

    ext = "xlsx" if args.excel else "csv"
    write = ((lambda df, p: df.to_excel(p, index=False)) if args.excel
             else (lambda df, p: df.to_csv(p, index=False)))
    # Canonical stems -- the same names the real drop uses, so a synthetic run
    # and a real one differ only in the directory they point at.
    for name, df in (("mediator_input", raw), ("mediator_output", abstract),
                     ("context_data", context), ("qa_scores", qa)):
        path = os.path.join(args.out, f"{name}.{ext}")
        write(df, path)
        print(f"[synthetic] {path}  rows={len(df):,}")

    print(f"\n[synthetic] {context['VisitId'].nunique()} admissions ({context['person_id'].nunique()} people). Copy them "
          f"into data/source/, or point the pipeline at them in place:\n"
          f"  KINERET_RAW_TEMPORAL={args.out}/mediator_input.{ext} \\\n"
          f"  KINERET_ABSTRACT={args.out}/mediator_output.{ext} \\\n"
          f"  KINERET_CONTEXT={args.out}/context_data.{ext} \\\n"
          f"  KINERET_QA={args.out}/qa_scores.{ext} \\\n"
          f'    python -c "from kineret.benchmark import smoke_test; smoke_test()"')


if __name__ == "__main__":
    main()
