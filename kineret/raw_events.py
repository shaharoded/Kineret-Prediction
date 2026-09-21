"""
Reproduce the Mediator's complication events from the raw measurement stream.

`mediator_input.csv` holds measurements; `mediator_output.csv` holds the events
the Mediator derived from them. Both files are then held to one canonical event
table, and the two views are intersected — so any occurrence present in one and
missing from the other is a defect in **this** module, not a disagreement
between two sources. The Mediator is deterministic and its rules are published;
there is no honest reason for a mismatch.

The rules are therefore not written here. They are compiled from the knowledge
base XML by `kineret.mediator_rules` and executed verbatim:

    HYPERGLYCEMIA_EVENT   or( GLUCOSE >= 250 | sustained GLUCOSE >= 180 )
    HYPOGLYCEMIA_EVENT    or( GLUCOSE <= 54  | sustained GLUCOSE <= 70  )
    KIDNEY_COMPLICATION   or( CREATININE/baseline >= 2.0 | CREATININE >= 4.0
                              | KIDNEY_COMPLICATION observation )

*sustained* means the reading AND the immediately preceding reading within the
concept's `good-before` window both satisfy the threshold — the engine's
`id_if_thresh_met` gate. *baseline* is the admission's first reading of that
concept, which the engine consumes and excludes from its own output.

Readings outside a raw concept's declared numeric range are dropped first, as
the engine drops them before any rule sees them.
"""

import json
import os

import numpy as np
import pandas as pd

from kineret.config import data_config as C
from kineret.config import paths

# Transformation kinds, mirroring kineret.mediator_rules.
SUSTAINED = "sustained"
RATIO_TO_BASELINE = "ratio_to_baseline"

_RULES_CACHE = {}


def rules_path() -> str:
    """Purpose: Where the compiled event rules live."""
    override = getattr(C, "EVENT_RULES_PATH", None)
    if override:
        return override
    return os.path.join(os.path.dirname(paths.TAK_REPO_PATH), "event_rules.json")


def load_rules(path: str = None) -> dict:
    """
    Purpose: The compiled knowledge base, cached per path.
    Method:  Reads the JSON produced by `python -m kineret.mediator_rules`. If a
             Mediator checkout is configured (`MEDIATOR_KB_PATH`), the XML is
             compiled directly instead, so the rules can never lag the knowledge
             base that actually ran.

    Args:
        path (str|None): Override the rules file.

    Returns:
        dict: {'clippers': {...}, 'events': {...}}.
    """
    kb_path = getattr(C, "MEDIATOR_KB_PATH", None)
    if kb_path and os.path.isdir(kb_path):
        key = f"kb::{kb_path}"
        if key not in _RULES_CACHE:
            from kineret.mediator_rules import parse_knowledge_base
            _RULES_CACHE[key] = parse_knowledge_base(kb_path)
        return _RULES_CACHE[key]

    path = path or rules_path()
    if path not in _RULES_CACHE:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[raw_events] No compiled event rules at {path}. Generate them "
                f"with:  python -m kineret.mediator_rules <path-to>/core/knowledge-base")
        with open(path, encoding="utf-8") as handle:
            _RULES_CACHE[path] = json.load(handle)
    return _RULES_CACHE[path]


def _normalise(name: str) -> str:
    """Purpose: Collapse punctuation so CARDIO-VASCULAR matches CARDIOVASCULAR."""
    return "".join(ch for ch in str(name).upper() if ch.isalnum())


def resolve_concepts(rules: dict, present) -> dict:
    """
    Purpose: Map each rule's source concept to every way the raw file spells it.
    Method:  A raw concept DECLARES the ConceptNames it accepts, and the
             knowledge base is the authority on that: `KIDNEY_COMPLICATION`
             accepts an observation code filed under `KIDNEY_COMPLICATION_OBS`,
             and `CARDIO-VASCULAR_DISORDER` accepts the ETL's unhyphenated
             spelling. Every declared attribute present in the raw file counts,
             because the engine reads them all into the same concept.

             A punctuation-insensitive match is kept as a last resort for an
             extract that spells something the knowledge base did not
             anticipate. A rule that cannot find its source derives nothing,
             which looks exactly like "this complication never happened" --
             which is why this returns a list and the caller reports what it
             could not resolve.

    Args:
        rules   (dict):     Compiled rules.
        present (iterable): ConceptNames in the raw file.

    Returns:
        dict: knowledge-base concept -> list of raw-file spellings present.
    """
    present = list(present)
    exact = set(present)
    folded = {}
    for name in present:
        folded.setdefault(_normalise(name), name)

    declared = rules.get("attributes", {})
    sources = set()
    for spec in rules.get("events", {}).values():
        for rule in spec["rules"]:
            for clause in rule["clauses"]:
                sources.add(clause["source"])

    resolved = {}
    for source in sources:
        spellings = []
        for candidate in [source] + list(declared.get(source, [])):
            if candidate in exact:
                if candidate not in spellings:
                    spellings.append(candidate)
            elif _normalise(candidate) in folded:
                match = folded[_normalise(candidate)]
                if match not in spellings:
                    spellings.append(match)
        if spellings:
            resolved[source] = spellings
    return resolved


def target_spellings(targets, present) -> set:
    """
    Purpose: Every way the raw file spells a target this module has re-derived.
    Method:  Match on the punctuation-insensitive name with the `_EVENT` suffix
             optional, so `CARDIOVASCULAR_DISORDER` is recognised as the raw
             file's spelling of `CARDIO-VASCULAR_DISORDER_EVENT`. Those rows are
             dropped before the derived ones are spliced in; keeping both would
             double-count a target the ETL had already computed.

    Args:
        targets (iterable): Canonical target names.
        present (iterable): ConceptNames in the raw file.

    Returns:
        set[str]: Spellings to remove.
    """
    wanted = set()
    for target in targets:
        stem = target[:-len("_EVENT")] if target.upper().endswith("_EVENT") else target
        wanted.add(_normalise(stem))
    return {name for name in present
            if _normalise(name) in wanted
            or _normalise(name).removesuffix("EVENT") in wanted}


def _clipped(rows: pd.DataFrame, concept: str, clippers: dict) -> pd.DataFrame:
    """
    Purpose: Drop readings outside the concept's declared numeric range.
    Method:  The engine validates every raw reading against the raw-concept's
             `numeric-allowed-values` before any rule runs, so a 5000 mg/dL
             glucose never reaches HYPERGLYCEMIA. Reproducing that here keeps
             the two views aligned at the source.

    Args:
        rows     (pd.DataFrame): Rows for one concept, with a `_value` column.
        concept  (str):          Concept name.
        clippers (dict):         concept -> [min, max].

    Returns:
        pd.DataFrame: In-range rows.
    """
    bounds = clippers.get(concept)
    if not bounds:
        return rows
    low, high = bounds
    keep = pd.Series(True, index=rows.index)
    if low is not None:
        keep &= rows["_value"] >= low
    if high is not None:
        keep &= rows["_value"] <= high
    return rows[keep]


def _constraint_mask(values: pd.Series, constraints: list) -> pd.Series:
    """
    Purpose: OR together one clause's `<allowed-value>` constraints.
    Method:  Mirrors the engine: `equal` compares as text, `min`/`max` are
             inclusive, `range` is inclusive on both ends.

    Args:
        values      (pd.Series):  The values under test.
        constraints (list[dict]): Constraint dicts.

    Returns:
        pd.Series: Boolean mask.
    """
    mask = pd.Series(False, index=values.index)
    numeric = pd.to_numeric(values, errors="coerce")
    for constraint in constraints:
        kind = constraint["type"]
        if kind == "equal":
            target = str(constraint["value"]).strip().lower()
            mask |= values.astype(str).str.strip().str.lower() == target
        elif kind == "min":
            mask |= numeric >= constraint["value"]
        elif kind == "max":
            mask |= numeric <= constraint["value"]
        elif kind == "range":
            mask |= (numeric >= constraint["min"]) & (numeric <= constraint["max"])
    return mask.fillna(False)


def _clause_hits(raw: pd.DataFrame, clause: dict, clippers: dict,
                 spellings=None) -> pd.DataFrame:
    """
    Purpose: The rows satisfying one clause of one event rule.
    Method:  Take the clause's source concept, clip it to its declared range,
             apply the clause's transformation, then test the constraints.

             `sustained` compares each reading against the immediately preceding
             one for the same admission, provided it falls inside the window —
             the engine resolves that parameter as "closest before", which for a
             per-admission time-ordered series is the previous row.

             `ratio_to_baseline` divides by the admission's first reading and
             drops that first row, exactly as the engine's baseline branch does.

    Args:
        raw      (pd.DataFrame): Raw temporal table.
        clause   (dict):         Compiled clause.
        clippers (dict):         Concept validity ranges.

    Returns:
        pd.DataFrame: Qualifying rows.
    """
    concept = clause["source"]
    names = spellings or [concept]
    if isinstance(names, str):
        names = [names]
    rows = raw[raw["ConceptName"].isin(names)]
    if rows.empty:
        return rows.iloc[0:0]

    rows = rows.copy()
    rows["_value"] = pd.to_numeric(rows["Value"], errors="coerce")
    transform = clause.get("transform")

    # A boolean observation concept carries text, not numbers; keep it as is.
    is_boolean = all(c["type"] == "equal" for c in clause["constraints"])
    if not is_boolean:
        rows = rows[rows["_value"].notna()]
        rows = _clipped(rows, concept, clippers)
        if rows.empty:
            return rows.iloc[0:0]

    if transform is None:
        tested = rows["Value"] if is_boolean else rows["_value"]
        return rows[_constraint_mask(tested, clause["constraints"])]

    rows = rows.sort_values(["PatientId", "StartDateTime"])
    grouped = rows.groupby("PatientId", sort=False)

    if transform["kind"] == SUSTAINED:
        previous = grouped["_value"].shift(1)
        previous_time = grouped["StartDateTime"].shift(1)
        gate_value, gate_op = transform["gate_value"], transform["gate_op"]
        gate = (previous >= gate_value) if gate_op == "ge" else (previous <= gate_value)
        window = transform.get("window")
        if window:
            gate &= previous_time >= rows["StartDateTime"] - pd.Timedelta(window)
        eligible = rows[gate.fillna(False)]
        return eligible[_constraint_mask(eligible["_value"], clause["constraints"])]

    if transform["kind"] == RATIO_TO_BASELINE:
        baseline = grouped["_value"].transform("first")
        position = grouped.cumcount()
        # The engine consumes the first reading as the baseline and removes it
        # from its own output, so it can never fire the rule itself.
        ratio = rows["_value"] / baseline.where(baseline > 0)
        eligible = rows[(position > 0) & ratio.notna()]
        return eligible[_constraint_mask(ratio.loc[eligible.index],
                                         clause["constraints"])]

    raise ValueError(f"[raw_events] Unsupported transform: {transform['kind']}")


def derive_raw_events(raw: pd.DataFrame, targets=None, rules: dict = None,
                      verbose: bool = True) -> pd.DataFrame:
    """
    Purpose: Rebuild the raw file's own view of every target complication.
    Method:  Execute the compiled rules. An event fires at the timestamp of the
             reading that satisfied any clause of any rule, which is what the
             engine emits — it preserves the source row's StartDateTime — so the
             two views align to the instant.

             Every target is derived, including the ones whose rule is a plain
             observation pass-through. Deriving those too costs nothing and
             means one code path produces the whole raw-side view, instead of
             some events coming from a rule and others from a name match.

    Args:
        raw     (pd.DataFrame): Raw temporal table, canonical columns.
        targets (list|None):    Event names to derive. Defaults to `OUTCOMES`.
        rules   (dict|None):    Compiled rules. Defaults to the shipped file.
        verbose (bool):         Print what each rule produced.

    Returns:
        pd.DataFrame: Derived event rows, same columns as `raw`.
    """
    say = print if verbose else (lambda *a, **kw: None)
    rules = rules or load_rules()
    targets = list(targets or C.OUTCOMES)
    clippers = rules.get("clippers", {})

    spellings = resolve_concepts(rules, raw["ConceptName"].unique())
    derived, missing = [], []
    for target in targets:
        spec = rules["events"].get(target)
        if spec is None:
            missing.append(target)
            continue

        hits, sources, unsupported, absent = [], [], [], set()
        for rule in spec["rules"]:
            unsupported.extend(rule.get("unsupported", []))
            for clause in rule["clauses"]:
                names = spellings.get(clause["source"])
                if not names:
                    absent.add(clause["source"])
                    continue
                found = _clause_hits(raw, clause, clippers, names)
                if len(found):
                    hits.append(found[["PatientId", "StartDateTime"]])
                    kind = (clause["transform"] or {}).get("kind", "value")
                    label = "+".join(names)
                    sources.append(f"{label}[{kind}]={len(found):,}")
        if unsupported:
            raise ValueError(
                f"[raw_events] {target}: the knowledge base uses clause(s) "
                f"{unsupported} that kineret.mediator_rules cannot compile. "
                f"Add support there rather than deriving a partial rule.")

        if not hits:
            reason = (f"source concept(s) {sorted(absent)} not in the raw file"
                      if absent else "no reading satisfied the rule")
            say(f"[derive] {target:<34} 0 events -- {reason}.")
            continue
        if absent:
            say(f"[derive] {target:<34} NOTE: {sorted(absent)} absent from the "
                f"raw file; that clause cannot contribute.")
        found = pd.concat(hits, ignore_index=True).drop_duplicates()
        derived.append(pd.DataFrame({
            "PatientId": found["PatientId"].to_numpy(),
            "ConceptName": target,
            "StartDateTime": found["StartDateTime"].to_numpy(),
            "EndDateTime": found["StartDateTime"].to_numpy(),
            "Value": "True",
        }))
        say(f"[derive] {target:<34} {len(found):>9,} events  "
            f"({found['PatientId'].nunique():,} admissions)  "
            + " | ".join(sources))

    if missing:
        say(f"[derive] No compiled rule for: {missing}. These targets keep "
            f"whatever the raw file already spells for them.")

    if not derived:
        return raw.iloc[0:0].copy()
    out = pd.concat(derived, ignore_index=True)
    # Carry the source table's dtypes onto the columns a derived row has no
    # value for (the admission-framing block). An all-NA object column makes
    # pandas infer the concat result's dtype from the wrong side, which it now
    # warns about and will one day change.
    for column in raw.columns:
        if column not in out.columns:
            out[column] = pd.Series(index=out.index, dtype=raw[column].dtype)
    return out[raw.columns]


def validate_rules(n_admissions: int = 4000, seed: int = 0,
                   verbose: bool = True) -> pd.DataFrame:
    """
    Purpose: Check the compiled rules reproduce the Mediator, before anything
             expensive runs.
    Method:  Read the two source tables, take a random sample of admissions
             present in BOTH, derive the raw-side events for that sample and
             compare them occurrence by occurrence against the Mediator's.

             This is deliberately a separate, early call. Discovering a wrong
             threshold after a full cohort build has already spent ten minutes
             on 23 million rows -- or, worse, discovering it as a quietly
             shrunken label set -- is the failure mode it exists to prevent.

             Sampling admissions rather than rows keeps every patient's series
             intact, which the `sustained` and `ratio_to_baseline` rules need:
             both look at a reading's neighbours.

    Args:
        n_admissions (int):  How many admissions to check. 0 checks all.
        seed         (int):  Sampling seed.
        verbose      (bool): Print the per-target table.

    Returns:
        pd.DataFrame: The agreement report, worst first.
    """
    from kineret.io_utils import load_table, normalise_temporal

    say = print if verbose else (lambda *a, **kw: None)
    say(f"[validate] reading source tables...")
    raw = normalise_temporal(load_table(paths.RAW_TEMPORAL_FILE))
    abstract = normalise_temporal(load_table(paths.ABSTRACT_FILE))

    # The same relevance filter the cohort applies, for the same reason:
    # a row belonging to a neighbouring admission was never in the series the
    # Mediator ran on, and leaving it in invents "previous reading" pairs that
    # the sustained gate then fires on. Validating a different input than the
    # pipeline uses would defeat the point of validating at all.
    if getattr(C, "DROP_IRRELEVANT_ADMISSION_ROWS", False):
        from kineret.cohort import _drop_irrelevant_rows
        raw = _drop_irrelevant_rows(raw, "raw temporal", verbose=verbose)
        abstract = _drop_irrelevant_rows(abstract, "mediator output", verbose=verbose)

    shared = np.intersect1d(raw["PatientId"].unique(), abstract["PatientId"].unique())
    if len(shared) == 0:
        raise RuntimeError(
            "[validate] The two source tables share no admissions. They are "
            "keyed on different things -- see normalise_temporal's VisitId "
            "re-keying and the Input contract section of the README.")
    if n_admissions and len(shared) > n_admissions:
        shared = np.random.default_rng(seed).choice(shared, n_admissions,
                                                    replace=False)
    raw = raw[raw["PatientId"].isin(set(shared))]
    abstract = abstract[abstract["PatientId"].isin(set(shared))]
    say(f"[validate] {len(shared):,} admissions sampled "
        f"({len(raw):,} raw rows / {len(abstract):,} mediator rows).")

    targets = list(C.OUTCOMES)
    derived = derive_raw_events(raw, targets=targets, verbose=verbose)

    # Anchor both views on the same per-admission origin so `hours` is
    # comparable; the absolute origin does not matter, only that it matches.
    origin = pd.concat([raw[["PatientId", "StartDateTime"]],
                        abstract[["PatientId", "StartDateTime"]]]) \
        .groupby("PatientId")["StartDateTime"].min()

    def _events(table, source):
        hit = table[table["ConceptName"].isin(targets)]
        return pd.DataFrame({
            "PatientId": hit["PatientId"].to_numpy(),
            "outcome": hit["ConceptName"].to_numpy(),
            "hours": ((hit["StartDateTime"] - hit["PatientId"].map(origin))
                      .dt.total_seconds() / 3600.0).to_numpy(),
            "source": source,
        })

    report = agreement_report(_events(derived, "raw"),
                              _events(abstract, "mediator"), targets)

    if verbose:
        print()
        print(f"    {'outcome':<34} {'raw':>9} {'mediator':>9} {'matched':>9} "
              f"{'raw_only':>9} {'med_only':>9}  agree")
        for _, row in report.iterrows():
            flag = "" if row["agreement"] >= 1.0 else "   <-- rule mismatch"
            print(f"    {row['outcome']:<34} {row['raw']:>9,} {row['mediator']:>9,} "
                  f"{row['matched']:>9,} {row['raw_only']:>9,} "
                  f"{row['med_only']:>9,}  {row['agreement']:6.2%}{flag}")
        perfect = int((report["agreement"] >= 1.0).sum())
        print(f"\n[validate] {perfect}/{len(report)} target(s) reproduce the "
              f"Mediator exactly.")
        if perfect < len(report):
            print("[validate] A mismatch is a defect in the rule execution, not "
                  "a disagreement between sources -- the Mediator built "
                  "mediator_output.csv from mediator_input.csv with these rules.")
    return report


def agreement_report(raw_events: pd.DataFrame, abstract_events: pd.DataFrame,
                     targets=None) -> pd.DataFrame:
    """
    Purpose: Per target, how exactly the derived view reproduces the Mediator's.
    Method:  Compare the two (PatientId, hours) sets directly. `matched` is the
             intersection, `raw_only` are events this module invented, and
             `med_only` are events it failed to reproduce. Both should be zero:
             the Mediator is deterministic and these are its own rules.

    Args:
        raw_events      (pd.DataFrame): ['PatientId','outcome','hours', ...].
        abstract_events (pd.DataFrame): The Mediator's view, same shape.
        targets         (list|None):    Restrict/order the rows.

    Returns:
        pd.DataFrame: One row per target, worst agreement first.
    """
    names = targets or sorted(set(raw_events["outcome"]) | set(abstract_events["outcome"]))
    rows = []
    for name in names:
        r = raw_events[raw_events["outcome"] == name]
        a = abstract_events[abstract_events["outcome"] == name]
        r_keys = set(zip(r["PatientId"], r["hours"].round(4)))
        a_keys = set(zip(a["PatientId"], a["hours"].round(4)))
        matched = len(r_keys & a_keys)
        union = len(r_keys | a_keys)
        rows.append({
            "outcome": name,
            "raw": len(r_keys), "mediator": len(a_keys), "matched": matched,
            "raw_only": len(r_keys - a_keys), "med_only": len(a_keys - r_keys),
            "agreement": (matched / union) if union else 1.0,
        })
    return pd.DataFrame(rows).sort_values("agreement").reset_index(drop=True)
