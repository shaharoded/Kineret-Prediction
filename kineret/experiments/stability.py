"""Bounded 96-hour sensitivity study using the deployed Mediator implementations.

Direct numeric states/trends only. Does not claim robustness of patterns, contexts,
outcome labels or model predictions. No database writes or model training.
"""
from pathlib import Path
import hashlib
import json
import math
import pickle
import sys

import numpy as np
import pandas as pd

from kineret.config import paths
from kineret.io_utils import normalise_temporal, load_table
from .cohorts import evaluation_window, hospital_series, people, fingerprint
from .workflow import write_json, _stamp
from .io import load_temporal

CONDITIONS = {"noise_1pct": ("noise", .01), "noise_5pct": ("noise", .05),
              "remove_5pct": ("remove", .05), "remove_10pct": ("remove", .10)}


def interval_rows(frame, origin, hours=96):
    if frame.empty:
        return []
    start = np.maximum((pd.to_datetime(frame.StartDateTime)-origin).dt.total_seconds(), 0)
    end = np.minimum((pd.to_datetime(frame.EndDateTime)-origin).dt.total_seconds(), hours*3600)
    result = sorted((float(a), float(b), str(v)) for a, b, v in zip(start, end, frame.Value) if b > a)
    if any(result[i][0] < result[i-1][1]-1e-7 for i in range(1, len(result))):
        raise ValueError("Overlapping intervals within one TAK; cannot score unambiguously.")
    return result


def interval_agreement(a, b):
    baseline = sum(y-x for x, y, _ in a)
    altered = sum(y-x for x, y, _ in b)
    overlap = same = 0.; i = j = 0
    while i < len(a) and j < len(b):
        left, right = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if right > left:
            overlap += right-left
            if a[i][2] == b[j][2]: same += right-left
        if a[i][1] <= b[j][1]: i += 1
        else: j += 1
    return dict(baseline=baseline, altered=altered, overlap=overlap, matching=same,
                union=baseline+altered-overlap)


def repository_scope(repo):
    from core.tak.raw_concept import RawConcept, ParameterizedRawConcept
    specs, coverage = {}, []
    for tak in sorted(repo.taks.values(), key=lambda t: t.name):
        if tak.family != "state": continue
        raw = repo.get(tak.derived_from)
        reason = None
        if isinstance(raw, ParameterizedRawConcept) or not isinstance(raw, RawConcept):
            reason = "derived/multivariate input"
        elif raw.concept_type != "raw-numeric": reason = "not numeric"
        elif any(set(r.constraints) != {0} or r.constraints[0]["type"] != "numeric" for r in tak.abstraction_rules):
            reason = "not a direct scalar threshold rule"
        if reason:
            coverage.append(dict(state=tak.name, status=reason)); continue
        cuts = sorted({float(v[k]) for rule in tak.abstraction_rules for v in rule.constraints[0]["rules"]
                       for k in ("min", "max") if k in v and math.isfinite(v[k])})
        if len(cuts) < 2:
            coverage.append(dict(state=tak.name, status="insufficient finite cutoffs")); continue
        trends = [t for t in repo.taks.values() if t.family == "trend" and t.derived_from == raw.name]
        specs[tak.name] = dict(raw=raw, state=tak, trends=trends, cuts=np.array(cuts), attributes=raw.attributes)
        coverage.append(dict(state=tak.name, status="included", raw=raw.name,
                             trends="|".join(t.name for t in trends), thresholds=cuts))
    return specs, pd.DataFrame(coverage)


def _sample(cohort, raw, visits_path, n, seed):
    with evaluation_window(4):
        frame = cohort.samples("test")[["PatientId"]].copy()
    frame["PersonId"] = people(cohort, frame.PatientId)
    frame = frame.sample(frac=1, random_state=seed).drop_duplicates("PersonId")
    frame["hospital"] = frame.PatientId.map(hospital_series(cohort, visits_path))
    frame = frame.dropna(subset=["hospital"]).copy()
    if frame.empty:
        raise ValueError("No test admissions with unambiguous hospital assignments.")
    frame["count"] = frame.PatientId.map(raw.groupby("PatientId").size()).fillna(0)
    frame["density"] = pd.qcut(frame["count"].rank(method="first"), 4, labels=False, duplicates="drop")
    frame["stratum"] = frame.hospital.astype(str)+":"+frame.density.astype(str)
    groups = list(frame.groupby("stratum", sort=True))
    n = min(n, len(frame))
    if n < len(groups): raise ValueError("Sample must cover every hospital/density stratum.")
    counts = np.ones(len(groups), int)
    sizes = np.array([len(g) for _, g in groups])
    for _ in range(n-len(groups)):
        # Proportional allocation with at least one person per stratum.
        score = sizes/(counts+1)
        score[counts >= sizes] = -1
        counts[np.argmax(score)] += 1
    selected = []
    for (name, g), count in zip(groups, counts):
        s = g.sample(int(count), random_state=seed).copy()
        s["weight"] = len(g)/count
        selected.append(s)
    return pd.concat(selected).reset_index(drop=True)


def run_stability(cohort, directory, mediator_root, repository_pickle, visits_path,
                  n_people=500, repeats=2, n_resamples=2000, seed=2023):
    mediator_root, repository_pickle = Path(mediator_root), Path(repository_pickle)
    if not (mediator_root/"core/tak").is_dir() or not repository_pickle.exists():
        raise FileNotFoundError("Supply the deployed Mediator checkout and Kineret knowledge-base folder or repository pickle.")
    sys.path.insert(0, str(mediator_root.resolve()))
    from core.tak.repository import set_tak_repository
    from core.mediator import Mediator
    if repository_pickle.is_dir():
        from types import SimpleNamespace
        # Repository parsing needs no database connection or patient processing.
        builder = Mediator(repository_pickle, data_access=SimpleNamespace(db_path=None))
        repo = builder.build_repository(silent=True)
        repository_stamp = {
            str(p.relative_to(repository_pickle)): _stamp(p)
            for p in sorted(repository_pickle.rglob("*"))
            if p.is_file() and p.suffix.lower() in (".xml", ".json", ".xsd")
        }
    else:
        with repository_pickle.open("rb") as f:
            repo = pickle.load(f)
        repository_stamp = _stamp(repository_pickle)
    set_tak_repository(repo)
    specs, coverage = repository_scope(repo)
    if not specs: raise ValueError("No eligible direct numeric state/trend definitions.")
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    design = dict(cohort=fingerprint(cohort), repository=repository_stamp, hours=96,
                  n_people=n_people, repeats=repeats, conditions=CONDITIONS, seed=seed,
                  inputs=_stamp(paths.RAW_TEMPORAL_FILE), visits=_stamp(visits_path),
                  state_engine=_stamp(mediator_root/"core/tak/state.py"),
                  trend_engine=_stamp(mediator_root/"core/tak/trend.py"))
    # JSON round trip normalizes condition tuples for resume comparisons.
    design = json.loads(json.dumps(design))
    manifest = directory/"design.json"
    if manifest.exists() and json.loads(manifest.read_text()) != design:
        raise ValueError("Stability design changed; use a new output directory.")
    write_json(manifest, design); coverage.to_csv(directory/"definition_coverage.csv", index=False)
    origins = cohort.patients.set_index("PatientId").admission_time
    with evaluation_window(4):
        eligible = cohort.samples("test").PatientId
    raw = load_temporal(paths.RAW_TEMPORAL_FILE, eligible, origins, max_hours=96)
    selected = _sample(cohort, raw, visits_path, n_people, seed)
    selected.to_csv(directory/"sample.csv", index=False)
    raw = raw[raw.PatientId.isin(selected.PatientId)]
    clipper = Mediator.__new__(Mediator)
    clipper.global_clippers = {"ADMISSION": "START", "RELEASE": "END", "DEATH": "END"}
    all_rows, proximity = [], []
    for ordinal, sample in selected.iterrows():
        cache = directory/f"person_{ordinal:04d}.json"
        if cache.exists():
            payload = json.loads(cache.read_text()); all_rows.extend(payload["rows"]); proximity.extend(payload["proximity"]); continue
        part = raw[raw.PatientId == sample.PatientId].copy()
        origin = origins.loc[sample.PatientId]
        clippers = pd.DataFrame([dict(PatientId=sample.PatientId, ConceptName="ADMISSION",
            StartDateTime=origin, EndDateTime=origin, Value="True", AbstractionType="raw-concept")])
        rows, prox = [], []
        for concept_index, (name, spec) in enumerate(specs.items()):
            inputs = part[part.ConceptName.isin([a["name"] for a in spec["attributes"]])].copy()
            number = pd.to_numeric(inputs.Value, errors="coerce")
            valid = pd.Series(False, index=inputs.index)
            for attr in spec["attributes"]:
                valid |= (inputs.ConceptName == attr["name"]) & number.between(
                    attr.get("min") if attr.get("min") is not None else -np.inf,
                    attr.get("max") if attr.get("max") is not None else np.inf)
            inputs = inputs[valid].copy(); number = number[valid].to_numpy(float)
            if inputs.empty: continue
            inputs["AbstractionType"] = "raw-concept"
            cuts = spec["cuts"]; gaps = np.diff(cuts)
            scales = np.minimum(np.r_[gaps[0], gaps], np.r_[gaps, gaps[-1]])
            nearest = np.abs(number[:, None]-cuts).argmin(axis=1)
            delta = np.abs(number-cuts[nearest]); local = scales[nearest]
            prox.append(dict(PatientId=sample.PatientId, state=name, observations=len(number),
                             near_5pct=int((delta <= .05*local).sum()), exact=int((delta == 0).sum()), weight=sample.weight))
            def abstract(frame):
                numeric = spec["raw"].apply(frame.copy())
                numeric = clipper._apply_global_clippers(numeric, clippers)
                return {tak.name: (tak.family, interval_rows(tak.apply(numeric.copy()), origin))
                        for tak in [spec["state"]]+spec["trends"]}
            base = abstract(inputs)
            if base != abstract(inputs): raise ValueError("Unchanged-input control failed.")
            for condition_index, (condition, (mode, amount)) in enumerate(CONDITIONS.items()):
                for repeat in range(repeats):
                    rng = np.random.default_rng(np.random.SeedSequence([seed, ordinal, concept_index, condition_index, repeat]))
                    altered = inputs.copy()
                    if mode == "noise": altered.Value = number+rng.uniform(-amount, amount, len(number))*local
                    else: altered = altered[rng.random(len(altered)) >= amount]
                    changed = abstract(altered)
                    for concept, (family, baseline) in base.items():
                        rows.append(dict(PatientId=sample.PatientId, PersonId=sample.PersonId, stratum=sample.stratum,
                            weight=sample.weight, concept=concept, family=family, condition=condition, repeat=repeat,
                            **interval_agreement(baseline, changed[concept][1])))
        write_json(cache, dict(rows=rows, proximity=prox))
        all_rows.extend(rows); proximity.extend(prox)
        if ordinal % 20 == 0: print(f"[stability] {ordinal+1}/{len(selected)} people")
    detail = pd.DataFrame(all_rows)
    if detail.empty: raise ValueError("No supported state/trend intervals for this sample.")
    detail.to_csv(directory/"interval_metrics.csv", index=False)
    pd.DataFrame(proximity).to_csv(directory/"threshold_proximity.csv", index=False)
    summary = summarize(detail, selected, n_resamples, seed)
    summary.to_csv(directory/"stability_summary.csv", index=False)
    return summary


def summarize(detail, sample, n_resamples=2000, seed=2023):
    metrics = ["baseline", "altered", "overlap", "matching", "union"]
    # Average perturbation realizations before resampling PEOPLE (not intervals).
    by_person = detail.groupby(["condition", "family", "PatientId"])[metrics].sum()/detail.repeat.nunique()
    rows = []
    for (condition, family), group in by_person.groupby(level=[0, 1]):
        g = sample.merge(group.droplevel([0, 1]), on="PatientId", how="left").fillna({m: 0. for m in metrics})
        values = g[metrics].to_numpy(float)*g.weight.to_numpy()[:, None]
        def measure(v):
            b, a, overlap, same, union = v.sum(axis=0)
            return [same/union if union else np.nan, (b-overlap)/b if b else np.nan,
                    (a-overlap)/b if b else np.nan]
        point = measure(values)
        strata = [np.asarray(ix) for ix in g.groupby("stratum").indices.values()]
        rng = np.random.default_rng(seed)
        draws = np.array([measure(values[np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in strata])])
                          for _ in range(n_resamples)])
        for j, metric in enumerate(("label_duration_agreement", "lost_baseline_coverage", "added_coverage_vs_baseline")):
            finite = draws[:, j][np.isfinite(draws[:, j])]
            ci = np.quantile(finite, [.025, .975]) if len(finite) else [np.nan, np.nan]
            rows.append(dict(condition=condition, family=family, metric=metric, estimate=point[j],
                             lo=ci[0], hi=ci[1], n_people=len(g), valid_resamples=len(finite)))
    return pd.DataFrame(rows)
