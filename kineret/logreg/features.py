"""
Feature engineering for the non-temporal, non-neural baseline.

STraTS and INTERVenE-Enc both model the trajectory as a sequence. Logistic
regression cannot, so this module collapses the same [0, K*24] h window into a
fixed-width vector of per-variable summary statistics. That is precisely the
comparison the baseline is there to make: how much of the signal survives when
you throw the temporal structure away and keep only "what was measured, how
much of it, how high, how it moved".

Input is the RAW temporal table -- the same source ss-STraTS reads -- with the
same leakage blocklist and the same canonical event support, so the models see
the same substrate.

Training is augmented the same way too: the unit is a **sample**, a (patient, K)
pair, so a train patient contributes one feature row per context window.
"""

import numpy as np
import pandas as pd

from kineret.io_utils import to_numeric_value

# Per-variable aggregations over the observation window.
#
# The point of this baseline is to ask how much signal survives when the
# temporal structure is thrown away, so what is left has to be a fair summary of
# each concept's DISTRIBUTION inside the window, not just its average:
#
#   count                 how often the concept was measured -- measurement
#                         frequency is itself a strong clinical signal
#   mean, std             centre and spread
#   min, max              the extremes, which is usually where the pathology is
#   p25, median, p75      robust shape, unmoved by a single wild reading
#   first, last           where the patient entered and left the window
#   slope                 direction of travel, (last - first) / elapsed hours
#   observed              measured at all, vs never
#
# `slope` and `observed` are computed separately; the rest go straight to pandas.
NUMERIC_AGGS = ["count", "mean", "std", "min", "max", "first", "last"]
QUANTILE_AGGS = {"p25": 0.25, "median": 0.50, "p75": 0.75}

# Variables observed in fewer than this fraction of TRAIN patients are dropped.
# Keeps the design matrix from exploding into thousands of near-empty columns
# that regularisation would only have to shrink back to zero.
MIN_VARIABLE_SUPPORT = 0.02


def _expand_categoricals(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose: Give categorical events a numeric representation.
    Method:  Rows whose Value does not parse as a number become indicator
             variables named `<Concept>_<Value>` carrying value 1.0 -- the same
             expansion the STraTS preprocessor applies, so the two models see
             an identical variable vocabulary.

    Args:
        raw (pd.DataFrame): Windowed rows with sample_id / ConceptName / Value.

    Returns:
        pd.DataFrame: sample_id, variable, value, hours.
    """
    df = raw.copy()
    df["value"] = to_numeric_value(df["Value"])
    categorical = df["value"].isna()
    df.loc[categorical, "ConceptName"] = (
        df.loc[categorical, "ConceptName"].astype(str) + "_"
        + df.loc[categorical, "Value"].astype(str)
    )
    df.loc[categorical, "value"] = 1.0
    df = df.dropna(subset=["value"])
    # `sample_id` is the grouping key -- a (patient, window) pair -- so it has
    # to survive the expansion, not the PatientId it was derived from.
    return df.rename(columns={"ConceptName": "variable"})[
        ["sample_id", "variable", "value", "hours"]]


def build_feature_frame(raw: pd.DataFrame, samples: pd.DataFrame,
                        blocked: set, keep_variables=None,
                        min_support: float = MIN_VARIABLE_SUPPORT):
    """
    Purpose: Collapse each SAMPLE's observation window into one feature row.
    Method:  A sample is a (patient, K) pair, so the window is that sample's own
             [0, K*24] h -- a patient appears once per training window, each time
             summarising strictly less history. Rows are grouped by window so the
             clip is applied once per K rather than once per patient.

             Per (sample, variable): count / mean / std / min / max / p25 /
             median / p75 / first / last / slope / span / rate / observed. The
             result is pivoted wide. `keep_variables` pins the column set to the
             one fitted on TRAIN so val and test cannot introduce new columns.

    Args:
        raw            (pd.DataFrame): Raw rows with an `hours` column.
        samples        (pd.DataFrame): sample_id / PatientId / k / split.
        blocked        (set):          Leakage blocklist.
        keep_variables (list|None):    Fixed variable vocabulary; None fits one.
        min_support    (float):        Support floor when fitting the vocabulary.

    Returns:
        tuple[pd.DataFrame, list]: (sample_id-indexed features, variable list).
    """
    index = pd.Index(samples["sample_id"].to_numpy(), name="sample_id")
    usable = raw[(raw["hours"] >= 0.0) & (~raw["ConceptName"].isin(blocked))]

    blocks = []
    for k, group in samples.groupby("k", sort=True):
        sample_of = dict(zip(group["PatientId"], group["sample_id"]))
        block = usable[(usable["hours"] <= float(k) * 24.0)
                       & (usable["PatientId"].isin(sample_of))].copy()
        block["sample_id"] = block["PatientId"].map(sample_of)
        blocks.append(block)
    window = pd.concat(blocks, ignore_index=True)
    long = _expand_categoricals(window)

    if keep_variables is None:
        support = long.groupby("variable")["sample_id"].nunique() / max(len(index), 1)
        keep_variables = sorted(support[support >= min_support].index)
    keep_set = set(keep_variables)
    long = long[long["variable"].isin(keep_set)]

    if len(long) == 0:
        return pd.DataFrame(index=index), list(keep_variables)

    long = long.sort_values(["sample_id", "variable", "hours"])
    grouped = long.groupby(["sample_id", "variable"])
    stats = grouped["value"].agg(NUMERIC_AGGS)

    # Quantiles: the robust half of the distribution summary. A single wild
    # reading moves mean/min/max a long way and these barely at all, so the two
    # families together tell the model whether an extreme was an excursion or
    # the patient's actual level.
    for name, q in QUANTILE_AGGS.items():
        stats[name] = grouped["value"].quantile(q)

    # Slope over the window: change per hour between first and last reading.
    # Single-observation variables get 0.0 rather than a divide-by-zero.
    span = grouped["hours"].agg(["first", "last"])
    elapsed = (span["last"] - span["first"]).to_numpy()
    stats["slope"] = np.where(elapsed > 0,
                              (stats["last"] - stats["first"]).to_numpy()
                              / np.where(elapsed > 0, elapsed, 1.0), 0.0)
    # How long the concept was watched for, and how densely.
    stats["span_hours"] = elapsed
    stats["rate_per_day"] = stats["count"].to_numpy() / np.maximum(elapsed / 24.0, 1e-6)
    stats["observed"] = 1.0

    wide = stats.unstack("variable")
    # MultiIndex (stat, variable) -> flat "<variable>__<stat>" column names.
    wide.columns = [f"{variable}__{stat}" for stat, variable in wide.columns]
    wide = wide.reindex(index)

    # Count-like columns mean "it never happened", which is genuinely zero.
    # Value columns mean "we do not know this patient's level", which is not
    # zero -- those stay NaN for the median imputer, with a missingness
    # indicator, so the model can tell the two situations apart.
    zero_fill = ("__observed", "__count", "__rate_per_day", "__span_hours")
    for col in wide.columns:
        if col.endswith(zero_fill):
            wide[col] = wide[col].fillna(0.0)
    return wide.astype("float32"), list(keep_variables)


def assemble_design_matrix(features: pd.DataFrame, context: pd.DataFrame,
                           medians: pd.Series = None):
    """
    Purpose: Join trajectory features with static context and impute.
    Method:  Column-concatenate, then fill remaining NaNs with the TRAIN median
             (passed in for val/test so no test statistic leaks into fitting).
             A `__missing` indicator is added for every column that had at least
             one NaN on TRAIN, so "not measured" stays a usable signal rather
             than silently becoming the median patient.

    Args:
        features (pd.DataFrame): Output of `build_feature_frame`.
        context  (pd.DataFrame): PatientId-indexed static block.
        medians  (pd.Series|None): TRAIN medians; fitted here when None.

    Returns:
        tuple[pd.DataFrame, pd.Series]: (design matrix, medians used).
    """
    X = features.join(context.reindex(features.index), how="left")
    if medians is None:
        medians = X.median(numeric_only=True)
        missing_cols = [c for c in X.columns if X[c].isna().any()]
        medians = medians.reindex(X.columns).fillna(0.0)
        medians.attrs["missing_cols"] = missing_cols
    # Build the indicator block in one go. Inserting ~1,500 columns one at a
    # time fragments the frame -- pandas warns about it on every insert, and on
    # a 179k-row table the repeated reallocation dominates the whole arm.
    missing_cols = [c for c in medians.attrs.get("missing_cols", []) if c in X.columns]
    if missing_cols:
        indicators = pd.DataFrame(
            {f"{col}__missing": X[col].isna().astype("float32") for col in missing_cols},
            index=X.index)
        X = pd.concat([X, indicators], axis=1)
    X = X.fillna(medians).fillna(0.0)
    return X.astype("float32"), medians
