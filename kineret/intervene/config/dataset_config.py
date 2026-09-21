"""
INTERVenE-Enc dataset configuration, re-pointed at the Kineret sources.

Ported from `INTERVenE-Enc/intervene_enc/config/dataset_config.py`. The module
keeps the same names the package imports with `import *`, but two things now
come from the shared layer instead of being hardcoded:

* the outcome list is the cohort's post-support-filter list, so INTERVenE's
  prediction head has exactly the columns STraTS and LogReg have;
* the observation window follows this run's K.

`configure(...)` rebinds those module globals for a run. The INTERVenE package
reads them at call time, so a single call before building the tokenizer is
enough -- which is why this is a mutation rather than a constructor argument.
"""

import os

from kineret.config import paths
from kineret.config import data_config as _shared

PROJECT_ROOT = paths.PROJECT_ROOT

# Data file paths. The Mediator OUTPUT (temporal abstractions) is what
# INTERVenE-Enc consumes -- unlike STraTS/LogReg, which read the raw table.
TAK_REPO_PATH            = paths.TAK_REPO_PATH
TRAIN_TEMPORAL_DATA_FILE = paths.ABSTRACT_FILE
TRAIN_CTX_DATA_FILE      = paths.CONTEXT_FILE
TEST_TEMPORAL_DATA_FILE  = paths.ABSTRACT_FILE
TEST_CTX_DATA_FILE       = paths.CONTEXT_FILE
QA_DATA_FILE             = paths.QA_FILE

# Prediction targets. Defaults to the shared list; `configure()` narrows this to
# the cohort's support-filtered set so every model shares a head layout.
OUTCOMES = list(_shared.OUTCOMES)

ADMISSION_TOKEN = _shared.ADMISSION_TOKEN
DEATH_TOKEN     = _shared.DEATH_TOKEN
RELEASE_TOKEN   = _shared.RELEASE_TOKEN

TERMINAL_OUTCOMES = [RELEASE_TOKEN, DEATH_TOKEN]   # <EOT> tokens; one per admission

# Keep ordered -- utils.py indexes this list positionally when building the
# meal-legality lookup table.
MEAL_TOKENS = ["MEAL_CONTEXT_Breakfast", "MEAL_CONTEXT_Lunch",
               "MEAL_CONTEXT_Dinner", "MEAL_CONTEXT_Night-Snack"]

# Rare-outcome demotion inside the tokenizer. Set to 0.0 by `configure()`:
# the shared cohort has already applied `OUTCOME_SUPPORT_THRESHOLD` on the
# train split, and a second, differently-measured filter here would silently
# desynchronise INTERVenE's head from the other two models'.
OUTCOME_RARE_THRESHOLD_PCT = 0.0

USE_QA_DATA = _shared.USE_QA_DATA

# Observation window in hours -- the seed the model seeds on, and the boundary
# past which outcome support is measured. `configure()` sets it to K*24.
OBSERVATION_WINDOW_HOURS = _shared.EVAL_CONTEXT_DAYS * 24

INCLUSION_EXCLUSION_CRITERIA = _shared.temporal_filters(USE_QA_DATA)


def configure(context_days: int, use_qa: bool, outcome_names=None):
    """
    Purpose: Point this config at one cell of the K x QA sweep.
    Method:  Rebind the module globals the INTERVenE package reads -- observation
             window, QA flag, inclusion/exclusion filters, outcome list -- and
             then push the new values into every already-imported
             `kineret.intervene.*` module.

             That push is load-bearing: the ported package pulls this config in
             with `from ...dataset_config import *`, which copies the values
             into each importing module's namespace at import time. Rebinding
             only here would leave `dataset.py` and friends holding the old
             numbers. Re-exporting explicitly is the small price for keeping the
             upstream `import *` style untouched.

             Must be called BEFORE the tokenizer is built:
             `EMRTokenizer.from_processed_df` reads OUTCOMES and
             OBSERVATION_WINDOW_HOURS while assembling the outcome head.

    Args:
        context_days  (int):        K -- context window in days.
        use_qa        (bool):       QA ablation arm.
        outcome_names (list|None):  Cohort outcome list; leaves the default when
                                    None (ad-hoc runs without a cohort).

    Returns:
        dict: The values that were bound, for logging.
    """
    import sys

    global OUTCOMES, USE_QA_DATA, OBSERVATION_WINDOW_HOURS
    global INCLUSION_EXCLUSION_CRITERIA, OUTCOME_RARE_THRESHOLD_PCT

    if outcome_names is not None:
        OUTCOMES = list(outcome_names)
    USE_QA_DATA = bool(use_qa)
    OBSERVATION_WINDOW_HOURS = float(context_days) * 24.0
    INCLUSION_EXCLUSION_CRITERIA = _shared.temporal_filters(USE_QA_DATA)
    OUTCOME_RARE_THRESHOLD_PCT = 0.0

    bound = {
        "OUTCOMES": OUTCOMES,
        "USE_QA_DATA": USE_QA_DATA,
        "OBSERVATION_WINDOW_HOURS": OBSERVATION_WINDOW_HOURS,
        "INCLUSION_EXCLUSION_CRITERIA": INCLUSION_EXCLUSION_CRITERIA,
        "OUTCOME_RARE_THRESHOLD_PCT": OUTCOME_RARE_THRESHOLD_PCT,
    }
    for name, module in list(sys.modules.items()):
        if not name.startswith("kineret.intervene") or module is None:
            continue
        for key, value in bound.items():
            if hasattr(module, key):
                setattr(module, key, value)

    print(f"[intervene/config] configured for K={context_days}, QA={use_qa}: "
          f"{len(OUTCOMES)} outcomes, window={OBSERVATION_WINDOW_HOURS:.0f} h")
    return bound
