import os
import re
import random
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Sampler
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump
import json
import numpy as np
from collections import Counter
from typing import List

# ───────── local code ─────────────────────────────────────────────────── #
from kineret.intervene.config.dataset_config import *
from kineret.intervene.config.model_config import CHECKPOINT_PATH


class DataProcessor:
    """
    Handles the dataprocess needed to build the tokenizer / train / val / test.
    use max_input_days to trim a test dataset before using it for prediction.

    Expected columns for temporal_df: ['PatientId', 'ConceptName', 'StartDateTime', 'EndDateTime', 'Value']
    Expected columns for context_df: ['PatientId'] + context columns.

    Attributes:
    df (pd.DataFrame): Transformed long-format event dataframe after all processing.
    context_df (pd.DataFrame): Patient context dataframe with PatientId as index.
    scaler (StandardScaler): Scaler fitted to context_df and optionally saved to disk.
    checkpoint_path (str): Path to save the scaler / tokenizer at for later usage.

    """
    def __init__(self, df, context_df, 
                 tak_repo_path=None, 
                 max_input_days=None, 
                 scaler=None, 
                 checkpoint_path=CHECKPOINT_PATH,
                 inclusion_exclusion_criteria=None,
                 enforce_min_duration=True):
        # Load TAK repository from JSON. Defaults to the copy that ships inside
        # the package; the sigma-bin arm passes its augmented copy instead.
        if tak_repo_path is None:
            from kineret.config.paths import TAK_REPO_PATH
            tak_repo_path = TAK_REPO_PATH
        if not str(tak_repo_path).endswith('.json'):
            raise ValueError(f"TAK repo must be a JSON file, got: {tak_repo_path}")
        
        with open(tak_repo_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        self.repo = json_data.get('taks')
        if self.repo is None:
            raise ValueError("JSON file missing 'taks' key.")
        if not isinstance(self.repo, dict):
            raise ValueError("'taks' must be a dictionary.")
        
        # Handle compatability issue in complete flow
        if "StartTime" in df.columns and "EndTime" in df.columns:
            df.rename(columns={"StartTime": "StartDateTime", "EndTime": "EndDateTime"}, inplace=True)
        df['StartDateTime'] = pd.to_datetime(df['StartDateTime'], format='ISO8601', utc=True, errors='raise')
        df['StartDateTime'] = df['StartDateTime'].dt.tz_convert(None)
        df['EndDateTime'] = pd.to_datetime(df['EndDateTime'], format='ISO8601', utc=True, errors='raise')
        df['EndDateTime'] = df['EndDateTime'].dt.tz_convert(None)

        self.df = df.copy()
        self.context_df = context_df.copy()
        self.max_input_days = max_input_days
        # See `_cut_after_k_days`: when False, patients whose trajectory does not
        # reach the cut are kept rather than dropped, because cohort membership
        # was already settled upstream.
        self.enforce_min_duration = enforce_min_duration
        self.scaler = scaler
        self.checkpoint_path = checkpoint_path
        self.inclusion_exclusion_criteria = (
            INCLUSION_EXCLUSION_CRITERIA
            if inclusion_exclusion_criteria is None
            else inclusion_exclusion_criteria
        )


    def run(self):
        self._apply_inclusion_exclusion_criteria()

        # Process on temporal_df
        self._validate_and_align_inputs()
        self._fix_back_to_back_intervals()
        self._truncate_after_terminal_event()
        self._normalize_time()
        self._expand_tokens() # Expands intervals into START/END tokens
        self._insert_null_tokens(gap_hrs=3)
        self._add_parent_raw_concepts()
        if self.max_input_days:
            self._cut_after_k_days()
        
        # Process on context_df
        if 'index' in self.context_df.columns:
            self.context_df = self.context_df.drop(columns=['index'])
        # QA features must be merged BEFORE indexing/scaling so the scaler picks them up.
        self._add_qa_features()
        self.context_df = self.context_df.set_index("PatientId").drop(columns=["PatientId"], errors="ignore").astype("float32")
        self._fit_scaler()
        return self.df, self.context_df

    def _apply_inclusion_exclusion_criteria(self):
        """
        Apply SQL-like inclusion/exclusion filters from INCLUSION_EXCLUSION_CRITERIA.

        Supported condition format:
            WHERE <column> LIKE '<pattern>'
            WHERE <column> NOT LIKE '<pattern>'

        Notes:
            - `WHERE` is optional.
            - `%` is treated as wildcard (0+ chars).
            - `_` is treated as a literal underscore.
        """
        criteria = self.inclusion_exclusion_criteria or {}
        if not isinstance(criteria, dict):
            raise ValueError("inclusion_exclusion_criteria must be a dict with 'temporal'/'context' keys")

        temporal_criteria = criteria.get("temporal", [])
        context_criteria = criteria.get("context", [])

        self.df = self._apply_sql_like_filters(self.df, temporal_criteria, source_name="temporal")
        self.context_df = self._apply_sql_like_filters(self.context_df, context_criteria, source_name="context")

    @staticmethod
    def _apply_sql_like_filters(df: pd.DataFrame, conditions: List[str], source_name: str) -> pd.DataFrame:
        """Apply a list of SQL-like WHERE ... LIKE/NOT LIKE filters to a dataframe."""
        if not conditions:
            return df

        if not isinstance(conditions, list):
            raise ValueError(f"Criteria for '{source_name}' must be a list of condition strings")

        out_df = df.copy()
        pattern = re.compile(
            r"^\s*(?:WHERE\s+)?(?P<column>[A-Za-z_]\w*)\s+(?P<op>NOT\s+LIKE|LIKE)\s+['\"](?P<like_pattern>[^'\"]+)['\"]\s*$",
            flags=re.IGNORECASE,
        )

        for cond in conditions:
            if not cond or not str(cond).strip():
                continue

            match = pattern.match(str(cond))
            if not match:
                raise ValueError(
                    f"Invalid condition format for '{source_name}': {cond}. "
                    "Expected: WHERE <column> LIKE '<pattern>' or WHERE <column> NOT LIKE '<pattern>'"
                )

            column = match.group("column")
            op = match.group("op").upper().replace("  ", " ")
            like_pattern = match.group("like_pattern")

            if column not in out_df.columns:
                raise ValueError(
                    f"Condition references unknown column '{column}' in '{source_name}' dataframe"
                )

            regex = "^" + re.escape(like_pattern).replace("%", ".*") + "$"
            col_as_str = out_df[column].astype(str)
            matches = col_as_str.str.match(regex, na=False)

            before_n = len(out_df)
            if op == "LIKE":
                out_df = out_df[matches].copy()
            elif op == "NOT LIKE":
                out_df = out_df[~matches].copy()
            else:
                raise ValueError(f"Unsupported operator '{op}' in condition: {cond}")

            after_n = len(out_df)
            print(f"[DataProcessor] Applied {source_name} filter: {cond} | rows {before_n} -> {after_n}")

        return out_df

    def _add_qa_features(self):
        """
        Purpose: Augment context_df with per-patient, per-PatternName mean ComplianceScore
                 over the first `qa_history_hours` after admission.
        Method:  Load qa_scores (path from config); compute elapsed hours from each
                 patient's earliest StartDateTime in self.df; filter QA rows to
                 [0, qa_history_hours]; group by (PatientId, PatternName) and mean
                 the score; pivot to wide; reindex to the canonical PatternName set
                 read from the FULL qa file (stable schema across train/val); reindex
                 to all PatientIds in context_df with zero-fill; merge as new columns.

                 No-op when USE_QA_DATA is False so the gate is a single config flag.

        Args:    None (uses self.df, self.context_df, self.max_input_days).

        Returns: None. Mutates self.context_df in place.
        """
        if not USE_QA_DATA:
            return

        if not os.path.exists(QA_DATA_FILE):
            raise FileNotFoundError(
                f"USE_QA_DATA=True but QA file missing: {QA_DATA_FILE}"
            )

        qa = pd.read_csv(QA_DATA_FILE, low_memory=False)
        required = ["PatientId", "PatternName", "StartDateTime", "ComplianceScore"]
        for col in required:
            if col not in qa.columns:
                raise ValueError(f"qa_scores missing required column: {col}")

        qa["StartDateTime"] = pd.to_datetime(
            qa["StartDateTime"], format="ISO8601", utc=True, errors="raise"
        ).dt.tz_convert(None)

        # Canonical pattern column set comes from the FULL qa file so train and val
        # splits always produce the same column layout. Sorted for determinism.
        canonical_patterns = sorted(qa["PatternName"].unique())
        qa_cols = [f"QA_{p}" for p in canonical_patterns]

        # Per-patient admission time = earliest StartDateTime cached in _normalize_time
        # (self.df no longer has StartDateTime after _expand_tokens replaced it with TimePoint).
        # Patients that were dropped (e.g. by _cut_after_k_days) still appear here, which
        # is fine — the reindex/zero-fill at the end aligns to context_df's patient set.
        if not hasattr(self, "_patient_admission_time"):
            raise RuntimeError(
                "_add_qa_features called before _normalize_time — admission timestamps unavailable."
            )
        admission = self._patient_admission_time.rename("AdmissionTime").reset_index()

        qa_history_hours = (
            float(self.max_input_days) * 24.0 if self.max_input_days else float(OBSERVATION_WINDOW_HOURS)
        )

        qa = qa.merge(admission, on="PatientId", how="inner")
        qa["elapsed_h"] = (qa["StartDateTime"] - qa["AdmissionTime"]).dt.total_seconds() / 3600.0
        qa = qa[(qa["elapsed_h"] >= 0.0) & (qa["elapsed_h"] <= qa_history_hours)]

        if len(qa) == 0:
            # No QA rows fell in the window for any patient — still emit zero-filled
            # columns so the schema is stable across runs.
            wide = pd.DataFrame(
                0.0,
                index=self.context_df["PatientId"].unique(),
                columns=qa_cols,
            )
        else:
            agg = qa.groupby(["PatientId", "PatternName"])["ComplianceScore"].mean().reset_index()
            wide = agg.pivot(index="PatientId", columns="PatternName", values="ComplianceScore")
            wide = wide.reindex(columns=canonical_patterns, fill_value=0.0)
            wide.columns = qa_cols

        all_pids = self.context_df["PatientId"].unique()
        wide = wide.reindex(all_pids, fill_value=0.0)
        wide.index.name = "PatientId"
        wide = wide.reset_index()

        # Drop any pre-existing QA columns so re-running DataProcessor stays idempotent.
        stale = [c for c in qa_cols if c in self.context_df.columns]
        if stale:
            self.context_df = self.context_df.drop(columns=stale)

        self.context_df = self.context_df.merge(wide, on="PatientId", how="left")
        for c in qa_cols:
            self.context_df[c] = self.context_df[c].fillna(0.0)

        print(
            f"[DataProcessor] QA features added: {len(qa_cols)} columns, "
            f"window=[0, {qa_history_hours:.1f}h], {len(qa)} qa rows in window."
        )


    def _fit_scaler(self):
        """
        Fit and / or use a standard scaler on the context dataframe. 
        Will save the scaler in the checkpoints (and load from there).
        """        
        if self.scaler is None:
            scaler = StandardScaler()
            self.context_df.loc[:, :] = scaler.fit_transform(self.context_df.values)
            os.makedirs(self.checkpoint_path, exist_ok=True)
            dump(scaler, os.path.join(self.checkpoint_path, 'scaler.pkl'))
        else:
            self.context_df.loc[:, :] = self.scaler.transform(self.context_df.values)

    def _validate_and_align_inputs(self):
        """
        Validates required columns, datetime types, and aligns PatientIds between
        temporal (df) and context (patient_context_df) data. Will also sort the temporal data.

        Returns:
            Tuple of (cleaned_df, cleaned_patient_context_df)
        """
        # 1. Required columns check
        required_columns = ['PatientId', 'ConceptName', 'StartDateTime', 'EndDateTime', 'Value']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column in temporal data: {col}")
        if 'PatientId' not in self.context_df.columns:
            raise ValueError("Missing 'PatientId' column in context data")

        # 2. Check datetime dtypes
        if not pd.api.types.is_datetime64_any_dtype(self.df['StartDateTime']):
            raise TypeError("StartDateTime column must be of datetime64[ns] dtype.")
        if not pd.api.types.is_datetime64_any_dtype(self.df['EndDateTime']):
            raise TypeError("EndDateTime column must be of datetime64[ns] dtype.")

        # 3. Handle duplicate PatientIds in context
        dupe_counts = self.context_df['PatientId'].value_counts()
        duplicates = dupe_counts[dupe_counts > 1]
        if not duplicates.empty:
            print(f"Found {len(duplicates)} PatientIds with duplicate rows in context_df. Aggregating by max value...")
            self.context_df = self.context_df.groupby('PatientId').max().reset_index()

        # 4. Align temporal and context data
        temporal_ids = set(self.df['PatientId'])
        context_ids = set(self.context_df['PatientId'])

        missing_ids = temporal_ids - context_ids
        extra_ids = context_ids - temporal_ids

        if missing_ids:
            print(f"Adding {len(missing_ids)} missing PatientIds to context_df with placeholder values (-1).")
            placeholder_df = pd.DataFrame({
                'PatientId': list(missing_ids),
                **{
                    col: [-1] * len(missing_ids)
                    for col in self.context_df.columns
                    if col != 'PatientId'
                }
            })
            self.context_df = pd.concat([self.context_df, placeholder_df], ignore_index=True)

        if extra_ids:
            print(f"Dropping {len(extra_ids)} unmatched PatientIds from context_df.")
            self.context_df = self.context_df[self.context_df['PatientId'].isin(temporal_ids)].copy()

        # 5. Final integrity checks
        assert self.context_df['PatientId'].is_unique, "PatientId must be unique in context_df after alignment"
        assert set(self.df['PatientId']) == set(self.context_df['PatientId']), "Mismatched PatientIds after alignment"
    

    def _fix_back_to_back_intervals(self, epsilon=pd.Timedelta(seconds=1)):
        """
        If an interval starts at exactly the same timestamp another one ends
        (same patient), shift the *start* forward by `epsilon` to preserve
        START/END ordering for tokenisation.
        """
        df = self.df.sort_values(['PatientId', 'StartDateTime']).reset_index(drop=True).copy()

        same_time = (
            (df['StartDateTime']
            == df.groupby('PatientId')['EndDateTime'].shift(1))
        )

        # shift only the conflicted rows
        df.loc[same_time, 'StartDateTime'] += epsilon
        # Also shift EndDateTime if duration would otherwise be negative/zero
        need_fix = df['EndDateTime'] <= df['StartDateTime']
        df.loc[need_fix, 'EndDateTime'] = df.loc[need_fix, 'StartDateTime'] + epsilon

        self.df = df


    def _truncate_after_terminal_event(self):
        """
        For each patient:
        - If a DEATH_TOKEN occurs within 30 days after a RELEASE_TOKEN:
            → Drop the DEATH_TOKEN.
            → Replace the RELEASE_TOKEN's ConceptName with DEATH_TOKEN (keep its time).
        - Then truncate any records after the first terminal event (RELEASE/DEATH).
        """
        def process_group(group):
            group = group.sort_values("StartDateTime").copy()

            # Handle RELEASE vs DEATH conflicts
            release_rows = group[group["ConceptName"] == RELEASE_TOKEN]
            death_rows = group[group["ConceptName"] == DEATH_TOKEN]

            if not release_rows.empty and not death_rows.empty:
                release_time = release_rows.iloc[0]["StartDateTime"]
                death_time = death_rows.iloc[0]["StartDateTime"]

                # If death is within 30 days after release → drop death, rename release
                if pd.Timedelta(0) <= (death_time - release_time) <= pd.Timedelta(days=30):
                    # Drop death row
                    group = group[group["ConceptName"] != DEATH_TOKEN]

                    # Replace concept name of release row
                    release_index = release_rows.index[0]
                    group.loc[release_index, "ConceptName"] = DEATH_TOKEN

            # Then truncate after first terminal event
            terminal_idx = group[group["ConceptName"].isin(TERMINAL_OUTCOMES)].index
            if not terminal_idx.empty:
                first_terminal_time = group.loc[terminal_idx[0], "StartDateTime"]
                group = group[group["StartDateTime"] <= first_terminal_time]

            return group

        self.df = (
            self.df.groupby("PatientId", group_keys=False)[self.df.columns]
                .apply(process_group)
                .reset_index(drop=True)
        )
    

    def _normalize_time(self):
        """
        Normalizes time to be relative to the start of each visit. Also adds VisitID and VisitStart columns.
        Caches the per-patient earliest StartDateTime on the instance so downstream
        steps (e.g. _add_qa_features) can recover absolute admission timestamps
        after _expand_tokens drops the raw datetime columns.
        """
        df = self.df.copy()
        df["IsAdmission"] = df["ConceptName"] == ADMISSION_TOKEN
        df["VisitCounter"] = df.groupby("PatientId")["IsAdmission"].cumsum()
        df["VisitID"] = df["PatientId"].astype(str) + "_" + df["VisitCounter"].astype(str)
        df["VisitStart"] = df.groupby("VisitID")["StartDateTime"].transform('min')
        df["RelStartTime"] = (df["StartDateTime"] - df["VisitStart"]).dt.total_seconds() / 3600.0 # In hours
        df["RelEndTime"] = (df["EndDateTime"] - df["VisitStart"]).dt.total_seconds() / 3600.0 # In hours
        self._patient_admission_time = df.groupby("PatientId")["StartDateTime"].min()
        self.df = df


    def _add_parent_raw_concepts(self):
        """
        Adds a 'ParentRawConcepts' column to self.df, which contains a list of
        top-level raw concepts for each ConceptName based on the TAKRepository.
        """
        def __deps_from_derived_from(name: str, tak_dict: dict):
            """
            Get the list of TAK names that the given TAK is derived from.
            """
            if 'derived_from' not in tak_dict:
                raise ValueError(f"TAK '{name}' has no 'derived_from' field.")

            df = tak_dict['derived_from']

            if df is None:
                return []

            if isinstance(df, str):
                return [df]

            if isinstance(df, list):
                out = []
                for item in df:
                    if isinstance(item, dict) and "name" in item:
                        out.append(item["name"])
                    elif isinstance(item, str):
                        out.append(item)
                    else:
                        raise ValueError(
                            f"Invalid derived_from entry in TAK '{name}': {item}"
                        )
                return out

            raise ValueError(
                f"Invalid derived_from type for TAK '{name}': {type(df)}"
            )

        def __resolve_top_raw(name: str, seen=None) -> List[str]:
            """
            Recursively resolves the top-level raw concept(s) for a given TAK name.
            Returns a list of raw concept names (can be multiple for patterns/events).
            Raises an error if a cycle is detected or if the TAK is not found.
            """
            if seen is None:
                seen = set()

            if name in seen:
                raise ValueError(f"Cycle detected while resolving raw parent for '{name}'.")

            seen.add(name)

            if name not in self.repo:
                raise ValueError(f"Concept '{name}' not found in TAKRepository.")
            
            tak_dict = self.repo[name]

            # A raw concept has derived_from == null
            deps = __deps_from_derived_from(name, tak_dict)
            if not deps:
                # This is a raw concept (derived_from is null)
                return [name]

            # Recursively resolve each dependency (handles both single and multi-parent cases)
            all_raw_parents = []
            for dep in deps:
                all_raw_parents.extend(__resolve_top_raw(dep, seen.copy()))
            
            return all_raw_parents
        

        def _parents_for_concept(concept_name: str):
            if concept_name in ("[NULL]", "[MASK]", "[PAD]",
                                "[MASK_INTERVAL_START]", "[MASK_INTERVAL_END]"):
                return ["[NULL]"]
            
            if concept_name not in self.repo:
                raise ValueError(f"Concept '{concept_name}' not found in TAKRepository.")
            
            tak_dict = self.repo[concept_name]
            deps = __deps_from_derived_from(concept_name, tak_dict)
            
            if not deps:
                # This is a raw concept (derived_from is null)
                return [concept_name]

            # Resolve all dependencies to their raw parents
            # __resolve_top_raw now returns a list, so we need to flatten
            parents = set()
            for dep in deps:
                raw_parents = __resolve_top_raw(dep)
                parents.update(raw_parents)

            return sorted(parents)
        
        self.df['ParentRawConcepts'] = self.df['Concept'].apply(_parents_for_concept)

    
    def _expand_tokens(self, min_interval_duration_sec=1):
        """
        Expands events into tokens with timepoints.

        - Splits state events into START and END tokens.
        - Keeps instantaneous events as single tokens.
        
        Returns:
            DataFrame with ['PatientId', 'RawConcept', 'Concept', 'ValueToken', 'PositionToken', 'TimePoint'].
        """
        df = self.df
        rows = []
        for row in df.itertuples(index=False):
            duration_sec = (row.EndDateTime - row.StartDateTime).total_seconds()
            base_token = f"{row.ConceptName}_{row.Value}" if row.Value not in ("True", "TRUE") else row.ConceptName
            concept = row.ConceptName
            value = base_token

            # Create interval tokenization if duration > threshold
            is_interval = (duration_sec > min_interval_duration_sec)
            pos_tokens = []

            if is_interval:
                pos_tokens = ["START", "END"]
                time_points = [row.RelStartTime, row.RelEndTime]
            else:
                pos_tokens = [""]
                time_points = [row.RelStartTime]

            for pos, tp in zip(pos_tokens, time_points):
                full_token = f"{base_token}_{pos}" if pos else base_token
                rows.append({
                    'PatientId': row.PatientId,
                    'Concept': concept,
                    'ValueToken': value,
                    'PositionToken': full_token,
                    'TimePoint': tp
                })

        df = pd.DataFrame(rows)

        # ⚠️  CRITICAL SORTING OPERATION ⚠️
        # The following sort by ['PatientId', 'TimePoint'] is MANDATORY and must NEVER be removed,
        # reordered, or performed differently. This sorted order is a hard requirement for:
        #
        #   1. GPU-efficient temporal target generation in utils.get_temporal_multi_hot_targets()
        #      which uses torch.searchsorted() and assumes per-batch non-decreasing timestamps.
        #      Violating this will cause silent correctness bugs (wrong multi-hot targets).
        #
        #   2. Proper EMR sequence semantics: events within a visit MUST be chronological
        #      for the model to learn meaningful temporal dependencies.
        #
        # If you need to change this sorting for any reason, you MUST also update:
        #   - intervene_enc/utils.py::get_temporal_multi_hot_targets()
        #   - intervene_enc/embedder.py::train_embedder() BCE loss computation
        #   - intervene_enc/transformer.py::pretrain_transformer() BCE loss computation
        #
        self.df = df.sort_values(['PatientId', 'TimePoint']).reset_index(drop=True)
    

    def _insert_null_tokens(self, gap_hrs: int = 3) -> None:
        """
        Insert a single synthetic [NULL] token whenever there is a gap > `gap_hrs`
        *and* no interval is open (open_stack==0).  Token is placed at gap midpoint.
        """
        if gap_hrs <= 0:
            return

        rows_out = []
        for pid, grp in self.df.groupby("PatientId"):
            grp = grp.sort_values("TimePoint")            # safety
            open_stack, last_tp = 0, None

            for row in grp.itertuples(index=False):
                tp = row.TimePoint

                # ---------- gap check ----------
                if last_tp is not None:
                    gap = tp - last_tp
                    if gap >= gap_hrs and open_stack == 0:
                        rows_out.append({
                            "PatientId": pid,
                            "RawConcept": "[NULL]",
                            "Concept":    "[NULL]",
                            "ValueToken": "[NULL]",
                            "PositionToken": "[NULL]",
                            "TimePoint":  last_tp + gap / 2
                        })

                # ---------- interval stack ----------
                tok = row.PositionToken
                if tok.endswith("_START"):
                    open_stack += 1
                elif tok.endswith("_END"):
                    open_stack = max(0, open_stack - 1)

                # ---------- keep real event ----------
                rows_out.append(row._asdict())
                last_tp = tp

        # Replace dataframe
        self.df = pd.DataFrame(rows_out)


    def _cut_after_k_days(self):
        """
        Purpose: Truncate the token stream to the first `max_input_days` days.
        Method:  Keep tokens at or before the cut, then drop any patient left
                 with fewer than two tokens (an unencodable sequence).

                 Kineret tweak -- `enforce_min_duration` gates the upstream rule
                 that ALSO dropped visits whose trajectory did not extend past
                 the cut. Cohort membership is decided once, in
                 `kineret.cohort`, against both source tables; re-deciding it
                 here would silently give INTERVenE a smaller test set than the
                 raw-data models, because the inclusion/exclusion filters run
                 first and can pull a patient's last abstraction back inside the
                 window. The driver passes False for exactly that reason.

        NOTE: This version fits a data process where PatientId is actually the visitID, meaning every ID belongs 
        to only 1 group of records. If you want generation based on PatientId that can have the information of a few 
        visits you'll need to change the key here to VisitCounter, but to ensure it is also passed from _expand_tokens().

        """
        df = self.df
        k_hours = self.max_input_days * 24

        if self.enforce_min_duration:
            # Only keep visits that originally last longer than k days
            visit_max_times = df.groupby("PatientId")["TimePoint"].max()
            long_enough_visits = visit_max_times[visit_max_times > k_hours].index
            df = df[df["PatientId"].isin(long_enough_visits)].copy()

        # Keep only events up to k_days
        df = df[df["TimePoint"] <= k_hours].copy()

        # Drop visits with no records remaining
        remaining_visits = df.groupby("PatientId").size()
        df = df[df["PatientId"].isin(remaining_visits[remaining_visits > 1].index)]

        self.df = df


class EMRTokenizer:
    """
    Tokenizer + metadata container for EMR sequence modeling.

    Build this from fully processed training data (`from_processed_df`) so all
    vocabularies, token statistics, and lookup tables are consistent.

    Attributes:
        token2id (Dict[str, int]): Full vocabulary mapping ("GLUCOSE_STATE_HIGH_START").
        id2token (Dict[int, str]): Reverse mapping for decoding.
        rawconcept2id (Dict[str, int]): Vocabulary mapping for raw concepts only ("GLUCOSE").
        concept2id (Dict[str, int]): Vocabulary mapping for concept names (e.g. "GLUCOSE_STATE").
        value2id (Dict[str, int]): Vocabulary mapping for concept+value tokens ("GLUCOSE_STATE_HIGH").
        special_tokens (List[str]): Special tokens (e.g. ["[PAD]", "[MASK]", "[NULL]"]).
        token_weights (torch.Tensor): Per-token loss weights; 0.0 for special/boundary tokens, 1.0 otherwise.
        token_counts (torch.Tensor): Token counts (distribution).
        tokenid2parent_raw_ids (torch.Tensor): Lookup table mapping each PositionToken id
            to its parent raw-concept ids, padded to `parent_pad_len`.
        parent_pad_len (int): Width of `tokenid2parent_raw_ids` (max number of parent raws).
        pad_token_id (int): ID for padding token.
        mask_token_id (int): ID for MASK token.
        null_token_id (int): ID for NULL token.

    Notes:
        Required special tokens are validated at init: `[PAD]`, `[MASK]`, `[NULL]`.
    """
    def __init__(self, token2id, rawconcept2id, concept2id, value2id, special_tokens,
                 token_weights, token_counts,
                 tokenid2parent_raw_ids, parent_pad_len,
                 outcome_patient_ratios=None):
        self.token2id = token2id
        self.id2token = {i: tok for tok, i in token2id.items()}
        self.rawconcept2id = rawconcept2id
        self.concept2id = concept2id
        self.value2id = value2id
        self.special_tokens = special_tokens
        self.token_weights = token_weights
        self.token_counts = token_counts
        self.tokenid2parent_raw_ids = tokenid2parent_raw_ids
        self.parent_pad_len = parent_pad_len
        # Keys = valid (non-rare) outcome names; values = patient prevalence ratio.
        self.outcome_patient_ratios = outcome_patient_ratios or {}

        # Validate presence of mandatory special tokens.  [MASK_INTERVAL_*]
        # let the MLM masker preserve interval structure when an endpoint is
        # masked (vs leaking the original concept identity through a generic
        # [MASK]).
        required_specials = ["[PAD]", "[MASK]", "[NULL]",
                             "[MASK_INTERVAL_START]", "[MASK_INTERVAL_END]"]
        for tok in required_specials:
            if tok not in token2id:
                raise ValueError(f"[Tokenizer Error] Missing required special token: {tok}")

        self.pad_token_id = token2id["[PAD]"]
        self.mask_token_id = token2id["[MASK]"]
        self.null_token_id = token2id["[NULL]"]
        self.mask_interval_start_id = token2id["[MASK_INTERVAL_START]"]
        self.mask_interval_end_id = token2id["[MASK_INTERVAL_END]"]

    @classmethod
    def from_processed_df(cls, df, special_tokens=None):
        if special_tokens is None:
            # [MASK] for point events; [MASK_INTERVAL_*] preserve interval
            # structure on masked endpoints so the encoder still sees that an
            # interval boundary was masked (vs leaking the original concept).
            special_tokens = [
                "[PAD]", "[MASK]", "[NULL]",
                "[MASK_INTERVAL_START]", "[MASK_INTERVAL_END]",
            ]
        """
        Takes in a processed dataframe from DataProcessor.run() and builds the 
        tokenizer vocabularies and weights.
        Args:
            df (pd.DataFrame): Processed dataframe with columns ['ParentRawConcepts', 'Concept', 'ValueToken', 'PositionToken'].
            special_tokens (List[str]): List of special tokens to include in the vocab.
        Returns:
            EMRTokenizer: Initialized tokenizer object.
        """
        # --- raw concepts are taken from ParentRawConcepts (list[str]) ---
        if "ParentRawConcepts" not in df.columns:
            raise ValueError("ParentRawConcepts column missing. Dataset not preprocessed.")

        raw_set = set()
        for lst in df["ParentRawConcepts"]:
            if not isinstance(lst, list):
                raise ValueError(f"ParentRawConcepts must be list[str], got {type(lst)}")
            for x in lst:
                raw_set.add(x)

        raw_concepts = sorted(raw_set)
        concepts = sorted(df['Concept'].unique())
        values = sorted(df['ValueToken'].unique())
        positions = sorted(df['PositionToken'].unique())

        # Sort and inject special tokens in order
        raw_concepts = [tok for tok in raw_concepts if tok not in special_tokens]
        concepts = [tok for tok in concepts if tok not in special_tokens]
        values = [tok for tok in values if tok not in special_tokens]
        positions = [tok for tok in positions if tok not in special_tokens]

        raw_concepts = special_tokens + raw_concepts
        concepts = special_tokens + concepts
        values = special_tokens + values
        positions = special_tokens + positions

        token2id = {tok: i for i, tok in enumerate(positions)}
        rawconcept2id = {tok: i for i, tok in enumerate(raw_concepts)}
        concept2id = {tok: i for i, tok in enumerate(concepts)}
        value2id = {tok: i for i, tok in enumerate(values)}

        # === Define Token Weights ===
        # Multiplies the frequency-based alpha in FocalBCELoss. Default 1.0 for all tokens.
        # Outcome upweighting is handled by aux_fraction_caps in TRAINING_SETTINGS, not here.
        # [PAD] and [MASK] get 0.0 — they are not real prediction targets.
        # [NULL] stays at 1.0 — it is a real sequence token (synthetic gap marker) and
        #   the model must learn to predict quiet periods correctly.
        # ADMISSION_TOKEN gets 0.0 — it is a sequence boundary marker, not a clinical event.
        token_weights = torch.ones(len(token2id))
        # Mask tokens (generic + interval) are never valid prediction targets;
        # zero them so the BCE-based losses ignore them entirely.
        _zero_toks = ["[PAD]", "[MASK]", "[MASK_INTERVAL_START]",
                      "[MASK_INTERVAL_END]", ADMISSION_TOKEN]
        for ignore_tok in _zero_toks:
            tok_id = token2id.get(ignore_tok)
            if tok_id is not None:
                token_weights[tok_id] = 0.0
        
        all_outcomes = list(set(OUTCOMES + TERMINAL_OUTCOMES))

        # Outcome support is measured only in the POST-observation portion of each
        # hospitalization. The model is seeded with [0, OBSERVATION_WINDOW_HOURS] and
        # asked to predict what happens after, so an outcome that fires only inside
        # the seed window contributes no learnable signal to the outcome head.
        # Denominator stays as full patient count (patients with no post-window events
        # count as negatives).
        total_patients = df['PatientId'].nunique()
        post_obs_df = df[df["TimePoint"] > OBSERVATION_WINDOW_HOURS]
        patient_tokens = post_obs_df.groupby("PatientId")["PositionToken"].apply(set)

        # outcome_patient_ratios: name → prevalence ratio for outcomes that meet the threshold.
        # Keys of this dict are the canonical valid outcome list used by the outcome head.
        # Outcomes below the threshold are demoted to regular LM tokens — they stay in
        # token2id (so the backbone still sees and predicts them) but are excluded from
        # the outcome head, CBM-forbid protection, and sampler upweighting.
        outcome_patient_ratios = {}
        dropped_outcomes = []
        for out_tok in all_outcomes:
            if out_tok not in token2id:
                continue

            n_pos = int(patient_tokens.apply(lambda s: out_tok in s).sum())
            ratio = n_pos / max(total_patients, 1)

            if ratio * 100.0 >= OUTCOME_RARE_THRESHOLD_PCT:
                outcome_patient_ratios[out_tok] = round(ratio, 6)
            else:
                dropped_outcomes.append((out_tok, n_pos, ratio * 100.0))

        if dropped_outcomes:
            print(f"[Tokenizer] Dropping {len(dropped_outcomes)} rare outcomes "
                  f"(threshold={OUTCOME_RARE_THRESHOLD_PCT}% of {total_patients} patients):")
            for name, n_pos, pct in sorted(dropped_outcomes, key=lambda x: x[1]):
                print(f"  - {name}: {n_pos} patients ({pct:.2f}%)")
        
        # Add token distribution
        count_series = df["PositionToken"].value_counts()
        counts_vec   = torch.zeros(len(token2id), dtype=torch.long)
        for tok, cnt in count_series.items():
            counts_vec[token2id[tok]] = cnt
        
        # ---- Build tokenid2parent_raw_ids LUT for inference ----
        # Expect df has PositionToken + ParentRawConcepts
        pos_to_parents = {}

        for pos_tok, grp in df.groupby("PositionToken"):
            # enforce determinism: all rows for same PositionToken must agree
            unique_lists = grp["ParentRawConcepts"].apply(lambda x: tuple(x)).unique()
            if len(unique_lists) != 1:
                raise ValueError(
                    f"[Tokenizer Error] Inconsistent ParentRawConcepts for PositionToken='{pos_tok}'. "
                    f"Got {len(unique_lists)} variants (example={unique_lists[:3]})."
                )
            pos_to_parents[pos_tok] = list(unique_lists[0])

        # compute global Pmax
        Pmax = max((len(v) for v in pos_to_parents.values()), default=1)
        if Pmax <= 0:
            raise ValueError("[Tokenizer Error] Computed Pmax <= 0 for parent raw concepts.")

        # fill LUT with PAD id from token2id
        pad_id = token2id["[PAD]"]
        lut = torch.full((len(token2id), Pmax), pad_id, dtype=torch.long)

        # helper for encoding parent names -> ids
        def encode_parent_names(parent_names):
            ids = []
            for name in parent_names:
                if name not in rawconcept2id:
                    raise ValueError(f"[Tokenizer Error] Raw parent '{name}' missing from rawconcept2id.")
                ids.append(rawconcept2id[name])
            if len(ids) == 0:
                raise ValueError("[Tokenizer Error] Empty parent list encountered.")
            return ids

        # populate LUT by token id
        for tok, tid in token2id.items():
            # Special tokens map to [NULL] as their parent raw concept
            if tok in special_tokens:
                if "[NULL]" not in rawconcept2id:
                    raise ValueError("[Tokenizer Error] '[NULL]' missing from rawconcept2id.")
                null_id = rawconcept2id["[NULL]"]
                lut[tid, 0] = null_id
                continue
            
            if tok not in pos_to_parents:
                raise ValueError(
                    f"[Tokenizer Error] Missing ParentRawConcepts for token '{tok}'. "
                    f"Processed df must include all PositionToken values."
                )
            ids = encode_parent_names(pos_to_parents[tok])
            lut[tid, :len(ids)] = torch.tensor(ids, dtype=torch.long)

        return cls(token2id, rawconcept2id, concept2id, value2id, special_tokens, token_weights,
                   counts_vec, lut, Pmax,
                   outcome_patient_ratios=outcome_patient_ratios)

    def save(self, path=os.path.join(CHECKPOINT_PATH, 'tokenizer.pt')):
        torch.save({
            'token2id': self.token2id,
            'rawconcept2id': self.rawconcept2id,
            'concept2id': self.concept2id,
            'value2id': self.value2id,
            'special_tokens': self.special_tokens,
            'token_weights': self.token_weights,
            'token_counts': self.token_counts,
            'tokenid2parent_raw_ids': self.tokenid2parent_raw_ids,
            'parent_pad_len': self.parent_pad_len,
            'outcome_patient_ratios': self.outcome_patient_ratios,
            'fingerprint': self.fingerprint()
        }, path)


    @classmethod
    def load(cls, path=os.path.join(CHECKPOINT_PATH, 'tokenizer.pt')):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obj = torch.load(path, map_location=device, weights_only=True)

        tokenizer = cls(
            token2id=obj['token2id'],
            rawconcept2id=obj['rawconcept2id'],
            concept2id=obj['concept2id'],
            value2id=obj['value2id'],
            special_tokens=obj['special_tokens'],
            token_weights=obj['token_weights'].to(device),
            token_counts=obj['token_counts'].to(device),
            tokenid2parent_raw_ids=obj['tokenid2parent_raw_ids'].to(device),
            parent_pad_len=obj['parent_pad_len'],
            outcome_patient_ratios=obj.get('outcome_patient_ratios', {}),
        )
        tokenizer._loaded_fingerprint = obj.get('fingerprint')
        return tokenizer
    

    def fingerprint(self):
        return hash(frozenset(self.token2id.items()))

    def get_valid_outcomes(self) -> list:
        """
        Purpose: Return valid (non-rare) outcome names as decided at tokenizer build time.
        Method: Keys of outcome_patient_ratios are the outcomes that passed the prevalence
                threshold (OUTCOME_RARE_THRESHOLD_PCT) when the tokenizer was built.

        Returns:
            list[str]: Valid outcome names.
        """
        return list(self.outcome_patient_ratios.keys())


class EMRDataset(Dataset):
    def __init__(self, processed_df: pd.DataFrame, context_df: pd.DataFrame, tokenizer: EMRTokenizer,
                 label_source_df: pd.DataFrame = None,
                 patient_labels: pd.DataFrame = None,
                 patient_gt_time: pd.DataFrame = None):
        """
        processed_df: processed DataFrame after running DataProcessor.run() on the original temporal df.
        context_df: Also processed by DataProcessor.run().
        label_source_df: Optional un-truncated trajectory DataFrame used to derive
            patient-level labels when `processed_df` is an input-truncated view
            (e.g. the Phase-3 48 h window). When provided, each sample carries
            additional `full_position_ids` / `full_abs_ts` tensors that
            `build_patient_labels` reads to derive outcome-anywhere labels;
            the model's forward pass still consumes the truncated sequence.
        patient_labels: Kineret addition. PatientId-indexed DataFrame of 0/1
            outcome flags from `kineret.cohort`. When given, Phase-3 supervises
            against THESE rather than re-deriving labels from the token stream,
            which is what keeps INTERVenE-Enc, STraTS and LogReg on
            byte-identical targets. Token-derived labels would drift the moment
            the Mediator emitted an outcome as an interval (`X_EVENT_START`)
            rather than an instant.
        patient_gt_time: Matching DataFrame of first-occurrence hours from
            admission (inf where the outcome does not occur), same column order.

        This class performs data cleaning, as well as prperation of data for input as train of for inference as test.

        Attr:
            self.tokenizer (EMRTokenizer): A tokenizer object capable of encoding and decoding all temporal tokens (and subtokens as required)
            self.context_df (pd.DataFrame): Patient-level context features (indexed by PatientId), scaled to zero mean and unit variance.
            self.tokens_df (pd.DataFrame): Long-format temporal event dataframe with per-token attributes and timing features.
            self.patient_ids (np.ndarray): Array of unique PatientIds present in the dataset.
            self.patient_groups (Dict[str, pd.DataFrame]): Mapping from PatientId to their corresponding token DataFrame.
            self.full_seq_by_pid (Dict[str, Dict[str, Tensor]] or None): When
                `label_source_df` is given, maps PatientId to a dict with the
                un-truncated `position_ids` (long) and `abs_ts` (float, normalised
                by 336 h) tensors. Else None.
        """
        self.tokenizer = tokenizer
        self.tokens_df = processed_df
        self.context_df = context_df

        # --- Mapping function with warnings ---
        def safe_map(column, vocab, label):
            mapped = self.tokens_df[column].map(vocab)
            unknown = self.tokens_df.loc[mapped.isna(), column].unique()
            if len(unknown) > 0:
                print(f"[Warning][EMRDataset] Unknown {label} values found (count={len(unknown)}) — mapping to [MASK]:")
                for tok in unknown[:10]:
                    print(f"  - {tok}")
                # Map unknowns to [MASK] so they are treated as masked/unknown without crashing.
                # This gracefully handles val-only concepts when the tokenizer was built from train only.
                mapped = mapped.fillna(self.tokenizer.mask_token_id)
            return mapped.astype(int)

        # --- Map with validation ---
        self.tokens_df['ConceptID']    = safe_map('Concept', self.tokenizer.concept2id, 'Concept')
        self.tokens_df['ValueID']      = safe_map('ValueToken', self.tokenizer.value2id, 'ValueToken')
        self.tokens_df['PositionID']   = safe_map('PositionToken', self.tokenizer.token2id, 'PositionToken')

        self.tokens_df["ParentRawConceptIDs"] = self.tokens_df["ParentRawConcepts"].apply(
            self.__encode_parent_list
        )

        self.patient_ids = self.tokens_df['PatientId'].unique()
        # Group once via pandas groupby (O(N), uses pandas' internal C-level
        # grouping) instead of the previous O(N²) dict-of-boolean-filters,
        # which on the full 16.9M-row training split materialised 40k slice
        # DataFrames and overran the 46.6 GB cgroup during torch.save.
        self.patient_groups = dict(tuple(self.tokens_df.groupby("PatientId", sort=False)))

        # ── Un-truncated label source (Phase-3 only) ───────────────────────
        # Decouple the model's input window from the prediction-horizon label
        # window. The Phase-3 input is truncated to PHASE3_INPUT_DAYS=2, but
        # labels must reflect outcomes anywhere in the full 0–336 h trajectory
        # (matching evaluation.py). Without this, RELEASE/DEATH tokens at hour
        # ~159/~209 are sliced off the training sequence and the model is
        # trained against near-empty positives for terminal outcomes.
        # Kineret addition: cohort-supplied patient-level supervision. Stored as
        # dense float32 matrices in `label_columns` order; `build_patient_labels`
        # permutes them into the model's own outcome order at training time.
        self.label_columns = None
        self.patient_label_mat = None
        self.patient_time_mat = None
        if patient_labels is not None:
            self.label_columns = list(patient_labels.columns)
            aligned = patient_labels.reindex(self.patient_ids).fillna(0.0)
            self.patient_label_mat = torch.tensor(
                aligned.to_numpy(dtype="float32"), dtype=torch.float32)
            if patient_gt_time is not None:
                times = patient_gt_time.reindex(self.patient_ids)[self.label_columns]
                # inf marks "never happened"; the loss masks those on the label,
                # so 0.0 here is inert and keeps the tensor finite.
                times = times.replace([np.inf, -np.inf], 0.0).fillna(0.0)
                self.patient_time_mat = torch.tensor(
                    times.to_numpy(dtype="float32"), dtype=torch.float32)
            else:
                self.patient_time_mat = torch.zeros_like(self.patient_label_mat)

        self.full_seq_by_pid = None
        if label_source_df is not None:
            ls = label_source_df
            if 'PositionID' not in ls.columns:
                ls = ls.copy()
                ls['PositionID'] = ls['PositionToken'].map(self.tokenizer.token2id) \
                                                    .fillna(self.tokenizer.mask_token_id) \
                                                    .astype(int)
            self.full_seq_by_pid = {
                pid: {
                    "position_ids": torch.tensor(g["PositionID"].values, dtype=torch.long),
                    "abs_ts": torch.tensor(g["TimePoint"].values, dtype=torch.float32) / 336.0,
                }
                for pid, g in ls.groupby("PatientId", sort=False)
            }

    def __getstate__(self):
        # patient_groups is a redundant view over tokens_df: pickling 40k slice
        # DataFrames OOMs the cgroup during torch.save (api.py processed_datasets
        # cache write). We persist only tokens_df + light attrs and rebuild
        # patient_groups in __setstate__ via the same groupby used in __init__.
        state = self.__dict__.copy()
        state["patient_groups"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.patient_groups is None and self.tokens_df is not None:
            self.patient_groups = dict(
                tuple(self.tokens_df.groupby("PatientId", sort=False))
            )

    def __encode_parent_list(self, parents: List[str]) -> List[int]:
        ids = []
        for p in parents:
            if p not in self.tokenizer.rawconcept2id:
                # Unknown raw concept — map to [MASK] id so it is ignored by the embedder.
                ids.append(self.tokenizer.mask_token_id)
            else:
                ids.append(self.tokenizer.rawconcept2id[p])
        return ids


    def __len__(self):
        """
        Returns the number of patients in the dataset
        """
        return len(self.patient_ids)


    def __getitem__(self, idx, allow_debug=False):
        """
        Returns the subset of records for 1 patient.
        """
        pid = self.patient_ids[idx]
        df = self.patient_groups[pid]

        # ---------------------------
        # Build padded parent_raw_ids: [T, P]
        # ---------------------------
        if "ParentRawConceptIDs" not in df.columns:
            raise ValueError(
                f"[Dataset Error] Missing ParentRawConceptIDs for patient {pid}. "
                f"Make sure EMRDataset built ParentRawConcepts/IDs before grouping."
            )

        parent_lists = df["ParentRawConceptIDs"].tolist()

        if len(parent_lists) == 0:
            # extremely rare, but keep it safe
            max_p = 1
        else:
            # validate type and compute max parents for this patient
            for i, lst in enumerate(parent_lists[:10]):  # sample validation
                if not isinstance(lst, list):
                    raise ValueError(
                        f"[Dataset Error] ParentRawConceptIDs must be list[int]. "
                        f"Got {type(lst)} at row {i} for patient {pid}."
                    )
            max_p = max(len(x) for x in parent_lists) if parent_lists else 1
            if max_p == 0:
                raise ValueError(
                    f"[Dataset Error] Patient {pid} has empty ParentRawConceptIDs lists."
                )

        parent_raw_ids = torch.full(
            (len(parent_lists), max_p),
            self.tokenizer.pad_token_id,
            dtype=torch.long
        )

        for i, lst in enumerate(parent_lists):
            if len(lst) == 0:
                raise ValueError(
                    f"[Dataset Error] Empty ParentRawConceptIDs for patient {pid} at token index {i}."
                )
            parent_raw_ids[i, :len(lst)] = torch.tensor(lst, dtype=torch.long)

        if allow_debug:
            # Check each ID column (no RawConceptID anymore)
            for col_name, vocab_dict in [
                ("ConceptID", self.tokenizer.concept2id),
                ("ValueID", self.tokenizer.value2id),
                ("PositionID", self.tokenizer.token2id),
            ]:
                ids = df[col_name].values
                max_valid = len(vocab_dict) - 1

                if (ids > max_valid).any():
                    bad_ids = ids[ids > max_valid]
                    print(f"ERROR: Patient {pid} has out-of-bounds {col_name}: {bad_ids}")
                    print(f"  Valid range for {col_name}: [0, {max_valid}]")
                    print(f"  Actual range: [{ids.min()}, {ids.max()}]")

                    bad_positions = np.where(ids > max_valid)[0]
                    for pos in bad_positions[:3]:
                        print(f"  Position {pos}: {col_name}={ids[pos]}")
                        if col_name == "PositionID":
                            original_token = df.iloc[pos]["PositionToken"]
                            print(f"    Original token: '{original_token}'")

                    raise ValueError(f"Found out-of-bounds {col_name} for patient {pid}")

            # Validate parent_raw_ids bounds too
            max_raw_valid = len(self.tokenizer.rawconcept2id) - 1
            if (parent_raw_ids > max_raw_valid).any():
                bad = parent_raw_ids[parent_raw_ids > max_raw_valid].unique()
                raise ValueError(
                    f"Found out-of-bounds parent_raw_ids for patient {pid}: {bad.tolist()} "
                    f"(valid range [0, {max_raw_valid}])"
                )

        sample = {
            "parent_raw_ids": parent_raw_ids,  # [T, P]
            "concept_ids": torch.tensor(df["ConceptID"].values, dtype=torch.long),
            "value_ids": torch.tensor(df["ValueID"].values, dtype=torch.long),
            "position_ids": torch.tensor(df["PositionID"].values, dtype=torch.long),
            "abs_ts": torch.tensor(df["TimePoint"].values, dtype=torch.float32) / 336.0,  # [0,1]
            "context_vec": torch.tensor(self.context_df.loc[pid].values, dtype=torch.float32),
            "targets": torch.tensor(df["PositionID"].values, dtype=torch.long),
        }
        # Phase-3 only: attach the un-truncated trajectory tokens so
        # build_patient_labels can derive outcome-anywhere labels even though
        # the model's forward pass consumes the truncated sequence above.
        if self.full_seq_by_pid is not None:
            full = self.full_seq_by_pid.get(pid)
            if full is not None:
                sample["full_position_ids"] = full["position_ids"]
                sample["full_abs_ts"]       = full["abs_ts"]
        if self.patient_label_mat is not None:
            sample["patient_labels"]  = self.patient_label_mat[idx]
            sample["patient_gt_time"] = self.patient_time_mat[idx]
        return sample


def collate_emr(batch, pad_token_id=0):
    """
    Collates a batch of patient EMR sequences into padded tensors.
    All patient's trajectories in the batch are padded to the same length (max_len) and max parent count (max_p).
    This makes the common 'block_size' concept from NLP less meaningful, as we want to preserve as much temporal information as possible.

    ASSUMES: Each patient sequence in the batch is already SORTED by TimePoint (ascending).
    This sorting is performed in DataProcessor._expand_tokens() and is maintained by EMRDataset.__getitem__().
    Do NOT reorder sequences, as temporal target generation in training relies on sortedness.

    Each sequence contains:
        - Parents Concept ID (2D: [T, P])
        - Concept ID (1D: [T])
        - Value ID (1D: [T])
        - Position ID (1D: [T]) (used for prediction)
        - Absolute time (Δt since admission) (1D: [T])
        - Patient context vector (no padding) (1D: [C])
    Returns:
        Dictionary of padded tensors: [B, T_max] + context_vec [B, C]
    
    NOTE: Padding token id ([PAD]) is always 0. Time Padding should be 0.0.
    """
    batch_size = len(batch)
    max_len = max(x["position_ids"].shape[0] for x in batch)
    max_p   = max(x["parent_raw_ids"].shape[1] for x in batch)
    
    def pad_2d(seq2d, pad_val=pad_token_id, dtype=torch.long):
        out = torch.full((batch_size, max_len, max_p), pad_val, dtype=dtype)
        for i, s in enumerate(seq2d):
            t, p = s.shape
            out[i, :t, :p] = s
        return out

    def pad_1d(seq1d, pad_val=pad_token_id, dtype=torch.long):
        out = torch.full((batch_size, max_len), pad_val, dtype=dtype)
        for i, s in enumerate(seq1d):
            out[i, :len(s)] = s
        return out

    parent_raw_ids = pad_2d([x["parent_raw_ids"] for x in batch], pad_val=pad_token_id)
    concept_ids    = pad_1d([x["concept_ids"] for x in batch], pad_val=pad_token_id)
    value_ids      = pad_1d([x["value_ids"] for x in batch], pad_val=pad_token_id)
    position_ids   = pad_1d([x["position_ids"] for x in batch], pad_val=pad_token_id)
    abs_ts         = pad_1d([x["abs_ts"] for x in batch], pad_val=0.0, dtype=torch.float32)
    context_vecs   = torch.stack([x["context_vec"] for x in batch])

    out = {
        "parent_raw_ids": parent_raw_ids,
        "concept_ids": concept_ids,
        "value_ids": value_ids,
        "position_ids": position_ids,
        "abs_ts": abs_ts,
        "context_vec": context_vecs,
        "targets": position_ids.clone(),
    }

    # Phase-3 only: pad and propagate the un-truncated trajectory tensors for
    # label derivation. These are NOT seen by the model forward pass — only
    # by build_patient_labels.
    if "full_position_ids" in batch[0]:
        full_max_len = max(x["full_position_ids"].shape[0] for x in batch)
        def pad_full(seqs, pad_val, dtype):
            o = torch.full((batch_size, full_max_len), pad_val, dtype=dtype)
            for i, s in enumerate(seqs):
                o[i, :len(s)] = s
            return o
        out["full_position_ids"] = pad_full([x["full_position_ids"] for x in batch],
                                            pad_val=pad_token_id, dtype=torch.long)
        out["full_abs_ts"]       = pad_full([x["full_abs_ts"] for x in batch],
                                            pad_val=0.0,          dtype=torch.float32)

    # Kineret addition: cohort-supplied patient-level supervision is already
    # fixed-width per patient, so it stacks rather than pads.
    if "patient_labels" in batch[0]:
        out["patient_labels"]  = torch.stack([x["patient_labels"] for x in batch])
        out["patient_gt_time"] = torch.stack([x["patient_gt_time"] for x in batch])

    return out


class BucketBatchSampler(Sampler):
    """
    Groups patients by sequence length into buckets, then yields batches
    from within the same bucket. This minimises padding waste inside each
    batch without changing which patients are seen per epoch.

    Patients are sorted into `n_buckets` equal-width length buckets. Within
    each bucket the order is shuffled every epoch so the model never sees
    the same batch composition twice.
    """
    def __init__(self, dataset: "EMRDataset", batch_size: int, n_buckets: int = 20, drop_last: bool = False):
        self.batch_size = batch_size
        self.drop_last  = drop_last

        lengths = [len(dataset.patient_groups[pid]) for pid in dataset.patient_ids]
        indices = list(range(len(lengths)))

        # sort indices by length, then split into n_buckets
        indices.sort(key=lambda i: lengths[i])
        bucket_size = max(1, len(indices) // n_buckets)
        self.buckets = [indices[i: i + bucket_size] for i in range(0, len(indices), bucket_size)]

    def __iter__(self):
        # shuffle within each bucket, then collect all batches and shuffle batch order
        all_batches = []
        for bucket in self.buckets:
            shuffled = bucket.copy()
            random.shuffle(shuffled)
            for i in range(0, len(shuffled), self.batch_size):
                batch = shuffled[i: i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                all_batches.append(batch)
        random.shuffle(all_batches)
        yield from all_batches

    def __len__(self):
        if self.drop_last:
            return sum(len(b) // self.batch_size for b in self.buckets)
        return sum((len(b) + self.batch_size - 1) // self.batch_size for b in self.buckets)


class WeightedBucketBatchSampler(Sampler):
    """
    Combines weighted oversampling with length-aware batching.

    Draws indices globally with replacement according to per-sample weights
    (same as WeightedRandomSampler), then sorts the drawn pool by sequence
    length before grouping into batches. This minimises padding waste while
    preserving the rare-outcome rebalancing effect of oversampling.

    PyTorch's DataLoader does not allow a sampler and batch_sampler
    simultaneously, so this class acts as a batch_sampler that internalises
    the weighted draw.
    """
    def __init__(self, dataset: "EMRDataset", batch_size: int,
                 weights: torch.Tensor, drop_last: bool = False):
        """
        Purpose: Initialise sampler with per-patient weights and sequence lengths.

        Args:
            dataset (EMRDataset): Source dataset.
            batch_size (int): Number of patients per batch.
            weights (torch.Tensor): 1-D float tensor of per-patient sampling weights.
            drop_last (bool): Drop the final incomplete batch if True.
        """
        self.batch_size = batch_size
        self.drop_last  = drop_last
        self.weights    = weights
        self.n          = len(dataset.patient_ids)
        self.lengths    = [len(dataset.patient_groups[pid]) for pid in dataset.patient_ids]

    def __iter__(self):
        # weighted draw with replacement (one full epoch worth of samples)
        drawn = torch.multinomial(self.weights, num_samples=self.n, replacement=True).tolist()
        # sort by sequence length to minimise padding within each batch
        drawn.sort(key=lambda i: self.lengths[i])
        batches = []
        for i in range(0, len(drawn), self.batch_size):
            batch = drawn[i: i + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                continue
            batches.append(batch)
        random.shuffle(batches)
        yield from batches

    def __len__(self):
        if self.drop_last:
            return self.n // self.batch_size
        return (self.n + self.batch_size - 1) // self.batch_size


def get_dataloader(
        dataset,
        batch_size: int,
        collate_fn,
        *,
        oversample: bool = False,
        bucket_batching: bool = False,
        num_workers: int = None,
        replacement: bool = True,
):
    """
    Build a DataLoader for an EMRDataset.
    If `oversample=True`, uses a WeightedRandomSampler to balance terminal outcomes.
    """
    # Set sensible default for num_workers.
    # Capped at 2 (not 4): api.py builds three persistent dataloaders
    # (transformer_train, phase3_train, val), each forking num_workers copies
    # of the dataset. With cpu_count=8 the previous default was 4 → 12 fork
    # children. Copy-on-write keeps them cheap in raw bytes, but Python
    # refcounting touches every PyObject on every access, gradually breaking
    # COW and tripping the 46.6 GB cgroup oom-killer mid-Phase-3 (observed
    # at epoch 37 of the X-traj-length run). Two workers per loader ⇒ 6
    # forks total, ~½ the COW pressure with a small throughput hit.
    # Default 0: no fork, one copy of the dataset.
    #
    # The cap above was 2, and on this study's data 2 was still too many. Each
    # worker shares `EMRDataset` by copy-on-write, but the dataset holds
    # `patient_groups` -- one pandas DataFrame per sample, ~179k of them -- and
    # reading any of them touches Python refcounts, which copies the page. The
    # result is roughly one full copy per worker.
    #
    # Measured on the real cohort: the sigma-bin arm peaked at 10.7 GB, the KB
    # arm (about twice the rows) at 27.4 GB on a 30 GB box with no swap, and
    # the KB arm killed the machine in Phase 1. Phases 2 and 3 report 8-9 % of
    # their time waiting on data, so the workers were buying almost nothing to
    # begin with.
    #
    # Override per run with TRAINING_SETTINGS["dataloader_workers"].
    if num_workers is None:
        num_workers = 0

    # Keep workers alive across epochs to avoid Python 3.11 multiprocessing teardown
    # crashes (AssertionError: can only test a child process) and tqdm rendering errors
    # that fire when DataLoader iterators are garbage-collected between epochs.
    persistent = num_workers > 0

    # Use fork (Linux default) with pin_memory disabled.
    # pin_memory=True triggers CUDA init when the first DataLoader iterator is
    # created (via the pin-memory thread), after which forked workers inherit
    # broken PyTorch thread state and die silently. Keeping pin_memory=False
    # avoids that init; fork workers survive and run cleanly. The parallel-
    # loading gain far exceeds the non-pinned H2D transfer cost on this model.
    # api.py has no __main__ guard so spawn and forkserver both fail.
    pin_memory = False
    dl_extra = {}

    # ---------- no oversampling ----------
    if not oversample:
        if bucket_batching:
            sampler = BucketBatchSampler(dataset, batch_size=batch_size)
            return DataLoader(dataset,
                              batch_sampler=sampler,
                              collate_fn=collate_fn,
                              num_workers=num_workers,
                              persistent_workers=persistent,
                              pin_memory=pin_memory,
                              **dl_extra)
        return DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=collate_fn,
                          num_workers=num_workers,
                          persistent_workers=persistent,
                          pin_memory=pin_memory,
                          **dl_extra)

    # ---------- build sample weights ----------
    # Each patient weight = sum of (n_patients / outcome_count) for each outcome
    # they have. Patients with multiple rare outcomes get higher weights.
    # DEATH also contributes as a terminal outcome.
    tid = dataset.tokenizer.token2id
    all_outcome_tokens = [o for o in OUTCOMES + [DEATH_TOKEN] if o in tid]

    outcome_counts: Counter = Counter()
    patient_outcome_sets = []
    for pid in dataset.patient_ids:
        ids = set(dataset.patient_groups[pid]["PositionID"].tolist())
        present = frozenset(o for o in all_outcome_tokens if tid[o] in ids)
        patient_outcome_sets.append(present)
        outcome_counts.update(present)

    n_patients = len(dataset.patient_ids)
    outcome_inv_freq = {o: n_patients / max(1, outcome_counts[o]) for o in all_outcome_tokens}

    sample_w = torch.tensor(
        [max(1.0, sum(outcome_inv_freq[o] for o in present))
         for present in patient_outcome_sets],
        dtype=torch.float32,
    )

    if bucket_batching:
        batch_sampler = WeightedBucketBatchSampler(dataset, batch_size=batch_size, weights=sample_w)
        return DataLoader(dataset,
                          batch_sampler=batch_sampler,
                          collate_fn=collate_fn,
                          num_workers=num_workers,
                          persistent_workers=persistent,
                          pin_memory=pin_memory,
                          **dl_extra)

    sampler = WeightedRandomSampler(sample_w,
                                    num_samples=len(sample_w),
                                    replacement=replacement)

    return DataLoader(dataset,
                      batch_size=batch_size,
                      sampler=sampler,
                      collate_fn=collate_fn,
                      num_workers=num_workers,
                      persistent_workers=persistent,
                      pin_memory=pin_memory,
                      **dl_extra)