"""
Tests for admission identity and framing.

`mediator_input.csv` carries BOTH identities -- `PatientId` is the person,
`VisitId` is the admission -- while `mediator_output.csv` carries only the
admission, under the name `PatientId`, because the Mediator accepts one id
column. Joining the two on `PatientId` compares people against admissions:
almost nothing matches, and the cohort collapses to the handful of numeric
collisions. That failure is silent -- it looks like a strict filter, not a bug --
so it is pinned down here.

The same export also carries the admission window it was cut on
(`AdmissionStart` / `AdmissionEnd`) and a `relevant_admission` flag.
"""

import numpy as np
import pandas as pd

from kineret.config import data_config as C
from kineret.config import paths
from kineret.io_utils import load_table, normalise_temporal


class TestRekeyingOntoTheAdmission:

    def test_visit_id_becomes_the_join_key(self):
        frame = pd.DataFrame({
            "PatientId": [1000185], "VisitId": [1615994],
            "ConceptName": ["HEART_RATE_MEASURE"],
            "StartDateTime": ["2021-04-28 12:57:00+00:00"],
            "EndDateTime": ["2021-04-28 12:57:01+00:00"], "Value": [83.0],
        })
        out = normalise_temporal(frame)
        assert out["PatientId"].iloc[0] == 1615994, "the visit must win"
        assert out["PersonId"].iloc[0] == 1000185, "the person must be preserved"

    def test_a_visit_only_table_is_untouched(self):
        """The Mediator output has no VisitId column; its PatientId IS the visit."""
        frame = pd.DataFrame({
            "PatientId": [1615994], "ConceptName": ["HYPERGLYCEMIA_EVENT"],
            "StartDateTime": ["2021-04-28 12:57:00"],
            "EndDateTime": ["2021-04-28 12:57:00"], "Value": ["True"],
        })
        out = normalise_temporal(frame)
        assert out["PatientId"].iloc[0] == 1615994
        assert "PersonId" not in out.columns

    def test_both_source_tables_land_on_one_id_space(self, cohort):
        """
        The regression this whole module exists for: keyed correctly, the two
        files overlap almost completely. Keyed on the person, they shared 220
        ids out of 135,000 and the cohort came out at 84 patients.
        """
        raw = normalise_temporal(load_table(paths.RAW_TEMPORAL_FILE))
        abstract = normalise_temporal(load_table(paths.ABSTRACT_FILE))
        shared = set(raw["PatientId"]) & set(abstract["PatientId"])
        smaller = min(raw["PatientId"].nunique(), abstract["PatientId"].nunique())
        assert len(shared) / smaller > 0.9, (
            f"only {len(shared):,} of {smaller:,} ids are shared -- the two "
            f"tables are keyed on different things")

    def test_the_cohort_keeps_essentially_everyone(self, cohort):
        """One-sided coverage should be the exception, not the rule."""
        assert len(cohort.patients) > 0
        abstract = normalise_temporal(load_table(paths.ABSTRACT_FILE))
        assert len(cohort.patients) / abstract["PatientId"].nunique() > 0.5

    def test_timezone_aware_and_naive_stamps_agree(self):
        """
        The raw export is tz-aware and the Mediator output is naive. If the two
        were not normalised the same way, every cross-file comparison would be
        silently offset -- including the 1-minute event alignment.
        """
        aware = normalise_temporal(pd.DataFrame({
            "PatientId": [1], "VisitId": [9], "ConceptName": ["X"],
            "StartDateTime": ["2021-04-28 12:57:00+00:00"],
            "EndDateTime": ["2021-04-28 12:57:00+00:00"], "Value": [1]}))
        naive = normalise_temporal(pd.DataFrame({
            "PatientId": [9], "ConceptName": ["X"],
            "StartDateTime": ["2021-04-28 12:57:00"],
            "EndDateTime": ["2021-04-28 12:57:00"], "Value": [1]}))
        assert aware["StartDateTime"].iloc[0] == naive["StartDateTime"].iloc[0]
        assert aware["StartDateTime"].dt.tz is None


class TestRelevantAdmissionFlag:

    def _frame(self, flags):
        from kineret.cohort import _drop_irrelevant_rows
        base = pd.DataFrame({
            "PatientId": [1] * len(flags),
            "ConceptName": ["X"] * len(flags),
            "StartDateTime": pd.date_range("2024-01-01", periods=len(flags), freq="h"),
            "relevant_admission": flags,
        })
        return _drop_irrelevant_rows(base, "test", verbose=False)

    def test_false_rows_are_dropped(self):
        assert len(self._frame([True, False, True])) == 2

    def test_blank_rows_are_kept(self):
        """The ADMISSION row carries no flag -- dropping blanks kills the anchor."""
        assert len(self._frame([None, True, np.nan])) == 3

    def test_string_booleans_are_understood(self):
        """CSV round-trips booleans as text."""
        assert len(self._frame(["True", "False", "true", "false"])) == 2


class TestAdmissionWindowColumns:

    def test_window_is_read_per_admission(self):
        from kineret.cohort import _admission_window
        frame = normalise_temporal(pd.DataFrame({
            "PatientId": [1, 1], "VisitId": [9, 9], "ConceptName": ["X", "Y"],
            "StartDateTime": ["2021-04-28 13:00:00", "2021-04-28 14:00:00"],
            "EndDateTime": ["2021-04-28 13:00:00", "2021-04-28 14:00:00"],
            "Value": [1, 2],
            "AdmissionStart": ["2021-04-28 12:50:36", "2021-04-28 12:50:36"],
            "AdmissionEnd": ["2021-05-04 15:08:10", "2021-05-04 15:08:10"]}))
        start, end = _admission_window(frame)
        assert start.loc[9] == pd.Timestamp("2021-04-28 12:50:36")
        assert end.loc[9] == pd.Timestamp("2021-05-04 15:08:10")

    def test_absent_columns_give_none(self):
        from kineret.cohort import _admission_window
        frame = pd.DataFrame({"PatientId": [1], "StartDateTime": [pd.Timestamp("2024-01-01")]})
        assert _admission_window(frame) == (None, None)

    def test_cohort_anchors_on_admission_start(self, cohort):
        """t=0 is the admission the extract was cut on, not the first event."""
        raw = normalise_temporal(load_table(paths.RAW_TEMPORAL_FILE))
        if "AdmissionStart" not in raw.columns:
            return
        expected = raw.groupby("PatientId")["AdmissionStart"].min()
        got = cohort.patients.set_index("PatientId")["admission_time"]
        shared = got.index.intersection(expected.index)
        assert len(shared)
        assert (got.loc[shared] == expected.loc[shared]).all()

    def test_length_of_stay_matches_the_window(self, cohort):
        raw = normalise_temporal(load_table(paths.RAW_TEMPORAL_FILE))
        if "AdmissionEnd" not in raw.columns:
            return
        end = raw.groupby("PatientId")["AdmissionEnd"].max()
        patients = cohort.patients.set_index("PatientId")
        shared = patients.index.intersection(end.index)
        expected = ((end.loc[shared] - patients.loc[shared, "admission_time"])
                    .dt.total_seconds() / 3600.0)
        assert np.allclose(patients.loc[shared, "los_hours"], expected, equal_nan=True)

    def test_every_admission_has_a_length_of_stay(self, cohort):
        """
        A missing LoS is a masked target, so those admissions teach the LoS head
        nothing. Reading the window column recovers the ones no RELEASE row
        covered.
        """
        assert cohort.patients["los_hours"].notna().all()


class TestPersonRecovery:

    def test_person_map_can_come_from_the_raw_export(self, cohort):
        """
        The raw file names the person on every row, so person-grouped splitting
        works even if the context table omits `person_id`.
        """
        raw = normalise_temporal(load_table(paths.RAW_TEMPORAL_FILE))
        if "PersonId" not in raw.columns:
            return
        pairs = raw[["PatientId", "PersonId"]].drop_duplicates("PatientId")
        expected = dict(zip(pairs["PatientId"], pairs["PersonId"]))
        for visit in list(cohort.patients["PatientId"])[:50]:
            if visit in expected:
                assert cohort.person_of_patient[visit] == expected[visit]

    def test_one_visit_maps_to_exactly_one_person(self, cohort):
        raw = normalise_temporal(load_table(paths.RAW_TEMPORAL_FILE))
        if "PersonId" not in raw.columns:
            return
        counts = raw.groupby("PatientId")["PersonId"].nunique()
        assert (counts <= 1).all(), "an admission belongs to two people"
