"""Read only selected admissions from large CSV temporal sources."""
from pathlib import Path
import pandas as pd

from kineret.io_utils import load_table, normalise_temporal


def load_temporal(path, admission_ids, origins=None, max_hours=None, chunksize=100000):
    ids = set(admission_ids)
    def select(frame):
        frame = normalise_temporal(frame)
        frame = frame[frame.PatientId.isin(ids)].copy()
        if max_hours is not None:
            if origins is None:
                raise ValueError("Admission origins are required for a prefix read.")
            hours = (frame.StartDateTime-frame.PatientId.map(origins)).dt.total_seconds()/3600
            frame = frame[hours.between(0, max_hours)]
        return frame
    if Path(path).suffix.lower() in (".csv", ".txt"):
        frames = []
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            selected = select(chunk)
            if not selected.empty:
                frames.append(selected)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value"])
    return select(load_table(str(path)))
