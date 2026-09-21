"""
Build the deployment zip for the VM.

    python scripts/package.py                 # -> kineret_deploy.zip
    python scripts/package.py --out /tmp/k.zip
    python scripts/package.py --list          # show what would ship, zip nothing

Ships an ALLOWLIST -- the package, the notebook, the scripts, the tests and the
install metadata. Everything else is left behind by construction rather than by
a list of exclusions, so a stray file in the working directory can never end up
in the archive.

**Nothing under `data/`, `checkpoints/` or `outputs/` is ever included.** The
cohort is private and must not leave the environment it lives in; the target
machine trains from scratch on its own tables. Those directories are recreated
empty inside the zip so the first run does not trip on a missing path.
"""

import argparse
import os
import sys
import zipfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# What ships. Directories are walked; files are taken as they are.
INCLUDE_DIRS = ["kineret", "notebooks", "scripts", "unittests"]
INCLUDE_FILES = ["pyproject.toml", "requirements.txt", "README.md", ".gitignore"]

# Runtime directories, recreated empty so the first run has somewhere to write.
RUNTIME_DIRS = ["data/source", "data/processed", "checkpoints",
                "outputs", "outputs/figures"]

# Never shipped, wherever they appear. Build artefacts and anything that could
# carry patient data.
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".git",
             ".venv", "venv", ".hf_cache", ".idea", ".vscode"}
SKIP_SUFFIXES = (".pyc", ".pyo", ".pyd", ".so", ".log", ".pkl", ".bin", ".pt",
                 ".csv", ".xlsx", ".xls", ".parquet")
SKIP_NAMES = {".DS_Store"}

# Data files the package genuinely needs. Checked before SKIP_SUFFIXES, so the
# concept hierarchy ships while patient tables never can.
ALLOW_FILES = {os.path.join("kineret", "config", "tak_repo_portable.json"),
               os.path.join("kineret", "config", "event_rules.json")}


def _wanted(rel_path: str) -> bool:
    """
    Purpose: Decide whether one file belongs in the archive.
    Method:  Allowlisted paths always ship. Otherwise reject build artefacts and
             every data-bearing extension -- `.csv`, `.xlsx`, `.pkl`, `.pt` and
             friends -- so no cohort table, cached artefact or checkpoint can
             leave the machine even if it is sitting inside a shipped directory.

    Args:
        rel_path (str): Path relative to the repository root.

    Returns:
        bool: True when the file should be included.
    """
    if rel_path.replace("/", os.sep) in ALLOW_FILES:
        return True
    name = os.path.basename(rel_path)
    if name in SKIP_NAMES:
        return False
    return not name.endswith(SKIP_SUFFIXES)


def collect() -> list:
    """
    Purpose: The exact file list the archive will contain.
    Method:  Walk each included directory, pruning skip-listed subdirectories in
             place so their contents are never even visited, then add the loose
             metadata files.

    Returns:
        list[str]: Repository-relative paths, sorted.
    """
    found = []
    for directory in INCLUDE_DIRS:
        base = os.path.join(ROOT, directory)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for filename in filenames:
                rel = os.path.relpath(os.path.join(dirpath, filename), ROOT)
                if _wanted(rel):
                    found.append(rel)
    for filename in INCLUDE_FILES:
        if os.path.exists(os.path.join(ROOT, filename)):
            found.append(filename)
    return sorted(found)


def build(out_path: str, files: list) -> str:
    """
    Purpose: Write the archive.
    Method:  Deflated zip, plus an empty `.gitkeep` in each runtime directory.
             Archive names always use forward slashes: a zip built on Windows
             would otherwise carry backslashes in its entry names, and `unzip`
             on the Linux target treats those as part of the filename rather
             than as separators -- producing one flat directory of files with
             literal backslashes in their names.

    Args:
        out_path (str):       Destination zip.
        files    (list[str]): Repository-relative paths to include.

    Returns:
        str: The path written.
    """
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for rel in files:
            archive.write(os.path.join(ROOT, rel), rel.replace(os.sep, "/"))
        for directory in RUNTIME_DIRS:
            archive.writestr(f"{directory}/.gitkeep", "")
    return out_path


def parse_args():
    """Purpose: CLI surface for the packager."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(ROOT, "kineret_deploy.zip"),
                   help="Destination zip. Default: kineret_deploy.zip at the root.")
    p.add_argument("--list", action="store_true",
                   help="Print what would ship and exit without writing.")
    return p.parse_args()


def main():
    """Purpose: Build the deployment zip and report what went into it."""
    args = parse_args()
    files = collect()

    if args.list:
        for rel in files:
            print(rel.replace(os.sep, "/"))
        print(f"\n{len(files)} files (+{len(RUNTIME_DIRS)} empty runtime dirs)")
        return

    build(args.out, files)
    size_mb = os.path.getsize(args.out) / 1024 ** 2
    print(f"Packaged {len(files)} files -> {args.out} ({size_mb:.2f} MB)")
    print("Excluded by construction: data/, checkpoints/, outputs/, "
          "virtualenvs, caches, and every .csv/.xlsx/.pkl/.pt.")
    print("\nOn the VM:")
    print("  unzip kineret_deploy.zip -d kineret && cd kineret")
    print("  python -m venv .venv && source .venv/bin/activate")
    print("  pip install -e .")
    print("  # drop the four source tables into data/source/:")
    print("  #   mediator_input.csv   mediator_output.csv")
    print("  #   context_data.csv     qa_scores.csv")
    print("  jupyter lab notebooks/benchmark.ipynb")


if __name__ == "__main__":
    main()
