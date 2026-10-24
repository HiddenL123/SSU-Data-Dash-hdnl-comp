"""Load the competition data.

Files in the project's data/ folder are used first. Otherwise, files are downloaded
once from Hugging Face and kept in a cache on your computer. You can also point
DATA_DIR at another folder containing the files. Historical "listings" is
strictly loaded from DATA_DIR when set, so chronological experiment fingerprints
cannot silently change. Prediction-only test files may fall back to project
data/ or Hugging Face when absent from DATA_DIR:

    macOS / Linux:       DATA_DIR=/path/to/folder python run.py validate
    Windows PowerShell:  $env:DATA_DIR = "C:/path/to/folder"; python run.py validate
"""
import os
from pathlib import Path

import pandas as pd

HF_DATASET = "Baliyang/SSU_data_dash"
FILE_NAMES = {
    "listings": "listings",            # 150,000 past sales, with sold_price (train on these)
    "off_market": "test_off_market",   # homes to value with no list price
    "list_aware": "test_list_aware",   # homes to value that do have a list price
}
LOCAL_DATA_DIR = Path(__file__).parent / "data"
DATE_COLUMNS = ["date_listed", "date_sold", "external_avm_estimate_date"]

# Columns the test files do not have: they are only known once a home has sold.
AFTER_SALE_COLUMNS = ["sold_price", "date_sold", "days_on_market",
                      "external_avm_estimate", "external_avm_estimate_date"]

DOWNLOAD_HELP = """
Could not download {file} from Hugging Face.

Check your internet connection and try again. If it still fails, download the files by hand from
https://huggingface.co/datasets/{repo} and set DATA_DIR to their folder (see data.py).

The error from Hugging Face was: {error}
"""


def _find_local(folder, stem):
    """Prefer Parquet over CSV, preserving the original file-selection order."""
    for ending in (".parquet", ".csv"):
        path = Path(folder) / (stem + ending)
        if path.is_file():
            return path
    return None


def find_file(name):
    """Return a data file, keeping historical listings strictly source-pinned.

    DATA_DIR may contain just listings.parquet for reproducible Colab/Windows
    training. When a prediction-only test file is missing from that override,
    look in the project's ordinary data/ directory, then the HF dataset cache.
    This fallback NEVER applies to labelled listings or changes the
    experiment's training-data fingerprint.
    """
    stem = FILE_NAMES[name]
    folder = os.environ.get("DATA_DIR")
    data_dir = Path(folder) if folder else LOCAL_DATA_DIR
    selected = _find_local(data_dir, stem)
    if selected is not None:
        return selected
    if folder:
        if name == "listings":
            raise FileNotFoundError(
                f"DATA_DIR is {folder}, but {stem}.parquet or {stem}.csv is not in it."
            )
        # Test-only fallback: a listings-only Parquet override must not
        # prevent inference when test files exist in the normal data folder.
        selected = _find_local(LOCAL_DATA_DIR, stem)
        if selected is not None:
            return selected

    from huggingface_hub import hf_hub_download
    try:
        return Path(hf_hub_download(HF_DATASET, stem + ".parquet", repo_type="dataset"))
    except Exception as error:
        message = DOWNLOAD_HELP.format(file=stem + ".parquet", repo=HF_DATASET,
                                    error=f"{type(error).__name__}: {error}")
        raise RuntimeError(message) from None


def load(name):
    """Load "listings", "off_market" or "list_aware" as a pandas DataFrame."""
    path = find_file(name)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    for column in DATE_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column])
    return df


def hide_answers(homes, track):
    """Keep only the columns a test file of this track has.

    Validation uses this so that a model is never shown the answer (sold_price) or
    anything else that is only known after the sale. Off-market homes also lose
    their list price.
    """
    hidden = AFTER_SALE_COLUMNS + (["list_price"] if track == "off_market" else [])
    return homes.drop(columns=[c for c in hidden if c in homes.columns])
