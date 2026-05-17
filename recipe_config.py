import os
from pathlib import Path


DATA_ROOT = Path(os.environ.get("RECIPEBOX_DATA_ROOT", "."))
READER_EMAILS = frozenset(
    email.strip().lower()
    for email in os.environ.get(
        "RECIPEBOX_READER_EMAILS",
        "richard.d.corbett@gmail.com,payal.sippy@gmail.com",
    ).split(",")
    if email.strip()
)
WRITER_EMAILS = frozenset(
    email.strip().lower()
    for email in os.environ.get(
        "RECIPEBOX_WRITER_EMAILS",
        "richard.d.corbett@gmail.com,payal.sippy@gmail.com",
    ).split(",")
    if email.strip()
)


def data_path(*parts):
    """Return a path inside the configured private data root."""
    return DATA_ROOT.joinpath(*parts)


def dataset_path(name):
    """Return a dataset-relative path as text for dataframe metadata."""
    return str(data_path(name))
