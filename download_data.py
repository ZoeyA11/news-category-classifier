"""
================================================================================
 Dataset download from Kaggle
================================================================================

 Usage:  python src/download_data.py

 Dataset: News Category Dataset (HuffPost), licensed CC BY 4.0
 https://www.kaggle.com/datasets/rmisra/news-category-dataset

--------------------------------------------------------------------------------
 Two things worth knowing
--------------------------------------------------------------------------------
 1. No credentials are required.
    Kaggle's public endpoint for downloading public *datasets* works without an
    API token and without accepting any rules. Note this is specific to
    datasets: downloading *competition* data does require both a token and
    accepting the competition rules on the website, which cannot be fully
    automated.

 2. Certificates are verified against the system certificate store.
    On networks with TLS inspection (a corporate proxy, for example) the
    certificate chain is re-signed by a corporate CA. The certificate bundle
    shipped with Python (certifi) does not know that CA, so the request fails
    with CERTIFICATE_VERIFY_FAILED. `truststore` redirects verification to the
    operating system's certificate store, where the corporate CA *is* installed.

    Note that this KEEPS certificate verification switched on. Do not "fix" this
    with verify=False or ssl._create_unverified_context - that removes
    protection against man-in-the-middle interception rather than solving the
    problem.
================================================================================
"""

import io
import os
import urllib.request
import zipfile

import truststore

DATASET_REF = "rmisra/news-category-dataset"
TARGET_DIR = "data/news_category"
EXPECTED_FILE = "News_Category_Dataset_v3.json"


def download(dataset_ref: str = DATASET_REF, target_dir: str = TARGET_DIR) -> str:
    """
    Downloads and extracts the dataset. Idempotent - if the file is already
    present it is left alone, so this is safe to run repeatedly.
    """
    truststore.inject_into_ssl()   # verify against the system store (still verified)
    os.makedirs(target_dir, exist_ok=True)

    dest = os.path.join(target_dir, EXPECTED_FILE)
    if os.path.exists(dest):
        print(f"File already present: {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")
        return dest

    url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_ref}"
    print(f"Downloading {dataset_ref} ...")
    request = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(request, timeout=300) as response:
        blob = response.read()
    print(f"Downloaded {len(blob) / 1e6:.2f} MB")

    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        print("Archive contents:", archive.namelist())
        archive.extractall(target_dir)

    print(f"Extracted to: {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")
    return dest


if __name__ == "__main__":
    import console_utf8  # noqa: F401  (switches the Windows console to UTF-8)

    download()
