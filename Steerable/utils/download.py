import os
import requests
import zipfile
import tarfile
from tqdm import tqdm
from urllib.parse import urlparse, unquote

def download_and_extract(url: str, download_dir: str):
    # Ensure target directory exists
    os.makedirs(download_dir, exist_ok=True)

    # Derive filename from URL
    default_name = "download"
    parsed = urlparse(url)
    fname = unquote(os.path.basename(parsed.path)) or default_name
    archive_path = os.path.join(download_dir, fname)

    # If we've already got the file, skip download
    if os.path.exists(archive_path):
        print(f"{archive_path!r} already exists; skipping download.")
    else:
        # Download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_bytes = int(response.headers.get('content-length', 0))
        chunk_size = 8192
        with open(archive_path, 'wb') as f, tqdm(
            total=total_bytes,
            unit='B', unit_scale=True, unit_divisor=1024,
            desc=f"Downloading {fname}"
        ) as dl_bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    dl_bar.update(len(chunk))
        print(f"Downloaded to {archive_path!r}.")

    # Extract with progress bar
    lname = fname.lower()
    if lname.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as z:
            members = z.namelist()
            with tqdm(members, desc=f"Extracting {fname}") as ex_bar:
                for member in ex_bar:
                    z.extract(member, download_dir)

    elif lname.endswith(('.tar', '.tar.gz', '.tgz')):
        with tarfile.open(archive_path, 'r:*') as t:
            members = t.getmembers()
            with tqdm(members, desc=f"Extracting {fname}") as ex_bar:
                for member in ex_bar:
                    t.extract(member, download_dir)
    else:
        print(f"No extraction: unrecognized extension on {archive_path!r}")
        return
    print(f"Extracted to {download_dir!r}.")
    
