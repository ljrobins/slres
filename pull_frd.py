import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import gzip
import shutil
import concurrent.futures
import re

# Base URL of the file server
BASE_URL = "https://edc.dgfi.tum.de/pub/slr/data/fr_crd/"
OUTPUT_DIR = "/media/liam/Backup/data/slr/data/fr_crd"


def ensure_directory(path):
    """Ensure the directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)


def download_file(url, output_path):
    """Download a file with a progress bar and extract it if it's .gz."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))
        temp_path = output_path + ".tmp"

        with open(temp_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)

        # Rename downloaded file
        os.rename(temp_path, output_path)

        # If the file is .gz, extract it
        if output_path.endswith(".gz"):
            extracted_path = output_path[:-3]  # Remove .gz extension
            with (
                gzip.open(output_path, "rb") as gz_file,
                open(extracted_path, "wb") as out_file,
            ):
                shutil.copyfileobj(gz_file, out_file)
            os.remove(output_path)  # Delete the .gz file after extraction

    else:
        print(f"Failed to download {url} (status code {response.status_code})")


def get_links(url):
    """Extract file and directory links from the HTML directory listing."""
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to access {url}")
        return [], []

    soup = BeautifulSoup(response.text, "html.parser")
    file_links = []
    dir_links = []

    # Regex pattern for monthly files: *_YYYYMM.frd.gz (not *_YYYYMMDD.frd.gz)
    monthly_file_pattern = re.compile(r".*_\d{6}\..*")

    for link in soup.find_all("a"):
        href = link.get("href")
        if not href or href in ("../", "/"):
            continue

        full_url = urljoin(url, href)

        # Check if it's a directory (heuristic: ends with '/')
        if monthly_file_pattern.match(href):  # Only allow monthly .gz files
            file_links.append(full_url)
        if href.startswith("/pub/slr/data/") and not "." in href and not 'quarantine' in href:
            dir_links.append(full_url)

    return file_links, dir_links


def get_satellite_folders(url):
    """Extract top-level satellite folder links."""
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to access {url}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    satellite_links = []

    for link in soup.find_all("a", style=True):  # Look for inline-styled <a> tags
        href = link.get("href")
        if href and href.startswith(
            "/pub/slr/data/fr_crd"
        ):  # Satellite folders end with '/'
            full_url = urljoin(url, href)
            satellite_links.append(full_url)

    return satellite_links


def crawl_and_download(base_url, output_dir):
    """Recursively crawl and download all .gz files while maintaining directory structure."""
    ensure_directory(output_dir)
    file_links, dir_links = get_links(base_url)

    # Download all .gz files in the current directory
    for file_url in file_links:
        file_name = os.path.basename(file_url)
        file_path = os.path.join(output_dir, file_name)

        if ".gz" in file_path:
            save_path = file_path[:-3]
        else:
            save_path = file_path

        if not os.path.exists(save_path):  # Avoid redownloading, cut off the .gz
            print(f"Downloading: {file_url}")
            download_file(file_url, file_path)

    # Recursively explore directories
    for dir_url in dir_links:
        dir_name = os.path.basename(dir_url.rstrip("/"))  # Extract directory name
        new_output_dir = os.path.join(output_dir, dir_name)
        crawl_and_download(dir_url, new_output_dir)


if __name__ == "__main__":
    satellite_folders = get_satellite_folders(BASE_URL)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_sat = {
            executor.submit(
                crawl_and_download,
                sat_url,
                os.path.join(OUTPUT_DIR, os.path.basename(sat_url.rstrip("/"))),
            ): sat_url
            for sat_url in satellite_folders
        }

        for future in concurrent.futures.as_completed(future_to_sat):
            sat_url = future_to_sat[future]
            try:
                future.result()  # Raise exceptions if any occur
                print(f"Finished processing: {sat_url}")
            except Exception as e:
                print(f"Error processing {sat_url}: {e}")

    print("Download complete!")
