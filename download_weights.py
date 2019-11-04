"""Thanks to https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-
using-url"""
import zipfile
import os
from pathlib import Path

import requests


def download_file_from_google_drive(google_id, dest):
    url = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(url, params={'id': google_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': google_id, 'confirm': token}
        response = session.get(url, params=params, stream=True)

    save_response_content(response, dest)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, dest):
    chunk_size = 32768

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download_file(google_id, dest):
    dest = Path(dest)
    dest.parent.mkdir(exist_ok=True)
    override = input(f"Override existing file {dest} ?") if dest.exists() else 'y'
    if override == 'y':
        print(f"Downloading {dest}..")
        download_file_from_google_drive(google_id, dest)

    if dest.name == 'seqvec.zip':
        print(f"Unzipping {dest}..")
        with zipfile.ZipFile(dest) as zf:
            zf.extractall(dest.parent)
        [os.rename(p, p.parents[1] / p.name) for p in (dest.parent / 'uniref50_v2').iterdir()]
        (dest.parent / 'uniref50_v2').rmdir()
        os.remove(dest)


if __name__ == "__main__":
    download_file(google_id='1VaA92XizlP88AjJTPr7BJ2Nh_q2qmi5s',
                  dest='elmo_model_uniref50/seqvec.zip')

    # download_file(google_id='1JF2e7JaeRmghN5FzKG04vOrztODnZcyJ',
    #               dest='weights/bacteriocin_classifier_params.dump')
