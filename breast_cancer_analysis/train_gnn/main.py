import requests
import zipfile
import os
import datetime
import uuid

import preprocess_data
import train_model

from omegaconf.omegaconf import OmegaConf

def extract_preprocessed_data(config):

    # Your Zenodo link
    url = config.p_processed_data
    output_dir = config.p_base
    os.makedirs(output_dir, exist_ok=True)

    zip_path = os.path.join(output_dir, "download.zip")

    # Download ZIP
    print(f"[+] Downloading {url} ...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    # Extract all files into output_dir
    print(f"[+] Extracting into {output_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    # Optionally remove the zip after extraction
    os.remove(zip_path)

def run_training():
    config = OmegaConf.load("../config/config.yaml")
    extract_preprocessed_data(config) if config.use_preprocessed_data else None
    preprocess_data.read_files(config)
    train_model.train_gnn_model(config)


if __name__ == "__main__":
    
    run_training()