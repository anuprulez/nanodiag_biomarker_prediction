import requests
import zipfile
import os
import datetime
import uuid

import preprocess_data
import train_model
import train_model_sub_graph

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

    # Extract into output_dir (flatten top-level folder)
    print(f"[+] Extracting into {output_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.infolist():
            # Get only the filename (ignoring any subfolders in the ZIP)
            filename = os.path.basename(member.filename)
            if not filename:
                continue  # skip directories
            target_path = os.path.join(output_dir, filename)

            # Extract file
            with zip_ref.open(member) as source, open(target_path, "wb") as target:
                target.write(source.read())

    # Optionally remove the zip after extraction
    os.remove(zip_path)

def run_training():
    config = OmegaConf.load("../config/config.yaml")
    extract_preprocessed_data(config) if config.download_preprocessed_data else None
    preprocess_data.read_files(config)
    #train_model.train_gnn_model(config)
    train_model_sub_graph.train_gnn_model(config)


if __name__ == "__main__":
    
    run_training()