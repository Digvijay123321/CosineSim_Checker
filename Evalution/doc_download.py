
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from google.cloud import storage
import json
from transformers import RobertaTokenizer, RobertaModel, AutoModelForMaskedLM


def documents_download():
    print("Starting document collection process...")

    # Temporary directory
    temp_dir = "./temp/"
    os.makedirs(temp_dir, exist_ok=True)

    # GCP Bucket details
    gcp_bucket_name = "auditpulse-data"

    # Define the two main folders
    doc1_folder = "Evaluation/Doc4/"
    doc2_folder = "Evaluation/Doc5/"

    # Download files
    doc1_files = download_all_files_from_gcp(gcp_bucket_name, doc1_folder, temp_dir)
    doc2_files = download_all_files_from_gcp(gcp_bucket_name, doc2_folder, temp_dir)
    print("Files downloaded:\nDoc1: ", doc1_files)
    print("Files downloaded:\nDoc2: ", doc2_files)

    # Process Doc1 files into input.txt
    input_text_path = os.path.join(temp_dir, "input.txt")
    with open(input_text_path, 'w', encoding='utf-8') as f:
        f.write(extract_text_from_files(doc1_files))
    print(f"Saved extracted text from Doc1 to {input_text_path}")


    # Process Doc2 files into generated.txt
    gen_text_path = os.path.join(temp_dir, "generated.txt")
    with open(gen_text_path, 'w', encoding='utf-8') as f:
        f.write(extract_text_from_files(doc2_files))
    print(f"Saved extracted text from Doc1 to {gen_text_path}")

    print("Document collection process completed.")

def download_all_files_from_gcp(bucket_name, folder_name, local_folder):
    """Downloads all files from a GCP folder."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder_name)
    os.makedirs(local_folder, exist_ok=True)

    downloaded_files = []
    for blob in blobs:
        if not blob.name.endswith("/"):
            local_path = os.path.join(local_folder, os.path.basename(blob.name))
            blob.download_to_filename(local_path)
            downloaded_files.append(local_path)
            print(f"Downloaded {blob.name} to {local_path}")

    return downloaded_files

def extract_text_from_files(files):
    """Extracts and consolidates text from JSON, TXT, and PDF files."""
    text_content = ""

    for file_path in files:
        if file_path.endswith(".json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    json_data = json.load(f)
                    if isinstance(json_data, dict):
                        for key, value in json_data.items():
                            text_content += f"{key}: {value}\n\n"
                    elif isinstance(json_data, list):
                        for item in json_data:
                            if isinstance(item, dict):
                                for key, value in item.items():
                                    text_content += f"{key}: {value}\n\n"
                except json.JSONDecodeError:
                    text_content += f"Invalid JSON format in {file_path}\n\n"

        elif file_path.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content += f.read() + "\n"

        elif file_path.endswith(".pdf"):
            text_content += extract_text_from_pdf(file_path) + "\n"

    return text_content

if __name__ == "__main__":
    documents_download()
