from flask import Flask, render_template, request, redirect, url_for, jsonify
import json
import os
import shutil
from datetime import datetime
import subprocess
import PyPDF2
import openai
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModel
import torch
from transformers import RobertaTokenizer, RobertaModel, AutoModelForMaskedLM
from google.cloud import firestore, storage
import tempfile
import traceback

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def clear_temp_folder(folder_path="temp"):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or symbolic link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory and contents
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
        print(f"Folder '{folder_path}' has been cleared.")
    else:
        print(f"Folder '{folder_path}' does not exist.")


def documents_download(path1):
    print("Starting document collection process...")

    # Temporary directory
    temp_dir = "./temp/"
    os.makedirs(temp_dir, exist_ok=True)

    # GCP Bucket details
    gcp_bucket_name = "auditpulse-data"

    # Define the two main folders
    doc1_folder = path1
    print(doc1_folder)

    # Download files
    doc1_files = download_all_files_from_gcp(gcp_bucket_name, doc1_folder, temp_dir)

    print("Files downloaded:\nDoc1:", doc1_files)

    # Process Doc1 files into input.txt
    input_text_path = os.path.join(temp_dir, "input.txt")
    with open(input_text_path, 'w', encoding='utf-8') as f:
        f.write(extract_text_from_files(doc1_files))
    print(f"Saved extracted text from Doc1 to {input_text_path}")


    print("Document collection process completed.")


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

        # elif file_path.endswith(".txt"):
        #     with open(file_path, 'r', encoding='utf-8') as f:
        #         text_content += f.read() + "\n"

        # elif file_path.endswith(".pdf"):
        #     text_content += extract_text_from_pdf(file_path) + "\n"

    return text_content


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

def load_sbert_model():
    model_name = 'Muennighoff/SBERT-base-nli-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

# Function to load model and tokenizer for BERT
def load_bert_model():
    model_name = 'google-bert/bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

# Function to load model and tokenizer for RoBERTa
def load_roberta_model():
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    return tokenizer, model

# Function to load model and tokenizer for mBERT
def load_mbert_model():
    tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
    model = AutoModel.from_pretrained('answerdotai/ModernBERT-base')
    return tokenizer, model

# Function to load embeddings from GCP bucket
def load_embeddings_from_gcs(bucket_name, gcs_file_path):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(gcs_file_path)
    temp_file_path = '/tmp/' + os.path.basename(gcs_file_path)

    # Download the file from GCS
    blob.download_to_filename(temp_file_path)

    # Load .pt file (assuming it is a PyTorch tensor)
    embeddings = torch.load(temp_file_path)
    return embeddings

# Function to calculate cosine similarity
def calculate_similarity(input_embedding, saved_embedding):
    return 1 - cosine(input_embedding, saved_embedding)

# Function to convert text to embedding using SBERT
def sbert_embedding(text):
    tokenizer, model = load_sbert_model()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze()

# Function to convert text to embedding using BERT
def bert_embedding(text):
    tokenizer, model = load_bert_model()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze()

# Function to convert text to embedding using RoBERTa
def roberta_embedding(text):
    tokenizer, model = load_roberta_model()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze()

# Function to convert text to embedding using mBERT
def mbert_embedding(text):
    tokenizer, model = load_mbert_model()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze()

# Main function to compute similarity scores
def evaluate_similarity(input_text_path):
    with open(input_text_path, 'r') as file:
        input_text = file.read()

    # Get input embedding for all models
    input_sbert = sbert_embedding(input_text)
    input_bert = bert_embedding(input_text)
    input_roberta = roberta_embedding(input_text)
    input_mbert = mbert_embedding(input_text)

    # Convert embeddings to PyTorch tensors
    input_sbert = torch.tensor(input_sbert)
    input_bert = torch.tensor(input_bert)
    input_roberta = torch.tensor(input_roberta)
    input_mbert = torch.tensor(input_mbert)

    # Load saved embeddings from GCP
    sbert_embeddings = load_embeddings_from_gcs('auditpulse-data', 'Evaluation/Doc2/10k_sbert_embedding.pt')
    bert_embeddings = load_embeddings_from_gcs('auditpulse-data', 'Evaluation/Doc2/10k_bert_embedding.pt')
    roberta_embeddings = load_embeddings_from_gcs('auditpulse-data', 'Evaluation/Doc2/10k_roberta_embedding.pt')
    mbert_embeddings = load_embeddings_from_gcs('auditpulse-data', 'Evaluation/Doc2/10k_mbert_embedding.pt')

    # Convert saved embeddings to PyTorch tensors
    sbert_embeddings = torch.tensor(sbert_embeddings)
    bert_embeddings = torch.tensor(bert_embeddings)
    roberta_embeddings = torch.tensor(roberta_embeddings)
    mbert_embeddings = torch.tensor(mbert_embeddings)

    # Calculate similarity for each model's embeddings
    sbert_similarity = calculate_similarity(input_sbert, sbert_embeddings)
    bert_similarity = calculate_similarity(input_bert, bert_embeddings)
    roberta_similarity = calculate_similarity(input_roberta, roberta_embeddings)
    mbert_similarity = calculate_similarity(input_mbert, mbert_embeddings)

    # Print the results
    print(f"SBERT Similarity: {sbert_similarity}")
    print(f"BERT Similarity: {bert_similarity}")
    print(f"RoBERTa Similarity: {roberta_similarity}")
    print(f"mBERT Similarity: {mbert_similarity}")



def calculate_similarity(input_embedding, saved_embeddings):
    # Ensure both input and saved embeddings are 1D tensors
    if input_embedding.dim() > 1:
        input_embedding = input_embedding.flatten()  # Flatten to 1D if necessary

    similarities = []
    
    # Loop through each saved embedding
    for saved_embedding in saved_embeddings:
        if saved_embedding.dim() > 1:
            saved_embedding = saved_embedding.flatten()  # Flatten to 1D if necessary
        dot_product = torch.dot(input_embedding, saved_embedding)
        norm_input = torch.norm(input_embedding)
        norm_saved = torch.norm(saved_embedding)
        similarity = dot_product / (norm_input * norm_saved)  # Cosine similarity
        similarities.append(similarity.item())

    return similarities


if __name__ == "__main__":
    try:
        path1 = "Evaluation/Doc1/"
        clear_temp_folder()                                                 # refresh temp
        print('\n\n\n')
        documents_download(path1)
        print('\n\n\n')
        evaluate_similarity('./temp/input.txt')  # Make sure you call the correct path here
        print('\n\n\n')
        clear_temp_folder()                                                 # refresh temp
    except Exception as e:
        print("Error occurred during the entire process:")
        print(e)
        traceback.print_exc()  # This will print the full error traceback
