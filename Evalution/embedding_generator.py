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

        elif file_path.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content += f.read() + "\n"

        elif file_path.endswith(".pdf"):
            text_content += extract_text_from_pdf(file_path) + "\n"

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


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])


def generate_embeddings(company_name, path1):
    """Generate embeddings for files in given path"""

    print("Generating embeddings:")
    
    doc1_path = "./temp/input.txt"
    
    doc1_text = read_text_file(doc1_path)
    
    if not doc1_text:
        print("Skipping embedding due to missing file/files")
        return
    
    sbert_embedding_status = sbert_embedding(doc1_text)
    mbert_embedding_status = mbert_embedding(doc1_text)
    bert_embedding_status = bert_embedding(doc1_text)
    roberta_embedding_status = roberta_embedding(doc1_text)

    print(f"SBERT Embedding Status: {sbert_embedding_status}")
    print(f"MBERT Embedding Status: {mbert_embedding_status}")
    print(f"MBERT Embedding Status: {bert_embedding_status}")
    print(f"MBERT Embedding Status: {roberta_embedding_status}")

    try:
        upload_all_embeddings_to_gcs()

    except Exception as e:
        print(e)

def upload_all_embeddings_to_gcs(local_dir='./embedding/', gcp_bucket_name='auditpulse-data', gcs_folder='Evaluation/Doc2/'):
    # Initialize GCS client
    client = storage.Client()
    bucket = client.bucket(gcp_bucket_name)

    # Check if local_dir exists
    if not os.path.exists(local_dir):
        print(f"Directory {local_dir} does not exist.")
        return

    # Walk through all files in local_dir
    for filename in os.listdir(local_dir):
        local_path = os.path.join(local_dir, filename)
        
        if os.path.isfile(local_path):
            # Construct GCS path
            gcs_path = os.path.join(gcs_folder, filename)
            blob = bucket.blob(gcs_path)
            
            # Upload the file
            blob.upload_from_filename(local_path)
            print(f"Uploaded {filename} to gs://{gcp_bucket_name}/{gcs_path}")

    print("✅ All files uploaded successfully.")


def read_text_file(file_name):
    """Reads a text file with the specified filename and returns its content."""
    text = ""
    try:
        with open(file_name, "r", encoding="utf-8") as file:
            text = file.read()
    except FileNotFoundError:
        print(f"Error: The file {file_name} was not found.")
        return None
    except IOError:
        print(f"Error: Unable to read {file_name}.")
        return None
    return text

def t5_similarity(text1, text2):
    """Evaluate similarity using T5 model."""
    print("inside t5")
    tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-small', legacy=False)
    model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
    
    inputs1 = tokenizer(text1, return_tensors="pt", padding=True)
    inputs2 = tokenizer(text2, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        embedding1 = model.encoder(inputs1.input_ids)[0].mean(dim=1)
        embedding2 = model.encoder(inputs2.input_ids)[0].mean(dim=1)
    
    similarity_score = cosine_similarity(embedding1.numpy(), embedding2.numpy())
    return similarity_score

def chunk_text(text, tokenizer, max_length=512, stride=256):
    """Splits text into overlapping chunks."""
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    chunks = [tokens[i: i + max_length] for i in range(0, len(tokens), stride)]
    
    # Ensure no chunk exceeds max_length
    chunks = [chunk[:max_length] for chunk in chunks]  # Truncate each chunk to max_length if necessary
    return chunks

def get_embedding_test(text, model, tokenizer, max_length=512, stride=256):
    """Computes embeddings for long texts using chunking."""
    print('start embidding')
    chunks = chunk_text(text, tokenizer, max_length, stride)
    embeddings = []
    
    with torch.no_grad():
        for chunk in chunks:
            # Convert chunk to tensor and ensure it's of the right shape
            chunk_tensor = torch.tensor(chunk).unsqueeze(0).clone().detach()  # Add batch dimension
            chunk_tensor = chunk_tensor[:, :max_length]  # Ensure it doesn't exceed max_length
            
            # Forward pass through the model
            outputs = model.encoder(chunk_tensor)
            embedding = outputs.last_hidden_state.mean(dim=1)  # Take the mean of the hidden states
            embeddings.append(embedding)

def get_embedding(text, model="text-embedding-ada-002"):
    """Gets text embeddings using OpenAI API."""
    client = openai.OpenAI()
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return np.array(response.data[0].embedding)

def sbert_embedding(text1):
    """Evaluate similarity using Sentence-BERT (SBERT) with chunking."""
    model_name = 'Muennighoff/SBERT-base-nli-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize without truncation to check token count
    tokenized = tokenizer(text1, return_tensors="pt", truncation=False, padding=False)
    token_count = tokenized["input_ids"].shape[1]

    print(f"Token count: {token_count}")

    if token_count > 512:
        print("⚠️ Text is too long, using chunking...")
        chunks = chunk_text(text1, tokenizer, max_length=512, stride=256)
        embeddings = []

        with torch.no_grad():
            for chunk in chunks:
                chunk_tensor = torch.tensor(chunk).unsqueeze(0)
                output = model(chunk_tensor)
                chunk_embedding = output.last_hidden_state.mean(dim=1)
                embeddings.append(chunk_embedding)

        # Combine all chunk embeddings (average)
        final_embedding = torch.stack(embeddings).mean(dim=0)
    else:
        print("✅ Text is short, processing normally.")
        inputs1 = tokenizer(text1, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            final_embedding = model(**inputs1).last_hidden_state.mean(dim=1)

    os.makedirs('./embedding/', exist_ok=True)
    torch.save(final_embedding.cpu().detach(), './embedding/10k_sbert_embedding.pt')
    
    return True

# Modern Bert model evaluation function
def mbert_embedding(text1):
    """Evaluate similarity using Modern Bert model."""
    # Load the tokenizer and model for Modern Bert
    tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
    model = AutoModel.from_pretrained('answerdotai/ModernBERT-base')

    # Tokenize without truncation to check token count
    tokenized = tokenizer(text1, return_tensors="pt", truncation=False, padding=False)
    token_count = tokenized["input_ids"].shape[1]

    print(f"Token count: {token_count}")

    if token_count > 512:
        print("⚠️ Text is too long, using chunking...")
        chunks = chunk_text(text1, tokenizer, max_length=512, stride=256)
        embeddings = []

        with torch.no_grad():
            for chunk in chunks:
                chunk_tensor = torch.tensor(chunk).unsqueeze(0)
                output = model(chunk_tensor)
                chunk_embedding = output.last_hidden_state.mean(dim=1)
                embeddings.append(chunk_embedding)

        # Average across chunk embeddings
        final_embedding = torch.stack(embeddings).mean(dim=0)
    else:
        print("✅ Text is short, processing normally.")
        inputs1 = tokenizer(text1, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            final_embedding = model(**inputs1).last_hidden_state.mean(dim=1)

    os.makedirs('./embedding/', exist_ok=True)
    torch.save(final_embedding.cpu().detach(), './embedding/10k_mbert_embedding.pt')

    return True

def bert_embedding(text1):
    """Evaluate similarity using BERT with chunking and saving."""
    model_name = 'google-bert/bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize without truncation to check token count
    tokenized = tokenizer(text1, return_tensors="pt", truncation=False, padding=False)
    token_count = tokenized["input_ids"].shape[1]

    print(f"Token count: {token_count}")

    if token_count > 512:
        print("⚠️ Text is too long, using chunking...")
        chunks = chunk_text(text1, tokenizer, max_length=512, stride=256)
        embeddings = []

        with torch.no_grad():
            for chunk in chunks:
                chunk_tensor = torch.tensor(chunk).unsqueeze(0)
                output = model(chunk_tensor)
                chunk_embedding = output.last_hidden_state.mean(dim=1)
                embeddings.append(chunk_embedding)

        final_embedding = torch.stack(embeddings).mean(dim=0)
    else:
        print("✅ Text is short, processing normally.")
        inputs1 = tokenizer(text1, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            final_embedding = model(**inputs1).last_hidden_state.mean(dim=1)

    os.makedirs('./embedding/', exist_ok=True)
    torch.save(final_embedding.cpu().detach(), './embedding/10k_bert_embedding.pt')

    return True

def roberta_embedding(text1):
    """Evaluate similarity using RoBERTa with chunking and saving."""
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')

    # Tokenize without truncation
    tokenized = tokenizer(text1, return_tensors="pt", truncation=False, padding=False)
    token_count = tokenized["input_ids"].shape[1]

    print(f"Token count: {token_count}")

    if token_count > 512:
        print("⚠️ Text is too long, using chunking...")
        chunks = chunk_text(text1, tokenizer, max_length=512, stride=256)
        embeddings = []

        with torch.no_grad():
            for chunk in chunks:
                chunk_tensor = torch.tensor(chunk).unsqueeze(0)
                output = model(chunk_tensor)
                chunk_embedding = output.last_hidden_state.mean(dim=1)
                embeddings.append(chunk_embedding)

        final_embedding = torch.stack(embeddings).mean(dim=0)
    else:
        print("✅ Text is short, processing normally.")
        inputs1 = tokenizer(text1, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            final_embedding = model(**inputs1).last_hidden_state.mean(dim=1)

    os.makedirs('./embedding/', exist_ok=True)
    torch.save(final_embedding.cpu().detach(), './embedding/10k_roberta_embedding.pt')

    return True

if __name__ == "__main__":
    try:
        company_name = "sudo"
        path1 = "Evaluation/Doc1/"
        path2 = "Evaluation/Doc2/"
        clear_temp_folder()                                                 # refresh temp
        print('\n\n\n')
        documents_download(path1)
        print('\n\n\n')
        generate_embeddings(company_name, path1)
        print('\n\n\n')
        clear_temp_folder()                                                 # refresh temp
    except:
        print("Error")
