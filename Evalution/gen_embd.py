import torch
import numpy as np
import tempfile
from transformers import AutoTokenizer, AutoModel
from google.cloud import storage
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def generate_and_upload_embedding_largefile_streaming(
    file_path, 
    model_name, 
    gcp_bucket, 
    gcs_path,
    chunk_char_size=1000
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    embedding_sum = None
    chunk_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        buffer = ""
        for line in f:
            buffer += line
            if len(buffer) >= chunk_char_size:
                inputs = tokenizer(buffer, return_tensors="pt", truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze()
                embedding_sum = embedding if embedding_sum is None else embedding_sum + embedding
                chunk_count += 1
                buffer = ""

        if buffer:
            inputs = tokenizer(buffer, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze()
            embedding_sum = embedding if embedding_sum is None else embedding_sum + embedding
            chunk_count += 1

    final_embedding = (embedding_sum / chunk_count).numpy()

    with tempfile.NamedTemporaryFile(suffix=".npy") as temp_file:
        np.save(temp_file.name, final_embedding)
        client = storage.Client()
        bucket = client.bucket(gcp_bucket)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(temp_file.name)

    print(f"✅ Final embedding uploaded to gs://{gcp_bucket}/{gcs_path} from {chunk_count} chunks.")

def generate_and_upload_bert_embedding_largefile(file_path):
    generate_and_upload_embedding_largefile_streaming(
        file_path=file_path,
        model_name='google-bert/bert-base-uncased',
        gcp_bucket='auditpulse-data',
        gcs_path='Evaluation/Doc3/bert_embd-large.npy'
    )

def generate_and_upload_modernbert_embedding_largefile(file_path):
    generate_and_upload_embedding_largefile_streaming(
        file_path=file_path,
        model_name='answerdotai/ModernBERT-base',
        gcp_bucket='auditpulse-data',
        gcs_path='Evaluation/Doc3/modernbert_embd-large.npy'
    )

def generate_and_upload_roberta_embedding_largefile(file_path):
    generate_and_upload_embedding_largefile_streaming(
        file_path=file_path,
        model_name='roberta-base',
        gcp_bucket='auditpulse-data',
        gcs_path='Evaluation/Doc3/roberta_embd-large.npy'
    )

def generate_and_upload_sbert_embedding_largefile(file_path):
    generate_and_upload_embedding_largefile_streaming(
        file_path=file_path,
        model_name='Muennighoff/SBERT-base-nli-v2',
        gcp_bucket='auditpulse-data',
        gcs_path='Evaluation/Doc3/sbert_embd-large.npy'
    )

if __name__ == "__main__":
    generate_and_upload_bert_embedding_largefile('./temp/input.txt')
    generate_and_upload_modernbert_embedding_largefile('./temp/input.txt')
    generate_and_upload_roberta_embedding_largefile('./temp/input.txt')
    generate_and_upload_sbert_embedding_largefile('./temp/input.txt')
