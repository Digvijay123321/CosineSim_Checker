import os
import PyPDF2
import openai
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModel
import torch
from transformers import RobertaTokenizer, RobertaModel, AutoModelForMaskedLM
from datetime import datetime
import json
from google.cloud import firestore, storage

# openai.api_key = "api_key"  # Ensure API key is set
OUTPUT_DIR = "./Database/metrics/"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "comparisons.json")


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

def save_text_to_file(text, filename):
    """Save extracted text to a file."""
    if text:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(text)

def get_embedding(text):
    """Get OpenAI embedding for a given text."""
    response = openai.embeddings.create(
        model="text-embedding-ada-002", 
        input=[text]
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    # Flatten the embeddings to 1D arrays
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# T5 model evaluation function
def t5_similarity(text1, text2):
    """Evaluate similarity using T5 model."""
    print("t_5_sim")
    tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-small', legacy=False)
    print("tok1")
    model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
    print("model")
    
    inputs1 = tokenizer(text1, return_tensors="pt", padding=True)
    inputs2 = tokenizer(text2, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        embedding1 = model.encoder(inputs1.input_ids)[0].mean(dim=1)
        embedding2 = model.encoder(inputs2.input_ids)[0].mean(dim=1)
    
    similarity_score = cosine_similarity(embedding1.numpy(), embedding2.numpy())
    return similarity_score

# Sentence-BERT (SBERT) model evaluation function
def sbert_similarity(text1, text2):
    """Evaluate similarity using Sentence-BERT (SBERT)."""
    model_name = 'Muennighoff/SBERT-base-nli-v2'  # You can change the model to a different SBERT variant
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    inputs1 = tokenizer(text1, return_tensors="pt", truncation=True, padding=True)
    inputs2 = tokenizer(text2, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        embedding1 = model(**inputs1).last_hidden_state.mean(dim=1)
        embedding2 = model(**inputs2).last_hidden_state.mean(dim=1)
    
    similarity_score = cosine_similarity(embedding1.numpy(), embedding2.numpy())
    return similarity_score

# BERT model eval function
def bert_similarity(text1, text2):
    """Evaluate similarity using BERT (BERT)."""
    model_name = 'google-bert/bert-base-uncased'  # You can change the model to a different SBERT variant
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    inputs1 = tokenizer(text1, return_tensors="pt", truncation=True, padding=True)
    inputs2 = tokenizer(text2, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        embedding1 = model(**inputs1).last_hidden_state.mean(dim=1)
        embedding2 = model(**inputs2).last_hidden_state.mean(dim=1)
    
    similarity_score = cosine_similarity(embedding1.numpy(), embedding2.numpy())
    return similarity_score

# RoBERTa model evaluation function
def roberta_similarity(text1, text2):
    """Evaluate similarity using RoBERTa model."""
    # Load the tokenizer and model for RoBERTa
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')

    # Tokenize the input texts
    inputs1 = tokenizer(text1, return_tensors="pt", truncation=True, padding=True)
    inputs2 = tokenizer(text2, return_tensors="pt", truncation=True, padding=True)

    # Get the embeddings for the texts (mean pooling of last hidden state)
    with torch.no_grad():
        embedding1 = model(**inputs1).last_hidden_state.mean(dim=1)
        embedding2 = model(**inputs2).last_hidden_state.mean(dim=1)

    # Convert embeddings to numpy arrays for cosine similarity
    embedding1 = embedding1.numpy()
    embedding2 = embedding2.numpy()

    # Calculate cosine similarity
    similarity_score = cosine_similarity(embedding1, embedding2)
    if similarity_score.shape == (1, 1):
        return similarity_score[0][0]  # This returns the scalar value of similarity
    else:
        return similarity_score


# Modern Bert model evaluation function
def m_bert_similarity(text1, text2):
    """Evaluate similarity using Modern Bert model."""
    # Load the tokenizer and model for Modern Bert
    tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
    model = AutoModel.from_pretrained('answerdotai/ModernBERT-base')

    # Tokenize the input texts
    inputs1 = tokenizer(text1, return_tensors="pt")
    inputs2 = tokenizer(text2, return_tensors="pt")

    # Get the embeddings for the texts (mean pooling of last hidden state)
    with torch.no_grad():
        embedding1 = model(**inputs1).last_hidden_state.mean(dim=1)
        embedding2 = model(**inputs2).last_hidden_state.mean(dim=1)

    # Convert embeddings to numpy arrays for cosine similarity
    embedding1 = embedding1.numpy()
    embedding2 = embedding2.numpy()

    # Calculate cosine similarity
    similarity_score = cosine_similarity(embedding1, embedding2)
    if similarity_score.shape == (1, 1):
        return similarity_score[0][0]  # This returns the scalar value of similarity
    else:
        return similarity_score



def get_document(db_client, collection_name, document_name):
    """
    Retrieves a document from the Firestore database.

    Args:
        db_client (firestore.Client): Firestore database client instance.
        collection_name (str): The name of the Firestore collection. Default is 'config'.
        document_name (str): The name of the document to retrieve. Default is 'policy'.

    Returns:
        dict: The document data if found, else returns a default policy structure.
    """
    policy_collection = db_client.collection(collection_name).document(document_name)
    policy_document = policy_collection.get()
    if policy_document.exists:
        return policy_document.to_dict()
    else:
        raise ValueError('Policy document does not exist.')


def update_collection(db_client, collection_name, document_name, updated_collection):
    """
    Updates the Firestore policy document with the new version details.

    Args:
        db_client (firestore.Client): Firestore database client instance.
        collection_name (str): The Firestore collection where the policy is stored. Default is 'config'.
        document_name (str): The name of the Firestore document to update. Default is 'policy'.
    Returns:
        None
    """
    policy_collection = db_client.collection(collection_name).document(document_name)
    policy_collection.update(updated_collection)


def evaluate_similarity():
    """Evaluate similarity between two files"""
    doc1_path = "./temp/input.txt"
    doc2_path = "./temp/generated.txt"
    doc1_text = read_text_file(doc1_path)
    doc2_text = read_text_file(doc2_path)
    
    if not doc1_text or not doc2_text:
        print("Skipping similarity evaluation due to missing file/files")
        return
    
    save_text_to_file(doc1_text, os.path.join("temp", "input.txt"))
    save_text_to_file(doc2_text, os.path.join("temp", "generated.txt"))
    
    # Using OpenAI embedding -->2nd best
    # embedding1 = get_embedding(doc1_text)
    # embedding2 = get_embedding(doc2_text)
    # openai_similarity = cosine_similarity(embedding1, embedding2)
    # sudo test score
    print("open_simm")
    openai_similarity = 99999.99999
    
    
    # Using T5 model
    # t5_similarity_score = t5_similarity(doc1_text, doc2_text)
    t5_similarity_score = 99999.99999
    print("t5_simm")
    
    # Using SBERT model --> 1st best
    # sbert_similarity_score = sbert_similarity(doc1_text, doc2_text)
    sbert_similarity_score = 99999.99999
    
    
    # Using RoBERTa model
    # roberta_similarity_score = roberta_similarity(doc1_text, doc2_text)
    roberta_similarity_score = 99999.99999

    # Using BERT model
    # bert_similarity_score = bert_similarity(doc1_text, doc2_text)
    bert_similarity_score = 99999.99999
    
    # Using Modern BERT model
    # m_bert_similarity_score = m_bert_similarity(doc1_text, doc2_text)
    m_bert_similarity_score = 99999.99999


    print(f"OpenAI Similarity Score: {openai_similarity:.4f}")
    print(f"T5 Similarity Score: {t5_similarity_score:.4f}")
    print(f"SBERT Similarity Score: {sbert_similarity_score:.4f}")
    print(f"RoBERTa Similarity Score: {roberta_similarity_score:.4f}")
    print(f"BERT Similarity Score: {bert_similarity_score:.4f}")
    print(f"Modern BERT Similarity Score: {m_bert_similarity_score:.4f}")
    

    score = np.array([float(openai_similarity), 
            float(t5_similarity_score), 
            float(sbert_similarity_score), 
            float(roberta_similarity_score), 
            float(bert_similarity_score), 
            float(m_bert_similarity_score)])
    # save_comparison("input.txt", "generated.txt", score)
    db_client = firestore.Client(project='auditpulse')
    collection_name = 'config'
    document_name = 'run'
    policy_doc = get_document(db_client, collection_name, document_name)

    try:
        compare_file_1 = 'input.txt'
        compare_file_2 = 'generated.txt'
        latest_version = policy_doc.get('latest_dev_version')
        current_version = int(latest_version[3:]) + 1

        updated_collection = {
                            'active_dev_version': f'run{current_version}',
                            'latest_dev_version': f'run{current_version}',
                            f'development.run{current_version}': {
                                        'created_at': firestore.SERVER_TIMESTAMP,
                                        "Compaired_files": {
                                            "file_1": compare_file_1,
                                            "file_2": compare_file_2
                                            },
                                            "score": {
                                                "OpenAI score": score[0],
                                                "T5": score[1],
                                                "Sentence Bert": score[2],
                                                "RoBERTa": score[3],
                                                "Bert": score[4],
                                                "Modern BERT": score[5]
                                            },
                                                    }
                            }
        update_collection(db_client, collection_name, document_name, updated_collection)

    except Exception as e:
        print(e)
    


if __name__ == "__main__":
    evaluate_similarity()
