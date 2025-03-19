from google.cloud import storage

def upload_json_to_gcp(local_json_path, bucket_name, gcp_path):
    """Uploads a local JSON file to GCP bucket under the specified path."""
    # Initialize a GCP storage client
    storage_client = storage.Client()
    
    # Get the GCP bucket
    bucket = storage_client.bucket(bucket_name)
    
    # Create a blob (GCP object) for the file path
    blob = bucket.blob(gcp_path)
    
    try:
        # Upload the local JSON file to GCP
        blob.upload_from_filename(local_json_path)
        print(f"Successfully uploaded {local_json_path} to gs://{bucket_name}/{gcp_path}")
    except Exception as e:
        print(f"Error uploading file: {e}")

if __name__ == "__main__":
    local_json_file = './Database/metrics/comparisons.json'
    bucket_name = 'auditpulse-data'  # Your GCP bucket name
    gcp_target_path = 'Evaluation/result/comparison.json'
    upload_json_to_gcp(local_json_file, bucket_name, gcp_target_path)