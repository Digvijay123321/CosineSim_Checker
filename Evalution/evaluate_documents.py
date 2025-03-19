import os
import json
import PyPDF2
from google.cloud import storage

def download_all_files_from_gcp(bucket_name, folder_name, local_folder):
    """Downloads all files from a GCP folder."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder_name)
    os.makedirs(local_folder, exist_ok=True)
    print("inside download function")
    
    # Debug: print the blob iterator
    blobs_list = list(blobs)  # Convert iterator to a list to inspect it
    # print(f"blobs_list: {blobs_list}")
    
    downloaded_files = []
    for blob in blobs_list:
        print("blob: ", blob)
        if not blob.name.endswith("/"):
            local_path = os.path.join(local_folder, os.path.basename(blob.name))
            blob.download_to_filename(local_path)
            downloaded_files.append(local_path)
            print(f"Downloaded {blob.name} to {local_path}.")
    
    return downloaded_files

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    with open(pdf_path, 'rb') as file:
        print("pdf path", pdf_path)
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def save_text_files(json_files, text_files, pdf_files, output_path):
    """Reads JSON, text, and PDF files, then saves their content into a single text file."""
    with open(output_path, 'w', encoding='utf-8') as file:
        for json_file in json_files:
            print("# of json files:", json_file)
            with open(json_file, 'r', encoding='utf-8') as jf:
                json_data = json.load(jf)
                if isinstance(json_data, dict):
                    print("Dict instance")
                    for agent, text in json_data.items():
                        file.write(f"{agent}: {text}\n\n")
                elif isinstance(json_data, list):
                    print("list instance")
                    for item in json_data:
                        if isinstance(item, dict):
                            for agent, text in item.items():
                                file.write(f"{agent}: {text}\n\n")
                else:
                    file.write("Invalid JSON format\n\n")
        
        for text_file in text_files:
            print("text file:", text_file)
            with open(text_file, 'r', encoding='utf-8') as tf:
                file.write(f"\nText File ({os.path.basename(text_file)}) Content:\n")
                file.write(tf.read())
        
        for pdf_file in pdf_files:
            print("pdf file:", pdf_file)
            pdf_text = extract_text_from_pdf(pdf_file)
            file.write(f"\nPDF File ({os.path.basename(pdf_file)}) Content:\n")
            file.write(pdf_text)

def main():
    print("\n\n\nStart of doc_collection...")
    # Ensure temp directory exists
    temp_dir = "./temp/"
    os.makedirs(temp_dir, exist_ok=True)
    print("started...")

    # GCP Bucket details
    gcp_bucket_name = "auditpulse-data"  # Update with your bucket name
    
    # Define GCP folder paths
    doc1_folder_pdf = "Evaluation/Doc1/"
    doc2_folder_pdf = "Evaluation/Doc2/"
    doc3_folder_json = "Evaluation/Doc3/"
    doc4_folder_txt = "Evaluation/Doc4/"
    print("decelered path...")
    
    # Download all files
    doc1_files = download_all_files_from_gcp(gcp_bucket_name, doc1_folder_pdf, temp_dir)
    doc2_files = download_all_files_from_gcp(gcp_bucket_name, doc2_folder_pdf, temp_dir)    # output
    doc3_files = download_all_files_from_gcp(gcp_bucket_name, doc3_folder_json, temp_dir)
    doc4_files = download_all_files_from_gcp(gcp_bucket_name, doc4_folder_txt, temp_dir)
    print("file name extracted...")
    print(doc1_files)
    print(doc2_files)
    print(doc3_files)
    print(doc4_files)

    # Categorize files
    pdf_files = [f for f in doc1_files if f.endswith(".pdf")]
    json_files = [f for f in doc3_files if f.endswith(".json")]
    text_files = [f for f in doc4_files if f.endswith(".txt")]
    print("categorized files...")
    print(pdf_files)
    print(json_files)
    print(text_files)

    # Process and save input.txt
    input_text_path = os.path.join(temp_dir, "input.txt")
    print("inside save_text_file function:")
    save_text_files(json_files, text_files, pdf_files, input_text_path)
    print("function executed")
    
    # Process a separate PDF into generated.txt
    generated_pdf_files = [f for f in doc2_files if f.endswith(".pdf")]
    if generated_pdf_files:
        generated_text_path = os.path.join(temp_dir, "generated.txt")
        pdf_text = extract_text_from_pdf(generated_pdf_files[0])  # Taking first PDF from doc2
        with open(generated_text_path, 'w', encoding='utf-8') as file:
            file.write(pdf_text)
    
    print("End of Doc collection...\n\n\n")
if __name__ == "__main__":
    main()
