from google.cloud import firestore, storage

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



def main():
    db_client = firestore.Client(project='auditpulse')
    collection_name = 'config'
    document_name = 'run'
    policy_doc = get_document(db_client, collection_name, document_name)

    try:
        latest_version = policy_doc.get('latest_dev_version')
        current_version = int(latest_version[3:]) + 1

        updated_collection = {
                            'active_dev_version': f'run{current_version}',
                            'latest_dev_version': f'run{current_version}',
                            f'version.run{current_version}': {
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
if __name__ == '__main__':
    main()
