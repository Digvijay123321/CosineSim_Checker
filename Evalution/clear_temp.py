import os
import shutil

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

# Example usage
clear_temp_folder("./temp/")