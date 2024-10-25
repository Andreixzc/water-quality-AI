import os

def list_files_with_extensions():
    root_folder = os.getcwd()  # Get the current working directory
    for foldername, _, filenames in os.walk(root_folder):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            file_extension = os.path.splitext(filename)[1] or "No extension"
            print(f"File: {file_path} - Extension: {file_extension}")

# Run the function
list_files_with_extensions()
