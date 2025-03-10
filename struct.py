import os

def save_directory_structure(root_folder, output_file, ignore_folders=None):
    if ignore_folders is None:
        ignore_folders = []

    with open(output_file, "w", encoding="utf-8") as f:
        for dirpath, dirnames, filenames in os.walk(root_folder):
            # Remove ignored folders from traversal
            dirnames[:] = [d for d in dirnames if d not in ignore_folders]

            level = dirpath.replace(root_folder, "").count(os.sep)
            indent = "    " * level
            f.write(f"{indent}d----- {os.path.basename(dirpath)}\n")
            
            sub_indent = "    " * (level + 1)
            for filename in filenames:
                f.write(f"{sub_indent}-a---- {filename}\n")

# Set root folder and output file
root_folder = os.getcwd()  # Current directory
output_file = "directory_structure.txt"
ignore_folders = ["myenv", ".git"]  # Folders to ignore

save_directory_structure(root_folder, output_file, ignore_folders)
print(f"Directory structure saved to {output_file}, ignoring {ignore_folders}.")
