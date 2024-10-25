import os

def list_files_in_root():
    root_folder = os.path.dirname(os.path.abspath(__file__))
    for filename in os.listdir(root_folder):
        if os.path.isfile(os.path.join(root_folder, filename)):
            print(filename)

if __name__ == "__main__":
    list_files_in_root()
