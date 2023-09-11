import os

def create_path(path):
    sub_folder_path = []
    files_in_folder = []
    all_files_path = []
    all_folder = os.listdir(path)

    for f in all_folder:
        sub_folder_path.append(os.path.join(path, f))

    for j in range(len(sub_folder_path)):
        for i in os.listdir(sub_folder_path[j]):
            files_path = os.path.join(sub_folder_path[j], i)
            files_in_folder.append(files_path)
        
        all_files_path.append(files_in_folder)
        files_in_folder = []

    return all_files_path