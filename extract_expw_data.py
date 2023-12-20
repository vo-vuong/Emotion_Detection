import os
from utils.file_helpers import get_root_foler
import py7zr


root_folder = get_root_foler()
archive_raw_data_path = os.path.join(
    root_folder, "data", "raw", "image", "origin.7z")
extract_data_path = os.path.join(root_folder, "data", "processed", "image")

combined_file_path = f'{archive_raw_data_path}.combined'
with open(combined_file_path, 'wb') as combined_file:
    part_number = 1
    while True:
        part_file_path = f'{archive_raw_data_path}.{str(part_number).zfill(3)}'
        try:
            with open(part_file_path, 'rb') as part_file:
                combined_file.write(part_file.read())
            part_number += 1
        except FileNotFoundError:
            break

with py7zr.SevenZipFile(combined_file_path, mode='r') as z:
    z.extractall(path=extract_data_path)
