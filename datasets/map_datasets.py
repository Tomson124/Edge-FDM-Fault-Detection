import os
import csv
import shutil

datasets = ['3D_fault_dataset', 'combined_ds', 'reduced_ds', 'Printing_Dataset']

def curate_dataset(dataset_name):
    script_dir = os.path.dirname(__file__)
    mapping_file = os.path.join(script_dir, 'mappings', f'{dataset_name}.csv')
    dest_base_dir = os.path.join(script_dir, dataset_name)

    with open(mapping_file, newline='', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        os.makedirs(dest_base_dir, exist_ok=True)
        os.makedirs(os.path.join(dest_base_dir, 'failure'), exist_ok=True)
        os.makedirs(os.path.join(dest_base_dir, 'success'), exist_ok=True)
        for row in reader:
            name = row.get('filename')
            label = row.get('label')
            relative_src_path = row.get('path')
            full_src_path = os.path.join(script_dir, relative_src_path)
            full_dest_path = os.path.join(dest_base_dir, label, name)

            if label in ['failure', 'success']:
                try:
                    shutil.copy(full_src_path, full_dest_path)
                except FileNotFoundError:
                    print(f"Source file not found: {full_src_path}")
                except Exception as e:
                    print(f"Error copying {full_src_path} to {full_dest_path}: {e}")
            else:
                print(f"Unknown label {label} for file {name}")

if __name__ == "__main__":
    for dataset in datasets:
        print(f"Curating dataset: {dataset}")
        curate_dataset(dataset)
        print(f"Finished curating dataset: {dataset}")

