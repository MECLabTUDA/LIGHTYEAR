import os
import shutil
import random

def split_test_into_val(test_dir, val_fraction=1/3, seed=42):
    """
    Moves a fraction of files from the test directory to a new validation directory.

    Args:
        test_dir (str): Path to the existing test directory.
        val_fraction (float): Fraction of files to move to validation.
        seed (int): Seed for randomization.
    """
    val_dir = os.path.join(os.path.dirname(test_dir), 'val')
    os.makedirs(val_dir, exist_ok=True)
    
    # List all files in the test directory
    files = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
    random.seed(seed)
    random.shuffle(files)

    split_idx = int(len(files) * val_fraction)
    val_files = files[:split_idx]

    for f in val_files:
        src_path = os.path.join(test_dir, f)
        dst_path = os.path.join(val_dir, f)
        shutil.move(src_path, dst_path)

    print(f"Moved {len(val_files)} files to '{val_dir}'. Remaining in '{test_dir}': {len(files) - len(val_files)}.")

if __name__ == '__main__':
    path = 'clients'
    clients = os.listdir(path)
    for client in clients:
        split_test_into_val(os.path.join(path, client, 'test'))
