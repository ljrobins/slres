import os
import shutil

DIR = "/media/liam/Backup/data/slr/data"


def delete_dot_dirs(base_dir):
    """Recursively deletes directories containing a dot (.) in their name."""
    for root, dirs, files in os.walk(
        base_dir, topdown=False
    ):  # Bottom-up to remove empty parents
        for d in dirs:
            if "." in d:  # Check if directory contains '.'
                dir_path = os.path.join(root, d)
                print(f"Deleting: {dir_path}")
                enddddd
                shutil.rmtree(dir_path)  # Remove directory tree
        for f in files:
            fp = os.path.join(root, f)
            size = os.path.getsize(fp)
            if os.path.getsize(fp) == 0:
                print(f'{fp} is empty, removing')
                os.remove(fp)
        

# Run deletion
delete_dot_dirs(DIR)
print("Cleanup complete.")
