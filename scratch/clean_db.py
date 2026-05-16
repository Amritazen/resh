import shutil
import os
import time

db_path = "chroma_db"
if os.path.exists(db_path):
    try:
        shutil.rmtree(db_path)
        print(f"Successfully deleted {db_path}")
    except Exception as e:
        print(f"Error deleting {db_path}: {e}")
else:
    print(f"{db_path} does not exist.")
