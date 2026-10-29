import os
import hashlib
from pathlib import Path
from collections import defaultdict

def audit_duplicates():
    train_dir = Path(r"C:\Users\A\Downloads\train")
    hash_to_files = defaultdict(list)
    
    for f in train_dir.glob("*.jpg"):
        h = hashlib.md5(f.read_bytes()).hexdigest()
        hash_to_files[h].append(f.name)

    duplicates = {h: files for h, files in hash_to_files.items() if len(files) > 1}
    print(f"Total files: {sum(len(files) for files in hash_to_files.values())}")
    print(f"Unique images: {len(hash_to_files)}")
    print(f"Duplicate groups: {len(duplicates)}")
    
    total_duplicate_copies = sum(len(files) - 1 for files in duplicates.values())
    print(f"Total duplicate copies: {total_duplicate_copies}")
    
    print("\nSample duplicate groups (first 5):")
    for i, (h, files) in enumerate(list(duplicates.items())[:5]):
        print(f"  Group {i+1}: {files}")

if __name__ == "__main__":
    audit_duplicates()

