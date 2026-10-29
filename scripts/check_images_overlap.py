import os
import hashlib
from pathlib import Path
from PIL import Image

def get_hash(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def check_overlap():
    train_dir = Path(r"C:\Users\A\Downloads\train")
    brain_uploads = Path(r"C:\Users\A\.gemini\antigravity\brain\dae2382e-fd39-4f64-9bf4-7f3f1f8cf1be\.user_uploaded")
    
    upload_hashes = {}
    for u in brain_uploads.glob("*.jpg"):
        upload_hashes[get_hash(u)] = u.name

    print(f"User uploads ({len(upload_hashes)}):")
    for h, name in upload_hashes.items():
        print(f"  {name}: {h}")

    train_hashes = {}
    matches = []
    for f in train_dir.glob("*.jpg"):
        f_hash = get_hash(f)
        train_hashes[f_hash] = f.name
        if f_hash in upload_hashes:
            matches.append((f.name, upload_hashes[f_hash]))

    print(f"\nTotal files in Downloads/train: {len(train_hashes)}")
    print(f"Matches with session uploads: {matches}")

    # Inspect filenames in train_dir
    filenames = sorted([int(f.stem) for f in train_dir.glob("*.jpg") if f.stem.isdigit()])
    print(f"Number of numeric filenames: {len(filenames)} (min: {min(filenames)}, max: {max(filenames)})")

if __name__ == "__main__":
    check_overlap()

