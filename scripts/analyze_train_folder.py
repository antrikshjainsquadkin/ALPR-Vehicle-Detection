import os
from pathlib import Path
from collections import Counter
from PIL import Image

def analyze_folder(folder_path):
    p = Path(folder_path)
    print(f"=== ANALYZING: {p} ===")
    if not p.exists():
        print("Folder does not exist!")
        return

    all_entries = list(p.iterdir())
    print(f"Total entries in root: {len(all_entries)}")
    
    subdirs = [e for e in all_entries if e.is_dir()]
    files = [e for e in all_entries if e.is_file()]
    print(f"Subdirectories: {[d.name for d in subdirs]}")
    
    exts = Counter(e.suffix.lower() for e in files)
    print(f"File extensions: {dict(exts)}")

    # Check for label formats
    txt_files = [f for f in files if f.suffix.lower() == '.txt']
    xml_files = [f for f in files if f.suffix.lower() == '.xml']
    json_files = [f for f in files if f.suffix.lower() == '.json']
    yaml_files = [f for f in files if f.suffix.lower() in ('.yaml', '.yml')]
    
    print(f"TXT files: {len(txt_files)}")
    print(f"XML files: {len(xml_files)}")
    print(f"JSON files: {len(json_files)}")
    print(f"YAML files: {len(yaml_files)}")

    if yaml_files:
        for y in yaml_files:
            print(f"--- YAML: {y.name} ---")
            print(y.read_text(encoding='utf-8', errors='ignore')[:500])

    if json_files:
        for j in json_files:
            print(f"JSON file: {j.name} ({j.stat().st_size} bytes)")

    if txt_files:
        print(f"Sample TXT file ({txt_files[0].name}):")
        print(txt_files[0].read_text(encoding='utf-8', errors='ignore')[:300])

    # Sample images analysis
    img_files = [f for f in files if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')]
    print(f"Total image files: {len(img_files)}")
    
    resolutions = []
    corrupt_count = 0
    for img_p in img_files[:100]:  # check first 100 for speed
        try:
            with Image.open(img_p) as img:
                resolutions.append(img.size)
        except Exception:
            corrupt_count += 1

    if resolutions:
        res_counter = Counter(resolutions)
        print(f"Common resolutions (sampled {len(resolutions)}): {res_counter.most_common(5)}")
    print(f"Corrupt count (sample): {corrupt_count}")

    # Check if subdirectories have labels/images
    for d in subdirs:
        sub_files = list(d.iterdir())
        sub_exts = Counter(e.suffix.lower() for e in sub_files if e.is_file())
        print(f"  Subdir '{d.name}': {len(sub_files)} items, exts: {dict(sub_exts)}")

if __name__ == "__main__":
    analyze_folder(r"C:\Users\A\Downloads\train")

