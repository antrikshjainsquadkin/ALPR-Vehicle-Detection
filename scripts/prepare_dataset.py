"""
Dataset Ingestion and Deduplication Script for Two-Wheeler Detection.
Preserves original source images without modification.
Structures canonical unique images into datasets/two_wheeler/raw_images/.
Generates a structured duplicates report.
"""

import os
import sys
import shutil
import hashlib
import json
from pathlib import Path
from collections import defaultdict
from PIL import Image

def compute_md5(file_path: Path) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def prepare_two_wheeler_dataset(
    source_dir: Path,
    target_base_dir: Path,
    additional_images_dir: Path = None
) -> dict:
    source_dir = Path(source_dir)
    target_base_dir = Path(target_base_dir)
    raw_images_dir = target_base_dir / "raw_images"
    raw_images_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Dataset Prep] Source directory: {source_dir}")
    print(f"[Dataset Prep] Target raw images directory: {raw_images_dir}")

    # Gather all images from source
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    source_files = [f for f in source_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
    source_files.sort(key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem)
    print(f"[Dataset Prep] Found {len(source_files)} images in source directory.")

    # Check additional session uploads if provided
    additional_files = []
    if additional_images_dir and Path(additional_images_dir).exists():
        additional_files = [f for f in Path(additional_images_dir).iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
        print(f"[Dataset Prep] Found {len(additional_files)} additional images in session uploads.")

    all_input_files = source_files + additional_files
    print(f"[Dataset Prep] Total candidates to inspect: {len(all_input_files)}")

    hash_to_files = defaultdict(list)
    file_metadata = {}

    for file_path in all_input_files:
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                img_format = img.format
            file_hash = compute_md5(file_path)
            file_size = file_path.stat().st_size
            hash_to_files[file_hash].append(file_path)
            file_metadata[str(file_path)] = {
                "name": file_path.name,
                "size_bytes": file_size,
                "dimensions": [width, height],
                "format": img_format,
                "hash": file_hash
            }
        except Exception as e:
            print(f"[Dataset Prep] WARNING: Skipping unreadable file {file_path.name}: {e}")

    # Classify duplicates and choose canonical files
    canonical_files = []
    duplicate_groups = {}
    duplicate_count = 0

    for file_hash, files in hash_to_files.items():
        # Select canonical file: prefer source files, lowest numeric stem if numeric
        def sort_key(p: Path):
            return (0 if p in source_files else 1, int(p.stem) if p.stem.isdigit() else 999999, str(p))
        sorted_files = sorted(files, key=sort_key)
        canonical = sorted_files[0]
        canonical_files.append(canonical)

        if len(sorted_files) > 1:
            duplicate_groups[file_hash] = {
                "canonical": canonical.name,
                "canonical_path": str(canonical),
                "duplicates": [str(d) for d in sorted_files[1:]],
                "duplicate_names": [d.name for d in sorted_files[1:]],
                "dimensions": file_metadata[str(canonical)]["dimensions"],
                "size_bytes": file_metadata[str(canonical)]["size_bytes"]
            }
            duplicate_count += len(sorted_files) - 1

    print(f"[Dataset Prep] Total unique images: {len(canonical_files)}")
    print(f"[Dataset Prep] Duplicate groups: {len(duplicate_groups)} (Total duplicate copies: {duplicate_count})")

    # Copy canonical images to target raw_images directory without overwriting source
    copied_count = 0
    canonical_manifest = []

    for src_path in canonical_files:
        # Determine standardized target filename
        target_name = src_path.name
        dest_path = raw_images_dir / target_name
        
        # Avoid collision if multiple sources had different paths but same name
        if dest_path.exists() and compute_md5(dest_path) != file_metadata[str(src_path)]["hash"]:
            target_name = f"{src_path.stem}_{src_path.parent.name}{src_path.suffix}"
            dest_path = raw_images_dir / target_name

        if not dest_path.exists():
            shutil.copy2(src_path, dest_path)
            copied_count += 1

        canonical_manifest.append({
            "target_filename": target_name,
            "source_path": str(src_path),
            "hash": file_metadata[str(src_path)]["hash"],
            "dimensions": file_metadata[str(src_path)]["dimensions"],
            "size_bytes": file_metadata[str(src_path)]["size_bytes"],
            "has_duplicates": file_metadata[str(src_path)]["hash"] in duplicate_groups
        })

    print(f"[Dataset Prep] Canonical images populated: {len(canonical_manifest)} ({copied_count} newly copied).")

    # Save structured audit report
    report = {
        "source_directories": [str(source_dir)] + ([str(additional_images_dir)] if additional_images_dir else []),
        "target_directory": str(raw_images_dir),
        "total_source_files_inspected": len(all_input_files),
        "total_unique_images": len(canonical_files),
        "total_duplicate_copies_detected": duplicate_count,
        "duplicate_groups_count": len(duplicate_groups),
        "duplicate_groups": duplicate_groups,
        "canonical_manifest": canonical_manifest
    }

    report_path = target_base_dir / "duplicates_and_dataset_audit_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[Dataset Prep] Audit report saved to {report_path}")

    # Generate Markdown Summary
    md_summary_path = target_base_dir / "DATASET_AUDIT_SUMMARY.md"
    with open(md_summary_path, "w", encoding="utf-8") as f:
        f.write("# Two-Wheeler Dataset Audit & Ingestion Summary\n\n")
        f.write(f"- **Total Candidate Files Inspected:** {len(all_input_files)}\n")
        f.write(f"- **Unique Clean Images Ingested:** {len(canonical_files)}\n")
        f.write(f"- **Duplicate Files Removed From Ingestion:** {duplicate_count}\n")
        f.write(f"- **Duplicate Groups:** {len(duplicate_groups)}\n")
        f.write(f"- **Source Integrity:** 100% preserved (all original source files remain intact in their original folders).\n")
        f.write(f"- **Target Directory:** `{raw_images_dir}`\n\n")
        f.write("## Sample Duplicate Mappings\n\n")
        f.write("| Canonical Filename | Duplicate Copies | Resolution | Size (bytes) |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for i, (h, group) in enumerate(list(duplicate_groups.items())[:15]):
            dups = ", ".join(group["duplicate_names"])
            dims = f"{group['dimensions'][0]}x{group['dimensions'][1]}"
            f.write(f"| `{group['canonical']}` | {dups} | {dims} | {group['size_bytes']} |\n")
        if len(duplicate_groups) > 15:
            f.write(f"| ... | *and {len(duplicate_groups) - 15} more groups* | ... | ... |\n")

    print(f"[Dataset Prep] Markdown summary saved to {md_summary_path}")
    return report

if __name__ == "__main__":
    src = Path(r"C:\Users\A\Downloads\train")
    target = Path(r"c:\Users\A\OneDrive\Desktop\vehicle_detect\datasets\two_wheeler")
    uploads = Path(r"C:\Users\A\.gemini\antigravity\brain\dae2382e-fd39-4f64-9bf4-7f3f1f8cf1be\.user_uploaded")
    prepare_two_wheeler_dataset(src, target, uploads)

