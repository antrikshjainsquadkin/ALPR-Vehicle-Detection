from pathlib import Path

pseudo_dir = Path("datasets/two_wheeler/pseudo_labels_round2")
txt_files = list(pseudo_dir.glob("*.txt"))
non_empty = [f for f in txt_files if f.stat().st_size > 0]
print(f"Total txt files: {len(txt_files)}")
print(f"Non-empty label files: {len(non_empty)}")

# Read lines in first 10 non-empty
for f in non_empty[:10]:
    lines = f.read_text().strip().split("\n")
    print(f"  {f.name}: {len(lines)} boxes")

