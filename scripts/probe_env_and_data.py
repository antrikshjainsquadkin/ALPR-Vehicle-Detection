import os
import sys
from pathlib import Path

def probe_environment():
    print("=== ENVIRONMENT PROBE ===", flush=True)
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}", flush=True)
        cuda_avail = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_avail}", flush=True)
        if cuda_avail:
            print(f"Device Name: {torch.cuda.get_device_name(0)}", flush=True)
            print(f"Device Count: {torch.cuda.device_count()}", flush=True)
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU Memory: {total_mem:.2f} GB", flush=True)
        else:
            print("Running on CPU.", flush=True)
    except Exception as e:
        print(f"PyTorch probe error: {e}", flush=True)

    try:
        import ultralytics
        print(f"Ultralytics Version: {ultralytics.__version__}", flush=True)
    except Exception as e:
        print(f"Ultralytics probe error: {e}", flush=True)

def probe_local_dirs():
    print("\n=== LOCAL DIRECTORY LISTING ===", flush=True)
    root_dir = Path(r"c:\Users\A\OneDrive\Desktop\vehicle_detect")
    for entry in os.scandir(root_dir):
        if entry.is_dir() and not entry.name.startswith("."):
            print(f"  [DIR] {entry.name}", flush=True)
        elif entry.is_file():
            print(f"  [FILE] {entry.name} ({entry.stat().st_size} bytes)", flush=True)

    brain_uploads = Path(r"C:\Users\A\.gemini\antigravity\brain\dae2382e-fd39-4f64-9bf4-7f3f1f8cf1be\.user_uploaded")
    if brain_uploads.exists():
        print("\nUser uploaded images in session:", flush=True)
        for u in os.scandir(brain_uploads):
            print(f"  {u.name} ({u.stat().st_size} bytes)", flush=True)

if __name__ == "__main__":
    probe_environment()
    probe_local_dirs()
