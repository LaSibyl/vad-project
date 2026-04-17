#!/usr/bin/env python
import os
import urllib.request
import tarfile
import zipfile

def download_file(url, out_path):
    """Download file with progress"""
    if os.path.exists(out_path):
        print(f"✓ Already exists: {out_path}")
        return
    print(f"⏳ Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, out_path)
        print(f"✓ Saved to: {out_path}")
    except Exception as e:
        print(f"✗ Failed to download {url}: {e}")

def extract_file(path):
    """Extract tar.gz or zip files"""
    if not os.path.exists(path):
        print(f"✗ File not found: {path}")
        return
    
    if path.endswith('.tar.gz'):
        outdir = os.path.splitext(os.path.splitext(path)[0])[0]
        if os.path.exists(outdir):
            print(f"✓ Already extracted: {outdir}")
            return
        print(f"⏳ Extracting {path}...")
        with tarfile.open(path, "r:gz") as f:
            f.extractall(os.path.dirname(path))
        print(f"✓ Extracted to: {outdir}")
    
    elif path.endswith('.zip'):
        outdir = os.path.splitext(path)[0]
        if os.path.exists(outdir):
            print(f"✓ Already extracted: {outdir}")
            return
        print(f"⏳ Extracting {path}...")
        with zipfile.ZipFile(path, "r") as f:
            f.extractall(os.path.dirname(path))
        print(f"✓ Extracted to: {outdir}")

if __name__ == "__main__":
    os.makedirs(r"data\raw", exist_ok=True)
    
    print("=" * 60)
    print("DOWNLOADING DATASETS")
    print("=" * 60)
    
    # LibriSpeech mini
    print("\n[1/4] Mini LibriSpeech - train-clean-5")
    download_file(
        "https://www.openslr.org/resources/31/train-clean-5.tar.gz",
        r"data\raw\train-clean-5.tar.gz"
    )
    
    print("\n[2/4] Mini LibriSpeech - dev-clean-2")
    download_file(
        "https://www.openslr.org/resources/31/dev-clean-2.tar.gz",
        r"data\raw\dev-clean-2.tar.gz"
    )
    
    # MUSAN
    print("\n[3/4] MUSAN (noise dataset)")
    download_file(
        "https://www.openslr.org/resources/17/musan.tar.gz",
        r"data\raw\musan.tar.gz"
    )
    
    # RIRS_NOISES
    print("\n[4/4] RIRS_NOISES (room impulse responses)")
    download_file(
        "https://www.openslr.org/resources/28/rirs_noises.zip",
        r"data\raw\rirs_noises.zip"
    )
    
    print("\n" + "=" * 60)
    print("EXTRACTING ARCHIVES")
    print("=" * 60)
    
    archives = [
        (r"data\raw\train-clean-5.tar.gz", "tar"),
        (r"data\raw\dev-clean-2.tar.gz", "tar"),
        (r"data\raw\musan.tar.gz", "tar"),
        (r"data\raw\rirs_noises.zip", "zip"),
    ]
    
    for i, (path, _) in enumerate(archives, 1):
        print(f"\n[{i}/{len(archives)}]", end=" ")
        extract_file(path)
    
    print("\n" + "=" * 60)
    print("✓ DOWNLOAD & EXTRACT COMPLETE")
    print("=" * 60)
