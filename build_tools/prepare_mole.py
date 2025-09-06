"""Build script for downloading and building MOLE."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional
import urllib.request
import zipfile

MOLE_REPO = "https://github.com/csrc-sdsu/mole/archive/refs/heads/main.zip"
MOLE_DIR = "mole-main"

def download_mole(target_dir: Path) -> None:
    """Download MOLE library from GitHub."""
    zip_path = target_dir / "mole.zip"
    print(f"Downloading MOLE from {MOLE_REPO}")
    
    # Download zip file
    urllib.request.urlretrieve(MOLE_REPO, zip_path)
    
    # Extract files
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    
    # Clean up zip file
    zip_path.unlink()

def copy_mole_sources(source_dir: Path, target_dir: Path) -> None:
    """Copy MOLE source files to build directory."""
    # Create cpp directory if it doesn't exist
    cpp_dir = target_dir / "cpp"
    cpp_dir.mkdir(exist_ok=True)
    
    # Copy all .cpp and .h files from MOLE src/cpp
    mole_cpp_dir = source_dir / MOLE_DIR / "src" / "cpp"
    for file in mole_cpp_dir.glob("*.[ch]pp"):
        shutil.copy2(file, cpp_dir)

def prepare_mole(base_dir: Optional[Path] = None) -> None:
    """Download and prepare MOLE library for building."""
    if base_dir is None:
        base_dir = Path(__file__).parent
    
    temp_dir = base_dir / "temp_build"
    target_dir = base_dir / "src" / "pymole" / "cpp"
    
    # Create temporary directory
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Download MOLE
        download_mole(temp_dir)
        
        # Copy source files
        copy_mole_sources(temp_dir, target_dir)
    finally:
        # Clean up
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    prepare_mole()
