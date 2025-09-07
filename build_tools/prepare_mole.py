"""Build script for downloading and building MOLE."""
import os
import shutil
import tarfile
import subprocess
from pathlib import Path
from typing import Optional
import urllib.request
import zipfile

ARMADILLO_SOURCE = "https://sourceforge.net/projects/arma/files/armadillo-12.6.6.tar.xz"
MOLE_REPO = "https://github.com/csrc-sdsu/mole/archive/refs/heads/main.zip"
MOLE_DIR = "mole-main"

def build_armadillo(source_dir) -> bool:
    """build Armadillo library."""
    # 1. Run CMake configure
    build_dir = os.path.join(source_dir, "build")
    os.makedirs(build_dir, exist_ok=True)

    cmake_configure = [
        "cmake",
        "-S", source_dir,
        "-B", build_dir,
        "-DARMADILLO_USE_SUPERLU=ON",
        "-DARMADILLO_USE_OPENMP=ON",
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
    ]
    print("Configuring with CMake...")
    subprocess.check_call(cmake_configure)

    # 2. Run CMake build
    cmake_build = ["cmake", "--build", build_dir, "--config", "Release"]
    print("Building Armadillo...")
    subprocess.check_call(cmake_build)
    return True

def download_dependencies(temp_dir: Path, target_dir: Path) -> bool:
    """ Download and build Armadillo (MOLE's dependency) library. """

    url = ARMADILLO_SOURCE
    archive_name = temp_dir / "armadillo-12.6.6.tar.xz"
    package_dir = target_dir / "armadillo-12.6.6"

    # 1. Download the tar.xz if not already present
    if not os.path.exists(archive_name):
        print(f"Downloading {archive_name}...")
        urllib.request.urlretrieve(url, archive_name)
    else:
        print(f"{archive_name} already exists, skipping download.")

    # 2. Extract tar.xz
    if not os.path.exists(package_dir):
        print(f"Extracting {archive_name}...")
        with tarfile.open(archive_name, mode="r:xz") as tar:
            tar.extractall(target_dir)
    else:
        print(f"{package_dir} already extracted, skipping.")

    # 3. Build Armadillo
    if not os.path.exists(os.path.join(package_dir, "build")):
        print("Building Armadillo...")
        build_armadillo(package_dir)
    else:
        print("Armadillo already built, skipping.")

    print("✅ Armadillo build completed!")
    return True


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
    # shutil.move(mole_cpp_dir, cpp_dir)
    if not os.path.exists(cpp_dir):
        os.makedirs(cpp_dir)

    for item in os.listdir(mole_cpp_dir):
        s = os.path.join(mole_cpp_dir, item)
        d = os.path.join(cpp_dir, item)
        shutil.move(s, d)

def prepare_mole(base_dir: Optional[Path] = None) -> None:
    """Download and prepare MOLE library for building."""
    if base_dir is None:
        base_dir = Path(__file__).parent
    
    temp_dir = base_dir / "temp_build"
    target_dir = base_dir / "src" / "pymole" / "cpp"
    
    # Create temporary directory
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Download dependencies
        download_dependencies(temp_dir, target_dir / "cpp")

        # Download MOLE
        download_mole(temp_dir)
        
        # Copy source files
        copy_mole_sources(temp_dir, target_dir)
    finally:
        # Clean up
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    path = Path("../").resolve()
    prepare_mole(path)
