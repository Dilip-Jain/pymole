"""Setup script for pymole package."""
import os
import sys
from pathlib import Path
from setuptools import setup
from setuptools.command.build_ext import build_ext
from build_tools.prepare_mole import prepare_mole

class PreBuildCommand(build_ext):
    """Custom build command to download and prepare MOLE before building."""
    
    def run(self):
        """Run the pre-build steps and then the regular build."""
        # Download and prepare MOLE
        prepare_mole(Path(__file__).parent)
        
        # Run the regular build
        super().run()

if __name__ == "__main__":
    setup(
        cmdclass={
            "build_ext": PreBuildCommand,
        }
    )
