import subprocess
import shutil
from pathlib import Path

# Import __version__ from your app
import gui_prediction_app

version = gui_prediction_app.__version__
build_name = f"veg_prediction_app_v{version}"

# Remove previous build directories
dist_dir = Path("dist")
if dist_dir.exists():
    shutil.rmtree(dist_dir)

build_dir = Path("build")
if build_dir.exists():
    shutil.rmtree(build_dir)

# Build with PyInstaller
subprocess.run(["pyinstaller", "veg_prediction_app.spec"], check=True)

# Define paths
dist_dir = Path("dist")
output_dir = dist_dir / build_name

# Rename/move the build output folder to include version
if output_dir.exists():
    shutil.rmtree(output_dir)
(dist_dir / "veg_prediction_app").rename(output_dir)

# Optionally zip the folder
shutil.make_archive(str(output_dir), "zip", root_dir=output_dir)

print(f"Build complete: {output_dir}.zip")
