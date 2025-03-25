import os
import shutil
from pathlib import Path

# Base paths
repo_path = Path(".")
docs_path = repo_path / "docs"

# Create directories if they don't exist
architecture_path = docs_path / "architecture"
report_path = docs_path / "report"
diagrams_path = docs_path / "diagrams"

os.makedirs(architecture_path, exist_ok=True)
os.makedirs(report_path, exist_ok=True)
os.makedirs(diagrams_path, exist_ok=True)

# Files to move to architecture directory
architecture_files = [
    "MessageGNN_Architecture_Analysis.txt",
    "VariableGNNLayer_analysis.txt"
]

# Files to move to report directory
report_files = [
    "Report_Modification_Suggestions.md",
    "REPORT_STRUCTURE.md"
]

# Files to move to diagrams directory
diagrams_files = [
    "MessageGNN_Architecture_Diagrams.md"
]

# Move files to their respective directories
for file in architecture_files:
    src = repo_path / file
    dst = architecture_path / file
    if src.exists():
        print(f"Moving {src} to {dst}")
        shutil.copy2(src, dst)
        os.remove(src)

for file in report_files:
    src = repo_path / file
    dst = report_path / file
    if src.exists():
        print(f"Moving {src} to {dst}")
        shutil.copy2(src, dst)
        os.remove(src)

for file in diagrams_files:
    src = repo_path / file
    dst = diagrams_path / file
    if src.exists():
        print(f"Moving {src} to {dst}")
        shutil.copy2(src, dst)
        os.remove(src)

print("Organization complete!") 