import kagglehub
import os

# Define the output directory in the project root
project_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(project_dir, "dataset")

# Download latest version directly to the project folder instead of cache
path = kagglehub.dataset_download(
    "ahmedhamada0/brain-tumor-detection",
    output_dir=output_dir
)

print("Path to dataset files:", path)
