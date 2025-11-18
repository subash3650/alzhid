import kagglehub
import os
import shutil

# Download latest version
print("Downloading Alzheimer's MRI Dataset...")
path = kagglehub.dataset_download("lukechugh/best-alzheimer-mri-dataset-99-accuracy")

print("Path to dataset files:", path)

# Copy to current working directory
current_dir = os.getcwd()
destination = os.path.join(current_dir, "alzheimer_dataset")

if os.path.exists(destination):
    shutil.rmtree(destination)

shutil.copytree(path, destination)
print(f"Dataset copied to: {destination}")
print(f"Dataset structure:")
for root, dirs, files in os.walk(destination):
    level = root.replace(destination, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    sub_indent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # Show first 5 files per directory
        print(f'{sub_indent}{file}')
    if len(files) > 5:
        print(f'{sub_indent}... and {len(files) - 5} more files')
