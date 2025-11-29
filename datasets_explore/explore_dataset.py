#!/usr/bin/env python3
"""
Script to explore the file structure of RadGenome-ChestCT dataset from Hugging Face
without downloading the actual files.
"""

from huggingface_hub import HfApi
import os
from collections import defaultdict

def explore_dataset_structure(repo_id="RadGenome/RadGenome-ChestCT"):
    """
    Explore and display the structure of a Hugging Face dataset repository.
    
    Args:
        repo_id (str): The Hugging Face repository ID
    """
    print(f"Exploring dataset structure for: {repo_id}")
    print("=" * 60)
    
    # Initialize the Hugging Face API
    api = HfApi()
    
    try:
        # Get the repository tree structure
        files = api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            recursive=True
        )
        
        # Organize files by directory
        directory_structure = defaultdict(list)
        
        for file in files:
            # Extract directory path and file name
            path_parts = file.path.split('/')
            if len(path_parts) > 1:
                directory = '/'.join(path_parts[:-1])
                filename = path_parts[-1]
            else:
                directory = "root"
                filename = path_parts[0]
            
            # Check if it's a directory or file based on the object type
            is_directory = hasattr(file, 'type') and file.type == 'directory'
            
            directory_structure[directory].append({
                'name': filename,
                'path': file.path,
                'size': getattr(file, 'size', None),
                'type': 'directory' if is_directory else 'file'
            })
        
        # Display the structure
        print("\n📁 Dataset Directory Structure:\n")
        
        for directory in sorted(directory_structure.keys()):
            print(f"📂 {directory}/")
            for item in sorted(directory_structure[directory], key=lambda x: x['name']):
                icon = "📁" if item['type'] == 'directory' else "📄"
                size_info = f" ({item['size']} bytes)" if item['size'] else ""
                print(f"  {icon} {item['name']}{size_info}")
            print()
        
        # Display statistics
        total_files = 0
        total_dirs = 0
        
        for file in files:
            if hasattr(file, 'type') and file.type == 'directory':
                total_dirs += 1
            else:
                total_files += 1
        
        print("\n📊 Statistics:")
        print(f"Total files: {total_files}")
        print(f"Total directories: {total_dirs}")
        
        # Display file extensions
        extensions = defaultdict(int)
        for file in files:
            if not (hasattr(file, 'type') and file.type == 'directory') and '.' in file.path:
                ext = file.path.split('.')[-1]
                extensions[ext] += 1
        
        if extensions:
            print("\n📋 File Extensions:")
            for ext, count in sorted(extensions.items()):
                print(f"  .{ext}: {count} files")
        
        return directory_structure
        
    except Exception as e:
        print(f"Error exploring dataset: {str(e)}")
        return None

def analyze_naming_patterns(directory_structure):
    """
    Analyze naming patterns in the dataset files.
    
    Args:
        directory_structure: Dictionary with directory structure
    """
    print("\n🔍 File Naming Pattern Analysis:")
    print("-" * 40)
    
    # Look for NIfTI files and analyze their naming patterns
    nifti_files = []
    
    for directory, files in directory_structure.items():
        for file in files:
            if file['name'].endswith('.nii.gz'):
                nifti_files.append(file['path'])
    
    if nifti_files:
        print(f"\nFound {len(nifti_files)} NIfTI files:")
        
        # Extract patterns from file names
        patterns = set()
        for file_path in nifti_files:
            filename = os.path.basename(file_path)
            # Extract pattern (e.g., train_16242_a_1)
            pattern = '_'.join(filename.split('_')[:3])
            patterns.add(pattern)
        
        print("\n📝 Naming Patterns:")
        for pattern in sorted(patterns):
            print(f"  - {pattern}_*.nii.gz")
        
        # Show some examples
        print("\n📄 Example Files:")
        for i, file_path in enumerate(nifti_files[:5]):  # Show first 5
            print(f"  {i+1}. {file_path}")
        
        if len(nifti_files) > 5:
            print(f"  ... and {len(nifti_files) - 5} more files")

if __name__ == "__main__":
    # Check if huggingface_hub is installed
    try:
        import huggingface_hub
    except ImportError:
        print("Installing required package: huggingface_hub")
        import subprocess
        subprocess.check_call(["pip", "install", "huggingface_hub"])
        import huggingface_hub
    
    # Explore the dataset
    structure = explore_dataset_structure()
    
    if structure:
        analyze_naming_patterns(structure)