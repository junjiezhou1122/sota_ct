#!/usr/bin/env python3
"""
Script to explore the directory structure of RadGenome-ChestCT dataset from Hugging Face
without downloading the actual files. This version focuses on the directory structure.
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
            
            # Count files and directories separately
            file_count = 0
            dir_count = 0
            
            for item in directory_structure[directory]:
                if item['type'] == 'directory':
                    dir_count += 1
                else:
                    file_count += 1
            
            # Show summary instead of listing all files if there are many
            if file_count + dir_count > 10:
                print(f"  📄 {file_count} files")
                print(f"  📁 {dir_count} subdirectories")
                
                # Show first few examples
                example_files = [item for item in directory_structure[directory] if item['type'] == 'file'][:3]
                for item in example_files:
                    print(f"    📄 {item['name']}")
            else:
                # List all items if there aren't many
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

def analyze_dataset_info(directory_structure):
    """
    Analyze and provide information about the dataset structure.
    
    Args:
        directory_structure: Dictionary with directory structure
    """
    print("\n🔍 Dataset Information:")
    print("-" * 40)
    
    # Look for key directories
    key_dirs = ['dataset/train_preprocessed', 'dataset/valid_preprocessed', 'dataset/radgenome_files']
    
    for dir_path in key_dirs:
        if dir_path in directory_structure:
            print(f"\n📂 {dir_path}/")
            
            # Count files and subdirectories
            file_count = sum(1 for item in directory_structure[dir_path] if item['type'] == 'file')
            dir_count = sum(1 for item in directory_structure[dir_path] if item['type'] == 'directory')
            
            print(f"  📄 {file_count} files")
            print(f"  📁 {dir_count} subdirectories")
            
            # Show some examples
            if dir_count > 0:
                print("  Subdirectories:")
                for item in directory_structure[dir_path]:
                    if item['type'] == 'directory':
                        print(f"    📁 {item['name']}")
            
            if file_count > 0:
                print("  Files:")
                for item in directory_structure[dir_path]:
                    if item['type'] == 'file':
                        print(f"    📄 {item['name']}")

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
        analyze_dataset_info(structure)