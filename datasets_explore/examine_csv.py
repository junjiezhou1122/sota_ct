#!/usr/bin/env python3
"""
Script to examine the contents of the train_case_disorders.csv file from the RadGenome-ChestCT dataset.
"""

from huggingface_hub import hf_hub_download
import pandas as pd
import os

def examine_csv_file(repo_id="RadGenome/RadGenome-ChestCT", filename="train_case_disorders.csv"):
    """
    Download and examine a CSV file from the Hugging Face dataset.
    
    Args:
        repo_id (str): The Hugging Face repository ID
        filename (str): The name of the CSV file to examine
    """
    print(f"Examining {filename} from {repo_id}")
    print("=" * 60)
    
    try:
        # Download the file from Hugging Face
        print(f"Downloading {filename}...")
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"dataset/radgenome_files/{filename}",
            repo_type="dataset"
        )
        
        print(f"File downloaded to: {file_path}")
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Display basic information
        print(f"\n📊 Dataset Information:")
        print(f"Shape: {df.shape} (rows, columns)")
        print(f"Columns: {list(df.columns)}")
        
        # Display first few rows
        print(f"\n📄 First 5 rows:")
        print(df.head())
        
        # Display data types
        print(f"\n📋 Column Data Types:")
        print(df.dtypes)
        
        # Display unique values for categorical columns
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_count = df[col].nunique()
                print(f"\n📝 Column '{col}' has {unique_count} unique values")
                
                if unique_count <= 20:
                    print("Unique values:")
                    for val in df[col].unique():
                        print(f"  - {val}")
                else:
                    print(f"First 10 unique values:")
                    for val in df[col].unique()[:10]:
                        print(f"  - {val}")
                    print(f"  ... and {unique_count - 10} more")
        
        # If there's a case_id column, show some statistics
        if 'case_id' in df.columns:
            print(f"\n🔍 Case ID Statistics:")
            print(f"Number of unique cases: {df['case_id'].nunique()}")
            print(f"Case ID range: {df['case_id'].min()} to {df['case_id'].max()}")
        
        # If there's a disorder column, show some statistics
        if 'disorder' in df.columns:
            print(f"\n🏥 Disorder Statistics:")
            disorder_counts = df['disorder'].value_counts()
            print(f"Number of unique disorders: {len(disorder_counts)}")
            print("\nTop 10 most common disorders:")
            for disorder, count in disorder_counts.head(10).items():
                print(f"  - {disorder}: {count} cases")
        
        # Save a summary of the file
        summary_file = f"{filename}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Summary of {filename}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Shape: {df.shape} (rows, columns)\n")
            f.write(f"Columns: {list(df.columns)}\n\n")
            f.write("Column Data Types:\n")
            f.write(str(df.dtypes) + "\n\n")
            f.write("First 5 rows:\n")
            f.write(str(df.head()) + "\n\n")
            
            if 'disorder' in df.columns:
                f.write("Disorder Counts:\n")
                f.write(str(df['disorder'].value_counts()) + "\n")
        
        print(f"\n💾 Summary saved to: {summary_file}")
        
        return df
        
    except Exception as e:
        print(f"Error examining file: {str(e)}")
        return None

if __name__ == "__main__":
    # Examine the train_case_disorders.csv file
    df = examine_csv_file()