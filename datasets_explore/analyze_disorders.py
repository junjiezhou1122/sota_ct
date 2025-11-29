#!/usr/bin/env python3
"""
Script to analyze the disorders in the train_case_disorders.csv file from the RadGenome-ChestCT dataset.
"""

from huggingface_hub import hf_hub_download
import pandas as pd
from collections import Counter
import re

def analyze_disorders(repo_id="RadGenome/RadGenome-ChestCT", filename="train_case_disorders.csv"):
    """
    Download and analyze the disorders in the CSV file.
    
    Args:
        repo_id (str): The Hugging Face repository ID
        filename (str): The name of the CSV file to examine
    """
    print(f"Analyzing disorders in {filename} from {repo_id}")
    print("=" * 60)
    
    try:
        # Download the file from Hugging Face
        print(f"Downloading {filename}...")
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"dataset/radgenome_files/{filename}",
            repo_type="dataset"
        )
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Basic statistics
        print(f"\n📊 Basic Statistics:")
        print(f"Total records: {len(df)}")
        print(f"Unique cases: {df['Volumename'].nunique()}")
        print(f"Cases with 'no findings': {df[df['Disorders'] == 'no findings'].shape[0]}")
        
        # Extract individual disorders
        all_disorders = []
        for disorders_str in df['Disorders']:
            # Split by comma and clean up
            disorders = [d.strip() for d in disorders_str.split(',')]
            all_disorders.extend(disorders)
        
        # Count disorder frequencies
        disorder_counts = Counter(all_disorders)
        
        print(f"\n🏥 Top 20 Most Common Disorders:")
        for disorder, count in disorder_counts.most_common(20):
            print(f"  {count:4d} - {disorder}")
        
        # Analyze cases with multiple disorders
        disorder_counts_per_case = [len(disorders_str.split(',')) for disorders_str in df['Disorders']]
        avg_disorders = sum(disorder_counts_per_case) / len(disorder_counts_per_case)
        
        print(f"\n📈 Disorders per Case:")
        print(f"Average: {avg_disorders:.2f}")
        print(f"Minimum: {min(disorder_counts_per_case)}")
        print(f"Maximum: {max(disorder_counts_per_case)}")
        
        # Create categories based on common medical terms
        categories = {
            'Lung-related': ['lung', 'pulmonary', 'pneumonia', 'emphysema', 'atelectasis', 'fibrosis', 'bronch'],
            'Heart-related': ['heart', 'cardiac', 'coronary', 'cardiomegaly', 'pericardial'],
            'Vascular': ['vascular', 'atherosclerosis', 'atheroma', 'aneurysm', 'embolism'],
            'Bone-related': ['bone', 'vertebral', 'spine', 'rib', 'fracture', 'osteopenia'],
            'Pleura-related': ['pleural', 'effusion', 'pleur'],
            'Mediastinum': ['mediastinal', 'mediastinum', 'lymphadenopathy'],
            'Nodule-related': ['nodule', 'nodules', 'mass'],
            'Normal/No findings': ['no findings', 'normal', 'unremarkable']
        }
        
        print(f"\n📋 Disorder Categories:")
        for category, keywords in categories.items():
            count = sum(1 for disorder in all_disorders if any(keyword in disorder.lower() for keyword in keywords))
            print(f"  {category}: {count} occurrences")
        
        # Examples of complex cases (with many disorders)
        complex_cases = df.iloc[[i for i, count in enumerate(disorder_counts_per_case) if count >= 10]]
        
        if not complex_cases.empty:
            print(f"\n🔍 Example of Complex Cases (with 10+ disorders):")
            for idx, row in complex_cases.head(3).iterrows():
                print(f"\nCase: {row['Volumename']}")
                disorders = [d.strip() for d in row['Disorders'].split(',')]
                print(f"Number of disorders: {len(disorders)}")
                print("Disorders:")
                for i, disorder in enumerate(disorders[:5], 1):
                    print(f"  {i}. {disorder}")
                if len(disorders) > 5:
                    print(f"  ... and {len(disorders) - 5} more")
        
        # Save the analysis
        analysis_file = f"{filename}_analysis.txt"
        with open(analysis_file, 'w') as f:
            f.write(f"Analysis of {filename}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total records: {len(df)}\n")
            f.write(f"Unique cases: {df['Volumename'].nunique()}\n")
            f.write(f"Cases with 'no findings': {df[df['Disorders'] == 'no findings'].shape[0]}\n\n")
            
            f.write("Top 20 Most Common Disorders:\n")
            for disorder, count in disorder_counts.most_common(20):
                f.write(f"  {count:4d} - {disorder}\n")
            
            f.write(f"\nDisorders per Case:\n")
            f.write(f"Average: {avg_disorders:.2f}\n")
            f.write(f"Minimum: {min(disorder_counts_per_case)}\n")
            f.write(f"Maximum: {max(disorder_counts_per_case)}\n")
            
            f.write(f"\nDisorder Categories:\n")
            for category, keywords in categories.items():
                count = sum(1 for disorder in all_disorders if any(keyword in disorder.lower() for keyword in keywords))
                f.write(f"  {category}: {count} occurrences\n")
        
        print(f"\n💾 Analysis saved to: {analysis_file}")
        
        return df, disorder_counts
        
    except Exception as e:
        print(f"Error analyzing file: {str(e)}")
        return None, None

if __name__ == "__main__":
    # Analyze the train_case_disorders.csv file
    df, disorder_counts = analyze_disorders()