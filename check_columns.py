#!/usr/bin/env python3
"""
Script to check the columns in the dataset.
"""
import pandas as pd
import yaml

def load_config():
    with open('config.yml', 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    input_file = config['data']['input_file']
    
    # Read just the first row to get column names
    df = pd.read_csv(input_file, nrows=1)
    
    print("\n=== Dataset Columns ===")
    for col in df.columns:
        print(f"- {col}")
    
    print("\n=== Target Column ===")
    if 'Claim' in df.columns:
        print("'Claim' column found!")
    elif 'HasClaim' in df.columns:
        print("'HasClaim' column found! (using as target)")
    else:
        print("No obvious target column found. Please check the dataset.")

if __name__ == "__main__":
    main()
