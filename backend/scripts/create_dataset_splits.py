#!/usr/bin/env python3
"""
Script to create train/valid/test splits from the main leads.csv dataset.
This meets the project requirement for explicit train/valid/test splits.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

def create_dataset_splits():
    """Create train/valid/test splits with proper stratification."""
    
    # Load the main dataset
    data_path = Path("backend/data/leads.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    print(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Create target variable for stratification (if not exists)
    # For lead scoring, we'll create a target based on prior_course_interest
    if 'target' not in df.columns:
        # Map prior_course_interest to numeric for stratification
        interest_mapping = {'low': 0, 'medium': 1, 'high': 2}
        df['target'] = df['prior_course_interest'].map(interest_mapping)
        print("Created target variable from prior_course_interest")
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['target', 'name', 'email', 'phone']]
    X = df[feature_cols]
    y = df['target']
    
    print(f"Features: {len(feature_cols)} columns")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # First split: 80% train+valid, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Second split: 80% train, 20% valid (of the 80% remaining)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    # Create full datasets with all columns
    train_df = df.iloc[X_train.index].copy()
    valid_df = df.iloc[X_valid.index].copy()
    test_df = df.iloc[X_test.index].copy()
    
    # Remove the temporary target column
    for split_df in [train_df, valid_df, test_df]:
        if 'target' in split_df.columns:
            split_df.drop('target', axis=1, inplace=True)
    
    print(f"\nSplit sizes:")
    print(f"Train: {len(train_df)} rows ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Valid: {len(valid_df)} rows ({len(valid_df)/len(df)*100:.1f}%)")
    print(f"Test: {len(test_df)} rows ({len(test_df)/len(df)*100:.1f}%)")
    
    # Save splits
    output_dir = Path("backend/data")
    output_dir.mkdir(exist_ok=True)
    
    train_path = output_dir / "leads_train.csv"
    valid_path = output_dir / "leads_valid.csv"
    test_path = output_dir / "leads_test.csv"
    
    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nSaved splits:")
    print(f"Train: {train_path}")
    print(f"Valid: {valid_path}")
    print(f"Test: {test_path}")
    
    # Verify splits
    print(f"\nVerification:")
    print(f"Train shape: {train_df.shape}")
    print(f"Valid shape: {valid_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Total: {len(train_df) + len(valid_df) + len(test_df)} (original: {len(df)})")
    
    return train_df, valid_df, test_df

if __name__ == "__main__":
    create_dataset_splits()
    print("\n✅ Dataset splits created successfully!")
