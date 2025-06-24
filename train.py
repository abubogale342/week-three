#!/usr/bin/env python3
"""
Model training and evaluation script for insurance claim prediction.
"""
import os
import sys
import yaml
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_prep import DataPreprocessor
from modeling import ModelTrainer

def load_config(config_path: str = 'config.yml') -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Load configuration
    config = load_config()
    
    # Set up directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    # Initialize data preprocessor with config
    print("\n=== Loading and preprocessing data ===")
    preprocessor = DataPreprocessor(config)
    
    # Load and preprocess data
    df = preprocessor.load_data(
        filepath=config['data']['input_file'],
        sample_size=config['data'].get('sample_size', 1.0)
    )
    df = preprocessor.create_features(df)
    df = preprocessor.optimize_memory(df)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        df=df, 
        target_col='HasClaim',  # Using 'HasClaim' as the target column
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )
    
    # Get preprocessor
    preprocessor_pipe = preprocessor.get_preprocessor(
        numerical_cols=config['features']['numerical'],
        categorical_cols=config['features']['categorical']
    )
    
    # Initialize model trainer with config
    print("\n=== Training model ===")
    model_trainer = ModelTrainer(config)
    
    # Define parameter grid for XGBoost
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8],
        'gamma': [0, 0.1, 0.2],
        'min_child_weight': [1, 3, 5]
    }
    
    # Train model
    best_model = model_trainer.train_model(
        model_type='xgb_classifier',  # Use XGBoost for classification
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor_pipe,
        param_grid=param_grid
    )
    
    # Save the model
    model_path = 'models/insurance_claim_model.joblib'
    joblib.dump(best_model, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Evaluate model with the full pipeline
    print("\n=== Evaluating model ===")
    X_test_processed = preprocessor_pipe.transform(X_test)
    
    # Get feature names after preprocessing
    try:
        # Try to get feature names from the preprocessor
        feature_names = preprocessor_pipe.get_feature_names_out()
    except AttributeError:
        # Fallback to default feature names if not available
        feature_names = [f'feature_{i}' for i in range(X_test_processed.shape[1])]
    
    # Evaluate model
    metrics = model_trainer.evaluate_model(
        best_model,
        X_test_processed,
        y_test,
        is_classification=True
    )
    
    # Analyze feature importance if possible
    try:
        model_trainer.analyze_feature_importance(
            best_model,
            X_test_processed,
            feature_names,
            output_dir='results/plots'
        )
    except Exception as e:
        print(f"\nCould not analyze feature importance: {e}")
    
    print("\n=== Training and evaluation complete! ===")

if __name__ == "__main__":
    main()
