import json
import os
import sys
import time
import psutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import load_config, setup_logging
from data_prep import DataPreprocessor
from modeling import ModelTrainer
from sklearn.pipeline import Pipeline

def log_memory_usage():
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"\nMemory usage: {mem_info.rss / 1024 / 1024:.2f} MB")
    print(f"Available memory: {psutil.virtual_memory().available / 1024 / 1024:.2f} MB")

def main():
    # Load configuration
    config = load_config('config.yml')
    
    # Setup logging
    setup_logging()
    
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Load and preprocess data with sampling
    sample_size = config['data'].get('sample_size', 0.05)  # Default to 5% sample
    print(f"\nUsing sample size: {sample_size*100:.1f}% of the data")
    
    # Load data with memory optimization
    print("\nLoading data...")
    log_memory_usage()
    
    try:
        df = preprocessor.load_data(
            filepath=config['data']['input_file'],
            sample_size=sample_size
        )
        print("Data loaded successfully!")
        log_memory_usage()
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Create features with progress tracking
    print("\nCreating features...")
    log_memory_usage()
    try:
        df = preprocessor.create_features(df)
        print("\nFeature creation completed!")
        log_memory_usage()
    except Exception as e:
        print(f"Error creating features: {e}")
        sys.exit(1)
    
    # Split data
    print("\nSplitting data into train/test sets...")
    try:
        X_train, X_test, y_train, y_test = preprocessor.split_data(
            df, 
            target_col=config['features']['target_frequency']
        )
        print(f"Training set size: {len(X_train):,} samples")
        print(f"Test set size: {len(X_test):,} samples")
        log_memory_usage()
    except Exception as e:
        print(f"Error splitting data: {e}")
        sys.exit(1)
    
    # Get preprocessor
    preprocessor = preprocessor.get_preprocessor(
        config['features']['numerical'],
        config['features']['categorical']
    )
    
    # Train frequency model with preprocessor
    print("\nInitializing model training...")
    log_memory_usage()
    
    try:
        model_trainer = ModelTrainer(config)
        
        # Use a simpler model for testing
        model_type = 'random_forest'  # Using random forest as it's more memory efficient
        print(f"Using model type: {model_type}")
        
        # Train the model with progress tracking
        print("\nStarting model training...")
        start_time = time.time()
        
        # Train model with reduced parameters
        model = model_trainer.train_model(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor,
            param_grid={
                'n_estimators': [50],  # Reduced number of trees
                'max_depth': [5, 10],  # Simpler trees
                'min_samples_split': [5],
                'min_samples_leaf': [2]
            } if model_type == 'random_forest' else None
        )
        
        training_time = time.time() - start_time
        print(f"\nModel training completed in {training_time:.2f} seconds")
        log_memory_usage()
        
    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Train the model with the preprocessor in the pipeline
    frequency_model = model_trainer.train_model(
        model_type=model_type,
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,  # Let the pipeline handle preprocessing
        param_grid=config['model']['frequency'].get('params')
    )
    
    # For evaluation, we'll use the pipeline which includes preprocessing
    X_test_transformed = X_test  # The pipeline will handle transformation
    
    # Create results directory
    results_dir = 'results/analysis'
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Evaluate model with default threshold (0.5)
    print("\n" + "="*80)
    print("INITIAL MODEL EVALUATION (threshold=0.5)")
    print("="*80)
    
    # Get probability predictions
    y_pred_proba = frequency_model.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold using precision-recall curve
    from sklearn.metrics import precision_recall_curve, f1_score
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)  # Add small epsilon to avoid division by zero
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Evaluate with optimal threshold
    print(f"\nOptimal threshold: {optimal_threshold:.4f}")
    print(f"Best F1-score: {f1_scores[optimal_idx]:.4f}")
    
    # Evaluate with both default and optimal thresholds
    eval_metrics_default = model_trainer.evaluate_model(
        frequency_model, X_test, y_test, is_classification=True, threshold=0.5
    )
    
    eval_metrics_optimal = model_trainer.evaluate_model(
        frequency_model, X_test, y_test, is_classification=True, threshold=optimal_threshold
    )
    
    # Print evaluation metrics
    print("\n" + "="*80)
    print("EVALUATION WITH DEFAULT THRESHOLD (0.5)")
    print("="*80)
    print("\nClassification Report:")
    print(eval_metrics_default['classification_report'])
    print(f"\nROC AUC: {eval_metrics_default['roc_auc']:.4f}")
    
    print("\n" + "="*80)
    print(f"EVALUATION WITH OPTIMAL THRESHOLD ({optimal_threshold:.4f})")
    print("="*80)
    print("\nClassification Report:")
    print(eval_metrics_optimal['classification_report'])
    print(f"\nROC AUC: {eval_metrics_optimal['roc_auc']:.4f}")
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.generic, np.ndarray)):
            return obj.item() if obj.size == 1 else obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(x) for x in obj]
        return obj
    
    # Prepare metrics for saving
    metrics_to_save = {
        'default_threshold': convert_numpy(eval_metrics_default),
        'optimal_threshold': convert_numpy(eval_metrics_optimal),
        'optimal_threshold_value': float(optimal_threshold)
    }
    
    # Save to file
    with open(f'{results_dir}/evaluation_metrics.json', 'w') as f:
        json.dump(metrics_to_save, f, indent=2, default=convert_numpy)
    
    # Plot precision-recall curve
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.scatter(recall[optimal_idx], precision[optimal_idx], 
               marker='o', color='red', label=f'Optimal threshold ({optimal_threshold:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{results_dir}/precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Analyze feature importance
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    try:
        # Get feature names from the preprocessor
        feature_names = []
        
        # Get the preprocessor from the pipeline
        preprocessor = frequency_model.named_steps['preprocessor']
        
        # Extract feature names from the ColumnTransformer
        for name, transformer, features in preprocessor.transformers_:
            if name == 'remainder':
                continue  # Skip the remainder transformer
                
            if transformer == 'drop':
                continue  # Skip dropped features
                
            if name == 'num':
                # For numerical features, use the original names
                feature_names.extend(features)
            elif name == 'cat':
                # For categorical features, get the one-hot encoded names
                if hasattr(transformer, 'named_steps') and 'onehot' in transformer.named_steps:
                    encoder = transformer.named_steps['onehot']
                    if hasattr(encoder, 'get_feature_names_out'):
                        cat_features = encoder.get_feature_names_out(features)
                        feature_names.extend(cat_features)
                    else:
                        # Fallback if get_feature_names_out is not available
                        feature_names.extend(
                            f"{col}_{i}" 
                            for i, col in enumerate(features) for _ in range(len(transformer.categories_[i]))
                        )
                else:
                    # If no onehot step, just use the original names
                    feature_names.extend(features)
        
        # Ensure we have the same number of features as expected by the model
        if X_test_transformed.shape[1] != len(feature_names):
            print(f"Warning: Number of features in transformed data ({X_test_transformed.shape[1]}) "
                  f"doesn't match number of feature names ({len(feature_names)}). "
                  "Using generic feature names.")
            feature_names = [f'feature_{i}' for i in range(X_test_transformed.shape[1])]
        
        # Analyze and save feature importance
        model_trainer.analyze_feature_importance(
            model=frequency_model,
            X_test=X_test_transformed,
            feature_names=feature_names,
            output_dir=results_dir
        )
        print(f"Feature importance analysis completed. Results saved to {results_dir}")
    
    except Exception as e:
        import traceback
        print(f"Error during feature importance analysis: {str(e)}")
        print("Stack trace:", traceback.format_exc())
    
    # 3. Save the trained model
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)
    model_path = 'models/frequency_model.pkl'
    model_trainer.save_model(frequency_model, model_path)
    print(f"Model saved to {model_path}")
    
    # 4. Generate predictions for analysis
    y_pred_proba = frequency_model.predict_proba(X_test)[:, 1]
    y_pred = frequency_model.predict(X_test)
    
    # Save predictions for further analysis
    predictions_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'probability': y_pred_proba
    })
    predictions_df.to_csv(f'{results_dir}/predictions.csv', index=False)
    print(f"\nPredictions saved to {results_dir}/predictions.csv")

if __name__ == "__main__":
    main()
