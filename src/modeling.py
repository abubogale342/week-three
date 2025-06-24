from typing import Dict, Any, Tuple, Union, Optional, List
import numpy as np
import pandas as pd
import os
from sklearn.metrics import (
    mean_squared_error, r2_score, classification_report, 
    roc_auc_score, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingClassifier
)
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Set style for plots
plt.style.use('ggplot')  # Using 'ggplot' style which is a valid Matplotlib style
sns.set_style('whitegrid')  # Using seaborn's whitegrid style
sns.set_palette('viridis')

# Set figure size and DPI for better quality plots
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

class ModelTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        
    def train_model(self, model_type: str, X_train: pd.DataFrame, 
                   y_train: pd.Series, preprocessor=None, param_grid: Dict = None) -> Any:
        """
        Train a machine learning model with enhanced handling of class imbalance.
        
        Args:
            model_type: Type of model to train ('xgb_classifier', 'lgbm_classifier', 'random_forest')
            X_train: Training features
            y_train: Training target
            preprocessor: Optional preprocessor (ColumnTransformer or similar)
            param_grid: Optional parameter grid for GridSearchCV
            
        Returns:
            Trained model or pipeline
        """
        # Calculate class weights and distribution
        n_pos = sum(y_train == 1)
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / max(n_pos, 1)  # Avoid division by zero
        
        print(f"\nClass Distribution:")
        print(f"- Positive samples: {n_pos:,} ({n_pos/len(y_train)*100:.2f}%)")
        print(f"- Negative samples: {n_neg:,} ({n_neg/len(y_train)*100:.2f}%)")
        print(f"- Scale pos weight: {scale_pos_weight:.2f}")
        
        # Define base models and their default parameters
        if model_type == 'xgb_classifier':
            # More aggressive parameters for imbalanced data
            base_model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='aucpr',  # Better for imbalanced data
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                tree_method='hist',  # More memory efficient
                use_label_encoder=False,
                n_jobs=-1,
                max_delta_step=1,  # Helps with class imbalance
                grow_policy='lossguide',  # More robust to imbalanced data
                max_leaves=31,  # Limit tree complexity
                min_child_weight=5,  # More regularization
                subsample=0.8,  # Stochastic gradient boosting
                colsample_bytree=0.8,  # More randomness
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=1.0,  # L2 regularization
                learning_rate=0.05  # Slower learning
            )
            
            # Default parameter grid if none provided
            if param_grid is None:
                param_grid = {
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [3, 4, 5],
                    'classifier__learning_rate': [0.01, 0.05],
                    'classifier__subsample': [0.7, 0.8],
                    'classifier__colsample_bytree': [0.7, 0.8],
                    'classifier__min_child_weight': [5, 10],
                    'classifier__gamma': [0.1, 0.2],
                    'classifier__reg_alpha': [0, 0.1, 0.2],
                    'classifier__reg_lambda': [0.8, 1.0, 1.2]
                }
                
        elif model_type == 'lgbm_classifier':
            base_model = lgb.LGBMClassifier(
                objective='binary',
                boosting_type='gbdt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            if param_grid is None:
                param_grid = {
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [3, 5, 7],
                    'classifier__learning_rate': [0.01, 0.1],
                    'classifier__subsample': [0.8, 1.0],
                    'classifier__colsample_bytree': [0.8, 1.0],
                    'classifier__min_child_samples': [20, 50, 100]
                }
                
        elif model_type == 'random_forest':
            base_model = RandomForestClassifier(
                class_weight='balanced',
                random_state=42,
                n_jobs=-1  # Use all available cores
            )
            
            # Default Random Forest parameters
            if param_grid is None:
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
                
        elif model_type == 'xgb_regressor':
            base_model = xgb.XGBRegressor(
                random_state=42,
                n_jobs=-1,
                tree_method='hist',
                enable_categorical=True
            )
            
        elif model_type == 'rf_regressor':
            base_model = RandomForestRegressor(
                random_state=42,
                n_jobs=-1  # Use all available cores
            )
        
        # Create pipeline with SMOTE and model
        pipeline_steps = []
        
        # Add preprocessor if provided
        if preprocessor is not None:
            pipeline_steps.append(('preprocessor', preprocessor))
        
        # Add SMOTE for handling class imbalance if we have enough positive samples
        if n_pos > 5:  # Need at least 5 samples for SMOTE to work
            # Use a smaller k_neighbors value for very small minority classes
            k_neighbors = min(3, n_pos - 1)
            sampling_strategy = min(0.5, (n_neg / n_pos) * 0.1)  # Don't over-sample too much
            
            try:
                smote = SMOTE(
                    sampling_strategy=sampling_strategy,
                    random_state=42,
                    k_neighbors=k_neighbors
                    # Removed n_jobs as it's not supported in newer versions
                )
                pipeline_steps.append(('smote', smote))
                print(f"\nUsing SMOTE with sampling_strategy={sampling_strategy:.2f} and k_neighbors={k_neighbors}")
            except Exception as e:
                print(f"\nWarning: SMOTE initialization failed: {e}")
                print("Continuing without SMOTE...")
        else:
            print("\nNot enough positive samples for SMOTE. Using class weights instead.")
        
        # Add the model as the final step
        pipeline_steps.append(('classifier', base_model))
        
        # Create the pipeline
        model = ImbPipeline(pipeline_steps)
        
        # Update parameter grid to include pipeline steps if needed
        if param_grid is not None:
            param_grid = {f'classifier__{k}': v for k, v in param_grid.items()}
        
        # Use stratified k-fold for cross-validation with more splits for better stability
        n_splits = min(5, n_pos)  # Ensure we don't have more splits than positive samples
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Define scoring metrics for model selection
        scoring = {
            'f1': 'f1',
            'precision': 'precision',
            'recall': 'recall',
            'roc_auc': 'roc_auc',
            'average_precision': 'average_precision'
        }
        
        # Perform grid search if parameter grid is provided
        if param_grid is not None:
            print(f"\nPerforming grid search with {len(param_grid)} parameter combinations...")
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                refit='average_precision',  # Focus on precision-recall for imbalanced data
                n_jobs=-1,
                verbose=1,
                error_score='raise',
                return_train_score=True
            )
            
            # Fit the grid search with progress tracking
            try:
                with tqdm(total=len(param_grid), desc="Grid Search Progress") as pbar:
                    grid_search.fit(X_train, y_train)
                    pbar.update(len(param_grid))
                
                # Print detailed results
                print("\n=== Grid Search Results ===")
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best average precision: {grid_search.best_score_:.4f}")
                
                # Get the best model
                model = grid_search.best_estimator_
                
                # Print cross-validation results
                print("\n=== Cross-validation results ===")
                cv_results = pd.DataFrame(grid_search.cv_results_)
                print(cv_results[['params', 'mean_test_average_precision', 'mean_test_f1', 
                                 'mean_test_precision', 'mean_test_recall', 'mean_test_roc_auc']]
                             .sort_values('mean_test_average_precision', ascending=False).head())
                
                # Generate cross-validated predictions for threshold optimization
                print("\nOptimizing classification threshold...")
                y_pred_proba = cross_val_predict(
                    model, X_train, y_train, 
                    cv=cv, method='predict_proba', n_jobs=-1
                )[:, 1]
                
                # Find optimal threshold using precision-recall curve
                precision, recall, thresholds = precision_recall_curve(y_train, y_pred_proba)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
                optimal_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[optimal_idx]
                
                print(f"Optimal threshold: {optimal_threshold:.4f}")
                print(f"Best F1-score: {f1_scores[optimal_idx]:.4f}")
                
                # Evaluate with optimal threshold
                print("\n=== Evaluation with optimal threshold ===")
                self.evaluate_model(model, X_train, y_train, is_classification=True, threshold=optimal_threshold)
            
            except Exception as e:
                print(f"Error during grid search: {e}")
                print("Falling back to default parameters...")
                model.fit(X_train, y_train)
        else:
            # If no grid search, just fit the model
            print("\nTraining model with default parameters...")
            model.fit(X_train, y_train)
            
            # Generate predictions
            y_pred_proba = model.predict_proba(X_train)[:, 1]
            
            # Calculate and print evaluation metrics
            self._evaluate_model(y_train, y_pred_proba, threshold=0.5)
            
        return model
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, 
                      y_test: pd.Series, is_classification: bool = True,
                      threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate model performance with comprehensive metrics for imbalanced classification.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: True labels
            is_classification: Whether this is a classification task
            threshold: Classification threshold (for binary classification only)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        if is_classification:
            # Get probability predictions
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Apply threshold to get binary predictions
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Calculate metrics
            accuracy = (y_pred == y_test).mean()
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Calculate precision-recall curve and average precision
            precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            
            # Find optimal threshold based on F1 score
            f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-7)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds_pr[optimal_idx] if optimal_idx < len(thresholds_pr) else threshold
            
            # Make predictions with optimal threshold
            y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
            
            # Print classification reports
            print("\n" + "="*80)
            print(f"EVALUATION WITH DEFAULT THRESHOLD ({threshold:.4f})")
            print("="*80)
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, zero_division=0))
            
            print(f"\nROC AUC: {roc_auc:.4f}")
            
            print("\n" + "="*80)
            print(f"EVALUATION WITH OPTIMAL THRESHOLD ({optimal_threshold:.4f})")
            print("="*80)
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred_optimal, zero_division=0))
            
            # Print confusion matrices
            print("\n=== Confusion Matrices ===")
            print("\nDefault Threshold (0.5):")
            print(pd.crosstab(y_test, y_pred, 
                            rownames=['Actual'], 
                            colnames=['Predicted'], 
                            normalize='all'))
            
            print("\nOptimal Threshold:")
            print(pd.crosstab(y_test, y_pred_optimal,
                            rownames=['Actual'],
                            colnames=['Predicted'],
                            normalize='all'))
            
            # Plot ROC and PR curves
            self._plot_roc_pr_curves(y_test, y_pred_proba, roc_auc, avg_precision, fpr, tpr, precision_curve, recall_curve)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'average_precision': avg_precision,
                'optimal_threshold': optimal_threshold,
                'f1_optimal': f1_score(y_test, y_pred_optimal, zero_division=0),
                'recall_optimal': recall_score(y_test, y_pred_optimal, zero_division=0),
                'precision_optimal': precision_score(y_test, y_pred_optimal, zero_division=0)
            }
        else:
            # Regression metrics
            y_pred = model.predict(X_test)
            return {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'y_true': y_test.values.tolist(),
                'y_pred': y_pred.tolist()
            }
    
    def _plot_roc_pr_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                          roc_auc: float, avg_precision: float,
                          fpr: np.ndarray, tpr: np.ndarray,
                          precision_curve: np.ndarray, recall_curve: np.ndarray) -> None:
        """
        Plot ROC and Precision-Recall curves.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            roc_auc: ROC AUC score
            avg_precision: Average precision score
            fpr: False Positive Rate values for ROC curve
            tpr: True Positive Rate values for ROC curve
            precision_curve: Precision values for PR curve
            recall_curve: Recall values for PR curve
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot ROC curve
        ax1.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax1.legend(loc="lower right")
        
        # Plot Precision-Recall curve
        ax2.step(recall_curve, precision_curve, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.4f})', where='post')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc="lower left")
        
        # Add no-skill line for PR curve (baseline for imbalanced data)
        no_skill = len(y_true[y_true==1]) / len(y_true)
        ax2.axhline(y=no_skill, color='navy', linestyle='--', 
                   label=f'No Skill (AP = {no_skill:.4f})')
        
        plt.tight_layout()
        
        # Save the figure
        os.makedirs('results/plots', exist_ok=True)
        plt.savefig('results/plots/roc_pr_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nROC and Precision-Recall curves saved to 'results/plots/roc_pr_curves.png'")
        
        # Plot probability distributions
        self._plot_probability_distributions(y_true, y_pred_proba)
    
    def _plot_probability_distributions(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> None:
        """Plot probability distributions for positive and negative classes.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
        """
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot histograms for each class
        plt.hist(y_pred_proba[y_true == 0], 
                bins=50, 
                alpha=0.5, 
                color='blue', 
                label='Negative Class (No Claim)',
                density=True)
        
        plt.hist(y_pred_proba[y_true == 1], 
                bins=50, 
                alpha=0.5, 
                color='red', 
                label='Positive Class (Claim)',
                density=True)
        
        # Add vertical line at 0.5 threshold
        plt.axvline(0.5, color='black', linestyle='--', label='Default Threshold (0.5)')
        
        # Calculate and plot optimal threshold
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        plt.axvline(optimal_threshold, color='green', linestyle='--', 
                   label=f'Optimal Threshold ({optimal_threshold:.4f})')
        
        # Add labels and title
        plt.xlabel('Predicted Probability of Positive Class')
        plt.ylabel('Density')
        plt.title('Predicted Probability Distributions by True Class')
        plt.legend()
        
        # Save the figure
        os.makedirs('results/plots', exist_ok=True)
        plt.savefig('results/plots/probability_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Probability distributions plot saved to 'results/plots/probability_distributions.png'")
    
    def analyze_feature_importance(self, model: Any, X_test: pd.DataFrame, 
                             feature_names: list[str], output_dir: str = 'results') -> None:
        """
        Generate and save feature importance plots.
        
        Args:
            model: Trained model
            X_test: Test features for SHAP analysis
            feature_names: List of feature names
            output_dir: Directory to save the output plots
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # SHAP analysis
            explainer = shap.TreeExplainer(model)
            
            # For XGBoost, we need to handle the output format
            if hasattr(model, 'predict_proba'):
                shap_values = explainer.shap_values(X_test)
                # If it's a binary classifier, take the second dimension
                if isinstance(shap_values, list) and len(shap_values) > 1:
                    shap_values = shap_values[1]
            else:
                shap_values = explainer.shap_values(X_test)
            
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values, 
                X_test, 
                feature_names=feature_names,
                show=False,
                plot_type="bar",
                max_display=min(20, len(feature_names))  # Limit number of features for better visualization
            )
            plt.tight_layout()
            plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Also save a detailed SHAP summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values,
                X_test,
                feature_names=feature_names,
                show=False,
                plot_type="dot"
            )
            plt.tight_layout()
            plt.savefig(f'{output_dir}/shap_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("Successfully generated feature importance plots")
            
        except Exception as e:
            print(f"Error generating feature importance plots: {str(e)}")
        
    def save_model(self, model, filepath):
        """Save the trained model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath: str) -> Any:
        """Load trained model from disk."""
        return joblib.load(filepath)
