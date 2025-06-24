import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

class OptimizedNumericalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputer = SimpleImputer(strategy='constant', fill_value=0)
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def fit(self, X, y=None):
        # Process in chunks to save memory
        chunk_size = 10000
        
        # First pass: fit imputer
        for i in range(0, len(X), min(chunk_size, len(X))):
            chunk = X.iloc[i:i + chunk_size]
            chunk_num = chunk.apply(pd.to_numeric, errors='coerce').fillna(0)
            self.imputer.fit(chunk_num)
            break  # Only need one chunk to fit the imputer
        
        # Second pass: fit scaler
        for i in range(0, len(X), chunk_size):
            chunk = X.iloc[i:i + chunk_size]
            chunk_num = chunk.apply(pd.to_numeric, errors='coerce').fillna(0)
            chunk_imputed = self.imputer.transform(chunk_num)
            self.scaler.partial_fit(chunk_imputed)
        
        self.feature_names = X.columns.tolist()
        return self
        
    def transform(self, X):
        # Process in chunks to save memory
        results = []
        chunk_size = 10000
        
        for i in range(0, len(X), chunk_size):
            chunk = X.iloc[i:i + chunk_size]
            chunk_num = chunk.apply(pd.to_numeric, errors='coerce').fillna(0)
            chunk_imputed = self.imputer.transform(chunk_num)
            chunk_scaled = self.scaler.transform(chunk_imputed)
            results.append(chunk_scaled)
        
        return np.vstack(results)

class OptimizedCategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputer = SimpleImputer(strategy='constant', fill_value='missing')
        self.onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        self.feature_names = None
        self.categories_ = None
        
    def fit(self, X, y=None):
        # Process in chunks to save memory
        chunk_size = 10000
        
        # First pass: fit imputer
        for i in range(0, len(X), min(chunk_size, len(X))):
            chunk = X.iloc[i:i + chunk_size]
            chunk_str = chunk.astype(str)
            self.imputer.fit(chunk_str)
            break  # Only need one chunk to fit the imputer
        
        # Collect all unique categories by processing in chunks
        unique_categories = {}
        for i in range(0, len(X), chunk_size):
            chunk = X.iloc[i:i + chunk_size]
            chunk_str = chunk.astype(str)
            chunk_imputed = self.imputer.transform(chunk_str)
            
            # Update unique categories for each column
            for j, col in enumerate(X.columns):
                unique_vals = set(chunk_imputed[:, j])
                if j not in unique_categories:
                    unique_categories[j] = unique_vals
                else:
                    unique_categories[j].update(unique_vals)
        
        # Convert to sorted lists for OneHotEncoder
        self.categories_ = [sorted(unique_categories[i]) for i in range(len(X.columns))]
        
        # Fit OneHotEncoder with all categories
        self.onehot = OneHotEncoder(
            categories=self.categories_,
            handle_unknown='ignore',
            sparse_output=True
        )
        
        # Fit with a small sample (needed for scikit-learn compatibility)
        sample = pd.DataFrame({
            col: [self.categories_[i][0]] if self.categories_[i] else ['missing']
            for i, col in enumerate(X.columns)
        })
        self.onehot.fit(sample.astype(str))
        
        self.feature_names = X.columns.tolist()
        return self
        
    def transform(self, X):
        # Process in chunks to save memory
        results = []
        chunk_size = 10000
        
        for i in range(0, len(X), chunk_size):
            chunk = X.iloc[i:i + chunk_size]
            chunk_str = chunk.astype(str)
            chunk_imputed = self.imputer.transform(chunk_str)
            chunk_encoded = self.onehot.transform(chunk_imputed)
            results.append(chunk_encoded)
        
        # Convert to sparse matrix and concatenate
        from scipy.sparse import vstack
        return vstack(results) if results else None

class DataPreprocessor:
    def __init__(self, config: Dict[str, any]):
        """
        Initialize the DataPreprocessor with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.numerical_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.target_encoder = {}
        self.categorical_cols = [
            'Province', 'VehicleType', 'make', 'Model', 'bodytype',
            'MainCrestaZone', 'SubCrestaZone', 'CoverType', 'CoverCategory'
        ]
        self.original_columns = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, filepath: str, sample_size: float = 1.0) -> pd.DataFrame:
        """
        Load and preprocess the dataset with memory optimization.
        
        Args:
            filepath: Path to the CSV file
            sample_size: Fraction of data to sample (0.0 to 1.0)
            
        Returns:
            Preprocessed DataFrame
        """
        print(f"Loading data from: {filepath}")
        
        # Define optimized dtypes for known columns
        dtype_map = {
            'underwrittencoverid': 'int32',
            'policyid': 'int32',
            'registrationyear': 'int16',
            'cubiccapacity': 'float32',
            'kilowatts': 'float32',
            'suminsured': 'float32',
            'numberofdoors': 'int8',
            'totalpremium': 'float32',
            'totalclaims': 'int8',
            'termfrequency': 'int8',
            'calculatedpremiumperterm': 'float32',
            'excessselected': 'float32',
        }
        
        # Read the data with optimized dtypes
        try:
            df = pd.read_csv(filepath, dtype=dtype_map, low_memory=False)
            
            # Sample the data if needed
            if sample_size < 1.0:
                df = df.sample(frac=sample_size, random_state=42)
                
            # Store original columns for reference
            self.original_columns = set(df.columns)
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from the raw data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new features
        """
        print("\nCreating features...")
        
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Create power-to-weight ratio feature if both columns exist
        if 'kilowatts' in df.columns and 'cubiccapacity' in df.columns:
            # Avoid division by zero and handle missing values
            df['power_to_weight'] = np.where(
                (df['cubiccapacity'] > 0) & (df['kilowatts'].notna()),
                df['kilowatts'] / df['cubiccapacity'],
                np.nan
            )
            # Fill any remaining NaN values with 0 or another appropriate value
            df['power_to_weight'] = df['power_to_weight'].fillna(0)
            print("Created 'power_to_weight' feature")
        
        # Add more feature engineering steps here as needed
        
        return df
    
    def optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by downcasting numeric types
        and converting object columns to category where appropriate.
        
        Args:
            df: Input DataFrame to optimize
            
        Returns:
            Memory-optimized DataFrame
        """
        print("\nOptimizing memory usage...")
        
        # Make a copy of the dataframe
        df = df.copy()
        
        # Downcast numeric columns
        for col in df.select_dtypes(include=['int64', 'int32', 'float64']):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # Convert object columns to category where appropriate
        for col in df.select_dtypes(include=['object']):
            if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
                df[col] = df[col].astype('category')
        
        return df
    
    def split_data(self, df: pd.DataFrame, target_col: str = 'HasClaim', 
                   test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column
            test_size: Size of the test set (0-1)
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("\nSplitting data into train and test sets...")
        
        try:
            # Separate features and target
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Split the data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            print(f"Training set size: {len(self.X_train)} samples")
            print(f"Test set size: {len(self.X_test)} samples")
            
            return self.X_train, self.X_test, self.y_train, self.y_test
            
        except Exception as e:
            print(f"Error during train-test split: {str(e)}")
            raise
    
    def get_preprocessor(self, numerical_cols: List[str] = None, 
                       categorical_cols: List[str] = None) -> ColumnTransformer:
        """
        Create preprocessing pipeline with memory-efficient handling of numeric and categorical data.
        Processes data in chunks to reduce memory usage.
        
        Args:
            numerical_cols: List of numerical column names
            categorical_cols: List of categorical column names
            
        Returns:
            Configured ColumnTransformer
        """
        print("\nCreating preprocessing pipeline...")
        
        # Ensure we have column lists
        if numerical_cols is None:
            numerical_cols = []
        if categorical_cols is None:
            categorical_cols = []
            
        print(f"Numerical columns: {len(numerical_cols)}")
        print(f"Categorical columns: {len(categorical_cols)}")
        
        # Create transformers
        numeric_transformer = OptimizedNumericalTransformer()
        categorical_transformer = OptimizedCategoricalTransformer()
        
        # Create column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='drop',
            n_jobs=-1
        )
        
        print("Preprocessing pipeline created!")
        return preprocessor
