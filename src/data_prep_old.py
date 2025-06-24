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
        print("\nStarting feature creation...")
        
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # 1. Time-based features
        try:
            if 'TransactionMonth' in df.columns:
                print("Creating time-based features...")
                current_date = pd.Timestamp.now()
                
                # Convert to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(df['TransactionMonth']):
                    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
                
                # Basic date features
                df['transaction_year'] = df['TransactionMonth'].dt.year.astype('int16')
                df['transaction_month'] = df['TransactionMonth'].dt.month.astype('int8')
                df['transaction_quarter'] = df['TransactionMonth'].dt.quarter.astype('int8')
                df['transaction_weekday'] = df['TransactionMonth'].dt.weekday.astype('int8')
                
                # Days since policy start (if registration year is available)
                if 'RegistrationYear' in df.columns:
                    df['policy_age_years'] = (df['TransactionMonth'].dt.year - df['RegistrationYear']).clip(0, 50).astype('int8')
                
                # Is weekend flag
                df['is_weekend'] = (df['transaction_weekday'] >= 5).astype('int8')
                
                # Seasonality
                df['is_q4'] = (df['transaction_quarter'] == 4).astype('int8')
                
        except Exception as e:
            print(f"Warning: Could not create time-based features: {e}")
        
        # 2. Vehicle features
        try:
            print("Creating vehicle features...")
            # Vehicle age
            if 'RegistrationYear' in df.columns:
                current_year = pd.Timestamp.now().year
                df['vehicle_age'] = (current_year - df['RegistrationYear']).clip(0, 50).astype('int8')
            
            # Power-related features
            if 'kilowatts' in df.columns and 'cubiccapacity' in df.columns:
                # Power to weight ratio (kW/cc)
                df['power_to_weight'] = (df['kilowatts'] / (df['cubiccapacity'] + 1e-6)).astype('float32')
                
                # Engine efficiency (kW per liter)
                df['engine_efficiency'] = (df['kilowatts'] / (df['cubiccapacity'] / 1000 + 1e-6)).astype('float32')
            
            # Value-related features
            if 'SumInsured' in df.columns:
                # Log transform for monetary values
                df['log_suminsured'] = np.log1p(df['SumInsured']).astype('float32')
                
                # Premium per unit sum insured
                if 'TotalPremium' in df.columns:
                    df['premium_ratio'] = (df['TotalPremium'] / (df['SumInsured'] + 1e-6)).astype('float32')
            
        except Exception as e:
            print(f"Warning: Could not create vehicle features: {e}")
        
        # 3. Policy features
        try:
            print("Creating policy features...")
            if 'TotalPremium' in df.columns and 'SumInsured' in df.columns:
                df['premium_to_sum_insured_ratio'] = (df['TotalPremium'] / (df['SumInsured'] + 1e-6)).astype('float32')
            
            if 'TotalPremium' in df.columns and 'ExcessSelected' in df.columns:
                # Excess as percentage of premium
                df['excess_ratio'] = (df['ExcessSelected'] / (df['TotalPremium'] + 1e-6)).astype('float32')
                
                # Binary flag for high excess
                df['high_excess'] = (df['excess_ratio'] > 0.5).astype('int8')
            
            # Create interaction between vehicle age and type
            if 'VehicleType' in df.columns and 'vehicle_age' in df.columns:
                df['vehicle_type_age'] = df['VehicleType'].astype(str) + '_' + df['vehicle_age'].astype(str)
            
        except Exception as e:
            print(f"Warning: Could not create policy features: {e}")
        
        # 4. Geographic features (if available)
        try:
            if 'SubCrestaZone' in df.columns:
                print("Creating geographic features...")
                # Create risk zone categories (example: split into high/medium/low)
                zone_counts = df['SubCrestaZone'].value_counts()
                df['is_common_zone'] = df['SubCrestaZone'].isin(zone_counts[zone_counts > 100].index).astype('int8')
        except Exception as e:
            print(f"Warning: Could not create geographic features: {e}")
        
        # 5. Risk scores for categorical variables (if target is available)
        if hasattr(self, 'target_encoder') and 'HasClaim' in df.columns:
            try:
                print("Creating risk scores for categorical variables...")
                for col in self.categorical_cols:
                    if col in df.columns:
                        # Calculate mean target per category with smoothing
                        agg = df.groupby(col)['HasClaim'].agg(['count', 'mean'])
                        m = 100  # Smoothing parameter
                        global_mean = df['HasClaim'].mean()
                        
                        # Calculate smoothed mean
                        agg['smooth_mean'] = (agg['count'] * agg['mean'] + m * global_mean) / (agg['count'] + m)
                        self.target_encoder[col] = agg['smooth_mean'].to_dict()
                        
                        # Apply encoding
                        encoded_col = f"{col}_risk_score"
                        df[encoded_col] = df[col].map(self.target_encoder[col]).astype('float32')
            except Exception as e:
                print(f"Warning: Could not create risk scores: {e}")
        
        # 6. Optimize memory usage
        print("Optimizing memory usage...")
        df = self.optimize_memory(df)
        
        # 7. Print summary of new features
        print(f"\nFeature creation complete. Final shape: {df.shape}")
        
        # Identify and report new features
        new_features = [col for col in df.columns if col not in self.original_columns]
        if new_features:
            print(f"\nCreated {len(new_features)} new features:")
            for i, feat in enumerate(sorted(new_features), 1):
                print(f"{i}. {feat} (dtype: {df[feat].dtype})")
                
            # Print sample values for a few new features
            print("\nSample values for new features:")
            for feat in sorted(new_features)[:5]:  # Limit to first 5 features
                sample = df[feat].dropna()
                if not sample.empty:
                    print(f"{feat}: {sample.iloc[0]}")
                else:
                    print(f"{feat}: All values are NA")
        
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
        print("Optimizing memory usage...")
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Downcast numeric types
        for col in df.select_dtypes(include=['int64', 'int32']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
            
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert object columns with low cardinality to category
        for col in df.select_dtypes(include=['object']).columns:
            num_unique = len(df[col].unique())
            num_total = len(df[col])
            if num_unique / num_total < 0.5:  # If less than 50% unique values
                df[col] = df[col].astype('category')
        
        return df
    
    def split_data(self, df: pd.DataFrame, target_col: str = 'HasClaim', 
               test_size: float = 0.2, random_state: int = 42) -> tuple:
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
            # Store the target column name for later use
            self.target_col = target_col
            
            # Store the full dataset before splitting
            self.df = df.copy()
            
            # Split into features and target
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Store numerical and categorical columns
            self.numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Store the split data as instance variables
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state,
                stratify=y if len(y.unique()) > 1 else None
            )
            
            print(f"Training set size: {len(self.X_train):,} samples")
            print(f"Test set size: {len(self.X_test):,} samples")
            
            return self.X_train, self.X_test, self.y_train, self.y_test
            
        except Exception as e:
            print(f"Error during train-test split: {str(e)}")
            raise

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
                self.scaler = StandardScaler()
                self.feature_names = None
                
            def fit(self, X, y=None):
                # Process in chunks to save memory
                chunk_size = 10000
                
                # First pass: fit imputer
                for i in range(0, len(X), chunk_size):
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
        
        numerical_transformer = OptimizedNumericalTransformer()
        
        # Categorical pipeline with minimal memory usage
        print("Creating optimized categorical pipeline...")
        
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
        
        categorical_transformer = OptimizedCategoricalTransformer()
        
        # Combine pipelines
        print("\nCreating ColumnTransformer...")
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='drop',  # Drop columns not specified in transformers
            verbose_feature_names_out=False
        )
        
        print("\n=== Preprocessing pipeline created successfully ===")
        return preprocessor

