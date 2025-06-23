import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from typing import Dict, List, Optional, Tuple, Union

class InsuranceRiskAnalyzer:
    def __init__(self, data_path: str, output_dir: str = 'results', 
             min_sample_size: int = 30, alpha: float = 0.05):
        """
        Initialize the analyzer with data and configuration.
        
        Args:
            data_path: Path to the insurance data file
            output_dir: Directory to save results and visualizations
            min_sample_size: Minimum sample size for analysis
            alpha: Significance level for statistical tests
        """
        # Read the data with low_memory=False to handle mixed types
        self.df = pd.read_csv(data_path, low_memory=False)
                
        # Standardize column names (trim whitespace and convert to lowercase)
        self.df.columns = self.df.columns.str.strip().str.lower()
        
        self.output_dir = output_dir
        self.min_sample_size = min_sample_size
        self.alpha = alpha
        self._setup_directories()
        
    def _setup_directories(self):
        """Create necessary directories for outputs."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'visualizations'), exist_ok=True)
        
    def analyze_risk_differences(self, group_col: str, value_col: str = 'TotalClaims',
                               freq_col: str = 'HasClaim', title: str = None):
        """
        Analyze risk differences across groups.
        
        Args:
            group_col: Column name to group by (e.g., 'Province', 'PostalCode')
            value_col: Column with claim amounts
            freq_col: Column indicating claim occurrence (0/1)
            title: Title for the analysis
        """
        print(f"\n{'='*80}")
        print(f"ANALYZING RISK DIFFERENCES BY {group_col.upper()}")
        print("="*80)
        
        # 1. Claim Frequency Analysis
        self._analyze_claim_frequency(group_col, freq_col)
        
        # 2. Claim Severity Analysis
        self._analyze_claim_severity(group_col, value_col, title)
        
    def _analyze_claim_frequency(self, group_col: str, freq_col: str):
        """Analyze claim frequency differences across groups."""
        # Create contingency table
        contingency = pd.crosstab(self.df[group_col], self.df[freq_col])
        
        # Check expected frequencies
        min_expected = stats.contingency.expected_freq(contingency).min()
        if min_expected < 5:
            print("\nWarning: Some expected frequencies < 5. Consider grouping categories.")
            
        # Perform Chi-Square Test
        chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
        print(f"\nChi-Square Test for Claim Frequency:")
        print(f"Chi2 = {chi2:.4f}, p-value = {p_value:.6f}")
        print(f"Degrees of freedom: {dof}")
        
        # Save results
        results = {
            'test': 'Chi-Square',
            'chi2': chi2,
            'p_value': p_value,
            'dof': dof,
            'significant': p_value < self.alpha,
            'effect_size': 'Cramer\'s V: ' + str(np.sqrt(chi2 / (len(self.df) * (min(contingency.shape) - 1))))
        }
        
        self._save_results(f'frequency_{group_col}.json', results)
        
    def _analyze_claim_severity(self, group_col: str, value_col: str, title: str = None):
        """Analyze claim severity differences across groups."""
        df_claims = self.df[self.df[value_col] > 0].copy()
        
        if len(df_claims) < self.min_sample_size:
            print(f"\nInsufficient data for severity analysis (n = {len(df_claims)})")
            return
            
        # Check normality
        self._check_normality(df_claims, group_col, value_col)
        
        # Kruskal-Wallis test
        groups = [group[value_col].values for name, group in df_claims.groupby(group_col)]
        h_stat, p_value = stats.kruskal(*groups)
        
        print(f"\nKruskal-Wallis Test for Claim Severity:")
        print(f"H = {h_stat:.4f}, p-value = {p_value:.6f}")
        
        # Effect size
        n = len(df_claims)
        k = len(groups)
        eta_squared = (h_stat - k + 1) / (n - k) if n > k else 0
        
        # Save results
        results = {
            'test': 'Kruskal-Wallis',
            'h_statistic': h_stat,
            'p_value': p_value,
            'effect_size': eta_squared,
            'significant': p_value < self.alpha
        }
        
        self._save_results(f'severity_{group_col}.json', results)
        
        # Post-hoc analysis if significant
        if p_value < self.alpha and k > 2:
            self._perform_posthoc_analysis(df_claims, group_col, value_col, title)
            
        # Create visualization
        self._create_severity_plot(df_claims, group_col, value_col, title)
        
    def _check_normality(self, df: pd.DataFrame, group_col: str, value_col: str):
        """Check normality of claim amounts within each group."""
        print("\nNormality Test (Shapiro-Wilk) for Claim Amounts:")
        for group_name, group_data in df.groupby(group_col):
            if len(group_data) >= 3:  # Shapiro-Wilk requires at least 3 samples
                stat, p = stats.shapiro(group_data[value_col])
                print(f"{group_name}: W = {stat:.4f}, p = {p:.6f}")

    def _perform_posthoc_analysis(self, df: pd.DataFrame, group_col: str, 
                                value_col: str, title: str = None):
        """Perform post-hoc analysis using Tukey's HSD."""
        print("\nPerforming post-hoc analysis (Tukey's HSD):")
        posthoc = pairwise_tukeyhsd(
            endog=df[value_col],
            groups=df[group_col],
            alpha=self.alpha
        )
        print(posthoc.summary())
        
        # Save posthoc results
        posthoc_df = pd.DataFrame(
            data=posthoc._results_table.data[1:],
            columns=posthoc._results_table.data[0]
        )
        posthoc_df.to_csv(
            os.path.join(self.output_dir, f'posthoc_{group_col}.csv'),
            index=False
        )
        
    def _create_severity_plot(self, df: pd.DataFrame, group_col: str, 
                            value_col: str, title: str = None):
        """Create visualization of claim severity by group."""
        plt.figure(figsize=(14, 6))
        
        # Boxplot for distribution
        sns.boxplot(x=group_col, y=value_col, data=df)
        
        # Formatting
        title = f'Claim Severity by {group_col}' if not title else title
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure
        safe_group = group_col.replace(' ', '_').lower()
        plt.savefig(os.path.join(self.output_dir, f'visualizations/severity_{safe_group}.png'))
        plt.close()
        
    def _save_results(self, filename: str, results: Dict):
        """Save analysis results to a JSON file with proper type conversion."""
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            return obj

        # Convert all numpy types in the results
        results = convert_numpy_types(results)
        
        # Ensure the directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save to file
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filepath}")
        
# Update the column names in the main block to use lowercase
if __name__ == "__main__":
    try:
        # Read the data with low_memory=False to handle mixed types
        df = pd.read_csv('data/cleaned_insurance_data.csv', low_memory=False)
        
        # Initialize analyzer
        analyzer = InsuranceRiskAnalyzer(
            data_path='data/cleaned_insurance_data.csv',
            output_dir='results',
            min_sample_size=30,
            alpha=0.05
        )
        
        # 1. Analyze by Province
        print("\n" + "="*80)
        print("ANALYZING RISK BY PROVINCE")
        print("="*80)
        analyzer.analyze_risk_differences(
            group_col='province',  # Changed to lowercase
            value_col='totalclaims',  # Changed to lowercase
            freq_col='hasclaim',  # Changed to lowercase
            title='Claim Analysis by Province'
        )
        
        # 2. Analyze by Gender
        print("\n" + "="*80)
        print("ANALYZING RISK BY GENDER")
        print("="*80)
        analyzer.analyze_risk_differences(
            group_col='gender',  # Changed to lowercase
            value_col='totalclaims',  # Changed to lowercase
            freq_col='hasclaim',  # Changed to lowercase
            title='Claim Analysis by Gender'
        )
        
        # 3. Analyze by Vehicle Type
        print("\n" + "="*80)
        print("ANALYZING RISK BY VEHICLE TYPE")
        print("="*80)
        analyzer.analyze_risk_differences(
            group_col='vehicletype',  # Changed to lowercase
            value_col='totalclaims',  # Changed to lowercase
            freq_col='hasclaim',  # Changed to lowercase
            title='Claim Analysis by Vehicle Type'
        )
        
        # 4. For Postal Code analysis
        print("\n" + "="*80)
        print("ANALYZING RISK BY POSTAL CODE")
        print("="*80)
        # Let's analyze only the top 20 most common zip codes
        top_zip_codes = df['PostalCode'].value_counts().nlargest(20).index
        df_top_zips = df[df['PostalCode'].isin(top_zip_codes)].copy()
        
        # Save this filtered dataset temporarily
        temp_csv = 'data/temp_top_zip_codes.csv'
        df_top_zips.to_csv(temp_csv, index=False)
        
        # Analyze top zip codes
        zip_analyzer = InsuranceRiskAnalyzer(
            data_path=temp_csv,
            output_dir='results/postal_codes',
            min_sample_size=10,  # Lower threshold due to smaller groups
            alpha=0.05
        )
        zip_analyzer.analyze_risk_differences(
            group_col='postalcode',  # Changed to lowercase
            value_col='totalclaims',  # Changed to lowercase
            freq_col='hasclaim',  # Changed to lowercase
            title='Claim Analysis by Top 20 Postal Codes'
        )
        
        # Clean up
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
            
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
