#!/usr/bin/env python
"""
Comprehensive Data Cleaning Tool
This tool automates the data cleaning process to replace manual human intervention
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    """Comprehensive data cleaning tool"""
    
    def __init__(self):
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        self.imputers = {}
        self.scalers = {}
        self.label_encoders = {}
        self.outlier_bounds = {}
        self.cleaning_report = {}
    
    def identify_column_types(self, df):
        """Identify column types automatically"""
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_columns = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
        
        print(f"Identified column types:")
        print(f"  Numeric: {self.numeric_columns}")
        print(f"  Categorical: {self.categorical_columns}")
        print(f"  DateTime: {self.datetime_columns}")
        
        return self.numeric_columns, self.categorical_columns, self.datetime_columns
    
    def handle_missing_values(self, df, strategy='auto'):
        """
        Handle missing values in the dataset
        strategy: 'auto', 'mean', 'median', 'mode', 'drop', 'forward', 'backward'
        """
        print("\nHandling missing values...")
        initial_missing = df.isnull().sum().sum()
        
        if initial_missing == 0:
            print("  No missing values found.")
            return df
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count == 0:
                continue
                
            print(f"  Processing '{column}' ({missing_count} missing values)")
            
            if column in self.numeric_columns:
                if strategy == 'auto':
                    # Use mean for normally distributed data, median for skewed data
                    skewness = df[column].skew()
                    method = 'mean' if abs(skewness) < 1 else 'median'
                else:
                    method = strategy
                    
                if method == 'mean':
                    df[column].fillna(df[column].mean(), inplace=True)
                elif method == 'median':
                    df[column].fillna(df[column].median(), inplace=True)
                elif method == 'drop':
                    df.dropna(subset=[column], inplace=True)
                elif method == 'forward':
                    df[column].fillna(method='ffill', inplace=True)
                elif method == 'backward':
                    df[column].fillna(method='bfill', inplace=True)
                    
            elif column in self.categorical_columns:
                if strategy == 'auto' or strategy == 'mode':
                    mode_value = df[column].mode()
                    if not mode_value.empty:
                        df[column].fillna(mode_value[0], inplace=True)
                elif strategy == 'drop':
                    df.dropna(subset=[column], inplace=True)
                    
        final_missing = df.isnull().sum().sum()
        print(f"  Missing values reduced from {initial_missing} to {final_missing}")
        
        self.cleaning_report['missing_values_handled'] = initial_missing - final_missing
        return df
    
    def detect_outliers(self, df, method='iqr', threshold=1.5):
        """
        Detect outliers in numeric columns
        method: 'iqr', 'zscore', 'percentile'
        """
        print(f"\nDetecting outliers using {method} method...")
        outliers_summary = {}
        
        for column in self.numeric_columns:
            if method == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[column]))
                outliers = df[z_scores > threshold]
                
            elif method == 'percentile':
                lower_bound = df[column].quantile(threshold)
                upper_bound = df[column].quantile(1 - threshold)
                outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            
            outlier_count = len(outliers)
            outliers_summary[column] = {
                'count': outlier_count,
                'percentage': (outlier_count / len(df)) * 100
            }
            
            if outlier_count > 0:
                print(f"  {column}: {outlier_count} outliers ({outliers_summary[column]['percentage']:.2f}%)")
        
        self.cleaning_report['outliers_detected'] = outliers_summary
        return outliers_summary
    
    def handle_outliers(self, df, method='iqr', threshold=1.5, action='cap'):
        """
        Handle outliers in numeric columns
        action: 'cap', 'remove', 'transform'
        """
        print(f"\nHandling outliers using {action} method...")
        initial_rows = len(df)
        
        for column in self.numeric_columns:
            if method == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
            elif method == 'zscore':
                mean = df[column].mean()
                std = df[column].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
                
            if action == 'cap':
                # Cap outliers to boundary values
                df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
                df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
                
            elif action == 'remove':
                # Remove rows with outliers
                df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
                
            elif action == 'transform':
                # Apply log transformation for positive values
                if df[column].min() > 0:
                    df[column] = np.log1p(df[column])
        
        final_rows = len(df)
        print(f"  Rows reduced from {initial_rows} to {final_rows}")
        
        self.cleaning_report['outliers_handled'] = initial_rows - final_rows
        return df
    
    def remove_duplicates(self, df, subset=None):
        """Remove duplicate rows"""
        print("\nRemoving duplicates...")
        initial_rows = len(df)
        
        if subset:
            df = df.drop_duplicates(subset=subset)
        else:
            df = df.drop_duplicates()
            
        final_rows = len(df)
        duplicates_removed = initial_rows - final_rows
        
        print(f"  Duplicates removed: {duplicates_removed}")
        self.cleaning_report['duplicates_removed'] = duplicates_removed
        
        return df
    
    def encode_categorical_variables(self, df, method='label'):
        """
        Encode categorical variables
        method: 'label', 'onehot', 'frequency'
        """
        print(f"\nEncoding categorical variables using {method} encoding...")
        
        for column in self.categorical_columns:
            print(f"  Processing '{column}'")
            
            if method == 'label':
                # Label encoding
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column].astype(str))
                self.label_encoders[column] = le
                
            elif method == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(df[column], prefix=column)
                df = pd.concat([df, dummies], axis=1)
                df.drop(column, axis=1, inplace=True)
                
            elif method == 'frequency':
                # Frequency encoding
                freq_map = df[column].value_counts().to_dict()
                df[column] = df[column].map(freq_map)
        
        print(f"  Categorical variables encoded: {len(self.categorical_columns)}")
        self.cleaning_report['categorical_encoded'] = len(self.categorical_columns)
        
        return df
    
    def standardize_numeric_features(self, df):
        """Standardize numeric features"""
        print("\nStandardizing numeric features...")
        
        scaler = StandardScaler()
        df[self.numeric_columns] = scaler.fit_transform(df[self.numeric_columns])
        self.scalers['standard'] = scaler
        
        print(f"  Features standardized: {len(self.numeric_columns)}")
        self.cleaning_report['features_standardized'] = len(self.numeric_columns)
        
        return df
    
    def clean_data(self, df, 
                   missing_strategy='auto',
                   outlier_method='iqr',
                   outlier_threshold=1.5,
                   outlier_action='cap',
                   remove_duplicates=True,
                   encode_categorical='label',
                   standardize=True):
        """
        Complete data cleaning pipeline
        """
        print("STARTING DATA CLEANING PROCESS")
        print("=" * 40)
        
        # Store original shape
        original_shape = df.shape
        print(f"Original dataset shape: {original_shape}")
        
        # Identify column types
        self.identify_column_types(df)
        
        # Handle missing values
        df = self.handle_missing_values(df, missing_strategy)
        
        # Detect outliers
        self.detect_outliers(df, outlier_method, outlier_threshold)
        
        # Handle outliers
        df = self.handle_outliers(df, outlier_method, outlier_threshold, outlier_action)
        
        # Remove duplicates
        if remove_duplicates:
            df = self.remove_duplicates(df)
        
        # Encode categorical variables
        if encode_categorical and len(self.categorical_columns) > 0:
            df = self.encode_categorical_variables(df, encode_categorical)
        
        # Standardize numeric features
        if standardize and len(self.numeric_columns) > 0:
            df = self.standardize_numeric_features(df)
        
        # Final report
        final_shape = df.shape
        print("\n" + "=" * 40)
        print("DATA CLEANING COMPLETE")
        print("=" * 40)
        print(f"Final dataset shape: {final_shape}")
        print(f"Rows removed: {original_shape[0] - final_shape[0]}")
        print(f"Columns changed: {original_shape[1] - final_shape[1]}")
        
        return df
    
    def generate_cleaning_report(self):
        """Generate a detailed cleaning report"""
        print("\nCLEANING REPORT")
        print("=" * 20)
        for key, value in self.cleaning_report.items():
            if key == 'outliers_detected':
                print(f"{key}:")
                for col, stats in value.items():
                    print(f"  {col}: {stats['count']} ({stats['percentage']:.2f}%)")
            else:
                print(f"{key}: {value}")
        
        return self.cleaning_report

# Example usage with house price data
def create_sample_house_data():
    """Create sample house price data with issues for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    # Create base data
    size = np.random.normal(2000, 500, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.randint(1, 4, n_samples)
    age = np.random.randint(0, 50, n_samples)
    location_score = np.random.uniform(1, 10, n_samples)
    
    # Create price based on features
    base_price = (
        size * 100 +
        bedrooms * 5000 +
        bathrooms * 7000 +
        (50 - age) * 1000 +
        location_score * 10000
    )
    
    # Add noise
    noise = np.random.normal(0, 20000, n_samples)
    price = base_price + noise
    
    # Create DataFrame
    data = pd.DataFrame({
        'size': size,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age': age,
        'location_score': location_score,
        'price': price
    })
    
    # Introduce some data quality issues
    # 1. Add missing values
    missing_indices = np.random.choice(n_samples, size=50, replace=False)
    data.loc[missing_indices[:25], 'size'] = np.nan
    data.loc[missing_indices[25:], 'location_score'] = np.nan
    
    # 2. Add outliers
    outlier_indices = np.random.choice(n_samples, size=30, replace=False)
    data.loc[outlier_indices[:15], 'price'] = data.loc[outlier_indices[:15], 'price'] * 5  # Extreme high prices
    data.loc[outlier_indices[15:], 'size'] = data.loc[outlier_indices[15:], 'size'] * 3   # Extreme large houses
    
    # 3. Add duplicates
    duplicate_rows = data.sample(20)
    data = pd.concat([data, duplicate_rows], ignore_index=True)
    
    # 4. Add categorical column
    neighborhoods = ['Downtown', 'Suburb', 'Rural', 'Waterfront', 'University']
    data['neighborhood'] = np.random.choice(neighborhoods, size=len(data))
    
    # 5. Add some invalid data
    invalid_indices = np.random.choice(n_samples, size=10, replace=False)
    data.loc[invalid_indices[:5], 'bedrooms'] = -1  # Invalid negative bedrooms
    data.loc[invalid_indices[5:], 'age'] = 200      # Invalid age
    
    return data

def main():
    print("COMPREHENSIVE DATA CLEANING TOOL")
    print("=" * 40)
    
    # Create sample data with issues
    print("Creating sample house price data with data quality issues...")
    df = create_sample_house_data()
    print(f"Sample data shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    # Display first few rows
    print("\nFirst 5 rows of raw data:")
    print(df.head())
    
    # Initialize data cleaner
    cleaner = DataCleaner()
    
    # Clean the data
    cleaned_df = cleaner.clean_data(
        df,
        missing_strategy='auto',
        outlier_method='iqr',
        outlier_threshold=1.5,
        outlier_action='cap',
        remove_duplicates=True,
        encode_categorical='label',
        standardize=True
    )
    
    # Generate cleaning report
    cleaner.generate_cleaning_report()
    
    # Save cleaned data
    cleaned_df.to_csv('cleaned_house_data.csv', index=False)
    print(f"\nCleaned data saved to 'cleaned_house_data.csv'")
    
    print("\nFirst 5 rows of cleaned data:")
    print(cleaned_df.head())
    
    print("\nThe data cleaning tool has:")
    print("✓ Handled missing values automatically")
    print("✓ Detected and handled outliers")
    print("✓ Removed duplicate rows")
    print("✓ Encoded categorical variables")
    print("✓ Standardized numeric features")
    print("✓ Generated a detailed cleaning report")
    print("\nThis replaces manual human intervention in the data cleaning process!")

if __name__ == "__main__":
    main()