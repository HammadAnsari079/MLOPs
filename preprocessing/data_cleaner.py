#!/usr/bin/env python
"""
Data cleaning utilities for the MLOps platform
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from scipy import stats
import warnings
import logging
warnings.filterwarnings('ignore')

# Set up logging
logger = logging.getLogger(__name__)

class DataCleaner:
    """Data cleaning utilities"""
    
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
        if df.empty:
            logger.warning("DataFrame is empty, no columns to identify")
            return [], [], []
            
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_columns = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
        
        return self.numeric_columns, self.categorical_columns, self.datetime_columns
    
    def handle_missing_values(self, df, strategy='auto'):
        """
        Handle missing values in the dataset
        strategy: 'auto', 'mean', 'median', 'mode', 'drop', 'forward', 'backward'
        """
        if df.empty:
            return df
            
        initial_missing = df.isnull().sum().sum()
        
        if initial_missing == 0:
            return df
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count == 0:
                continue
                
            if column in self.numeric_columns:
                if strategy == 'auto':
                    # Use mean for normally distributed data, median for skewed data
                    skewness = df[column].skew()
                    method = 'mean' if abs(skewness) < 1 else 'median'
                else:
                    method = strategy
                    
                if method == 'mean':
                    mean_val = df[column].mean()
                    if pd.notna(mean_val):  # Check if mean is not NaN
                        df[column].fillna(mean_val, inplace=True)
                elif method == 'median':
                    median_val = df[column].median()
                    if pd.notna(median_val):  # Check if median is not NaN
                        df[column].fillna(median_val, inplace=True)
                elif method == 'drop':
                    df.dropna(subset=[column], inplace=True)
                elif method == 'forward':
                    df[column].fillna(method='ffill', inplace=True)
                elif method == 'backward':
                    df[column].fillna(method='bfill', inplace=True)
                    
            elif column in self.categorical_columns:
                if strategy == 'auto' or strategy == 'mode':
                    mode_series = df[column].mode()
                    if not mode_series.empty:
                        df[column].fillna(mode_series[0], inplace=True)
                elif strategy == 'drop':
                    df.dropna(subset=[column], inplace=True)
                    
        final_missing = df.isnull().sum().sum()
        self.cleaning_report['missing_values_handled'] = initial_missing - final_missing
        return df
    
    def handle_outliers(self, df, method='iqr', threshold=1.5, action='cap'):
        """
        Handle outliers in numeric columns
        action: 'cap', 'remove', 'transform'
        """
        if df.empty or not self.numeric_columns:
            return df
            
        initial_rows = len(df)
        
        for column in self.numeric_columns:
            # Check if column has any data
            if df[column].empty or df[column].isna().all():
                continue
                
            # Initialize bounds
            lower_bound = None
            upper_bound = None
            
            if method == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1 if Q3 != Q1 else 1.0  # Avoid division by zero
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
            elif method == 'zscore':
                mean = df[column].mean()
                std = df[column].std()
                # Avoid division by zero
                if std == 0:
                    std = 1.0
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
                
            # Only proceed if bounds were calculated
            if lower_bound is not None and upper_bound is not None:
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
        self.cleaning_report['outliers_handled'] = initial_rows - final_rows
        return df
    
    def remove_duplicates(self, df, subset=None):
        """Remove duplicate rows"""
        if df.empty:
            return df
            
        initial_rows = len(df)
        
        if subset:
            df = df.drop_duplicates(subset=subset)
        else:
            df = df.drop_duplicates()
            
        final_rows = len(df)
        duplicates_removed = initial_rows - final_rows
        
        self.cleaning_report['duplicates_removed'] = duplicates_removed
        return df
    
    def encode_categorical_variables(self, df, method='label'):
        """
        Encode categorical variables
        method: 'label', 'onehot', 'frequency'
        """
        if df.empty or not self.categorical_columns:
            return df
            
        for column in self.categorical_columns:
            # Skip columns with all NaN values
            if df[column].isna().all():
                continue
                
            if method == 'label':
                # Label encoding
                le = LabelEncoder()
                # Fill NaN values temporarily for encoding
                temp_series = df[column].fillna('missing_value').astype(str)
                df[column] = le.fit_transform(temp_series)
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
        
        self.cleaning_report['categorical_encoded'] = len(self.categorical_columns)
        return df
    
    def standardize_numeric_features(self, df):
        """Standardize numeric features"""
        if df.empty or not self.numeric_columns:
            return df
            
        # Remove columns with zero variance
        valid_numeric_cols = []
        for col in self.numeric_columns:
            if col in df.columns and df[col].std() != 0:
                valid_numeric_cols.append(col)
        
        if not valid_numeric_cols:
            return df
            
        scaler = StandardScaler()
        df[valid_numeric_cols] = scaler.fit_transform(df[valid_numeric_cols])
        self.scalers['standard'] = scaler
        
        self.cleaning_report['features_standardized'] = len(valid_numeric_cols)
        return df