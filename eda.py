"""
Exploratory Data Analysis of NYC Temperature Data
=================================================

This script performs exploratory data analysis on the datasets collected for
predicting high temperatures in Central Park, New York.

The analysis covers:
1. Data loading and cleaning
2. Statistical summaries
3. Temporal patterns
4. Relationships between variables
5. Feature importance for temperature prediction
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score


# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)

# Create output directories
os.makedirs('reports/figures', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)


#########################################################
# 1. Data Loading and Cleaning
#########################################################

def load_datasets():
    """
    Load all raw datasets and perform basic cleaning

    Returns:
    --------
    dict
        Dictionary of pandas DataFrames, one for each dataset
    """
    datasets = {}

    # Define datasets and their file paths
    dataset_files = {
        "temperature": "data/raw/noaa_temperature_data.csv",
        "visual_crossing": "data/raw/visual_crossing_weather_data.csv",
        "urban_heat": "data/raw/urban_heat_island_data.csv",
        "nasa_power": "data/raw/nasa_power_data.csv",
        "meteostat": "data/raw/meteostat_weather_data.csv",
        "holidays": "data/raw/holiday_event_data.csv",
        "air_quality": "data/raw/epa_air_quality_data.csv",
        "central_park_temp": "data/raw/central_park_temperature_data.csv"
    }

    print("Loading datasets...")

    # Load each dataset if it exists
    for name, file_path in dataset_files.items():
        if os.path.exists(file_path):
            datasets[name] = pd.read_csv(file_path)
            print(f"  {name}: {datasets[name].shape[0]} rows, {datasets[name].shape[1]} columns")

            # Convert date column to datetime
            if 'date' in datasets[name].columns:
                datasets[name]['date'] = pd.to_datetime(datasets[name]['date'])
        else:
            print(f"  {name}: File not found")

    return datasets


def clean_datasets(datasets):
    """
    Clean and prepare datasets for analysis

    Parameters:
    -----------
    datasets : dict
        Dictionary of pandas DataFrames

    Returns:
    --------
    dict
        Dictionary of cleaned pandas DataFrames
    """
    print("\nCleaning datasets...")
    cleaned = {}

    for name, df in datasets.items():
        # Make a copy to avoid modifying the original
        cleaned[name] = df.copy()

        # Basic cleaning steps
        if 'date' in cleaned[name].columns:
            # Set date as index for time series analysis
            cleaned[name].set_index('date', inplace=True)

            # Sort by date
            cleaned[name].sort_index(inplace=True)

            # Remove duplicates
            cleaned[name] = cleaned[name][~cleaned[name].index.duplicated(keep='first')]

        # Check for missing values
        missing = cleaned[name].isnull().sum()
        if missing.any():
            print(f"  {name}: Found {missing.sum()} missing values across {sum(missing > 0)} columns")
            # For now we'll keep the missing values, we'll handle them during merging

        # Check for outliers in numeric columns only
        numeric_cols = cleaned[name].select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            # Z-score method for outlier detection
            z_scores = stats.zscore(cleaned[name][col].dropna())
            outliers = (np.abs(z_scores) > 3).sum()
            if outliers > 0:
                print(f"  {name}.{col}: Found {outliers} potential outliers (|z| > 3)")

    return cleaned


def merge_datasets(datasets):
    """
    Merge all datasets into a single DataFrame for analysis

    Parameters:
    -----------
    datasets : dict
        Dictionary of pandas DataFrames with date as index

    Returns:
    --------
    pandas.DataFrame
        Merged DataFrame with all features
    """
    print("\nMerging datasets...")

    # Start with the temperature dataset
    if "temperature" in datasets:
        merged = datasets["temperature"].copy()
        print(f"Starting with temperature dataset: {merged.shape[0]} rows, {merged.shape[1]} columns")
    elif "central_park_temp" in datasets:
        merged = datasets["central_park_temp"].copy()
        print(f"Starting with central_park_temp dataset: {merged.shape[0]} rows, {merged.shape[1]} columns")
    else:
        raise ValueError("No temperature dataset found!")

    # Merge other datasets one by one
    for name, df in datasets.items():
        if name not in ["temperature", "central_park_temp"]:
            # Merge on the date index
            merged = merged.join(df, how='left')
            print(f"  After merging {name}: {merged.shape[0]} rows, {merged.shape[1]} columns")

    # Identify the target variable
    # If 'tmax' exists, use it as the target, otherwise look for alternatives
    if 'tmax' in merged.columns:
        print(f"Target variable: tmax with {merged['tmax'].count()} non-null values")
        # Make sure the target variable is at the beginning for clarity
        cols = ['tmax'] + [col for col in merged.columns if col != 'tmax']
        merged = merged[cols]
    else:
        print("Warning: Target variable 'tmax' not found!")
        # Try to find an alternative
        for col in ['ms_tempmax', 'nasa_temp_max', 'vc_tempmax']:
            if col in merged.columns:
                print(f"Using {col} as alternative target")
                # Make it the first column
                cols = [col] + [c for c in merged.columns if c != col]
                merged = merged[cols]
                break

    # Reset index to keep date as a column
    merged.reset_index(inplace=True)

    # Basic dataset statistics
    print(f"\nFinal merged dataset: {merged.shape[0]} rows, {merged.shape[1]} columns")
    print(f"Date range: {merged['date'].min()} to {merged['date'].max()}")

    # Check for missing values in the merged dataset
    missing = merged.isnull().sum()
    print(f"Total missing values: {missing.sum()}")

    # Save the merged dataset
    merged.to_csv("data/processed/merged_dataset.csv", index=False)
    print("Merged dataset saved to data/processed/merged_dataset.csv")

    return merged


#########################################################
# 2. Statistical Summaries
#########################################################

def summarize_dataset(df):
    """
    Perform statistical summary of the dataset

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to summarize
    """
    print("\n=== Statistical Summary ===")

    # Basic statistics
    print("\nBasic statistics for numeric columns:")
    print(df.describe().T)

    # Count of each data type
    print("\nData types:")
    print(df.dtypes.value_counts())

    # Missing values
    missing = df.isnull().sum()
    print("\nMissing values by column:")
    print(missing[missing > 0])

    # Create a missing values heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.tight_layout()
    plt.savefig('reports/figures/missing_values_heatmap.png')
    plt.close()

    # Correlation with target (if exists)
    target_col = df.columns[0]



    # Assuming first column is the target
    correlations = df.drop(columns=["precip_type", "conditions", "holiday_name"]).corr()[target_col].sort_values(ascending=False)

    print(f"\nTop 10 correlations with {target_col}:")
    print(correlations.head(11))  # 11 because it includes the target itself

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                square=True, linewidths=.5, annot=False)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('reports/figures/correlation_matrix.png')
    plt.close()

    # Plot correlation with target
    plt.figure(figsize=(12, 8))
    correlations = correlations.drop(target_col)  # Remove self-correlation
    correlations = correlations.iloc[:20]  # Top 20 for readability
    correlations.plot(kind='barh', color='skyblue')
    plt.title(f'Top 20 Correlations with {target_col}')
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    plt.savefig('reports/figures/target_correlations.png')
    plt.close()

    return correlations


#########################################################
# 3. Temporal Analysis
#########################################################

def analyze_temporal_patterns(df):
    """
    Analyze temporal patterns in the data

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with date column
    """
    print("\n=== Temporal Analysis ===")

    # Ensure we have a date column
    if 'date' not in df.columns:
        raise ValueError("No date column found in dataframe")

    # Set date as index temporarily for time series analysis
    temp_df = df.copy()
    temp_df.set_index('date', inplace=True)

    # Target variable (assumed to be first column of original df)
    target_col = df.columns[1]  # Index 1 because date is at index 0

    # Add time components
    temp_df['year'] = temp_df.index.year
    temp_df['month'] = temp_df.index.month
    temp_df['day'] = temp_df.index.day
    temp_df['dayofyear'] = temp_df.index.dayofyear
    temp_df['dayofweek'] = temp_df.index.dayofweek
    temp_df['quarter'] = temp_df.index.quarter

    # Target by time components
    print("\nTarget by year:")
    yearly = temp_df.groupby('year')[target_col].agg(['mean', 'min', 'max', 'std'])
    print(yearly)

    print("\nTarget by month:")
    monthly = temp_df.groupby('month')[target_col].agg(['mean', 'min', 'max', 'std'])
    print(monthly)

    # Plot target variable over time
    plt.figure(figsize=(14, 7))
    plt.plot(temp_df.index, temp_df[target_col], color='blue', alpha=0.6)
    # Add trend line
    try:
        from scipy.signal import savgol_filter
        temp_df_clean = temp_df[target_col].dropna()
        if len(temp_df_clean) > 20:  # Need enough data for smoothing
            smooth = savgol_filter(temp_df_clean,
                                   window_length=101,
                                   polyorder=3)
            plt.plot(temp_df_clean.index, smooth, color='red', linewidth=2)
    except Exception as e:
        print(f"Could not calculate trend line: {e}")

    plt.title(f'{target_col} Over Time')
    plt.ylabel(target_col)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'reports/figures/{target_col}_time_series.png')
    plt.close()

    # Monthly seasonality
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='month', y=target_col, data=temp_df.reset_index())
    plt.title(f'Monthly Distribution of {target_col}')
    plt.xlabel('Month')
    plt.ylabel(target_col)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'reports/figures/{target_col}_monthly_boxplot.png')
    plt.close()

    # Day of week seasonality
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='dayofweek', y=target_col, data=temp_df.reset_index())
    plt.title(f'{target_col} by Day of Week')
    plt.xlabel('Day of Week (0=Monday)')
    plt.ylabel(target_col)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'reports/figures/{target_col}_dayofweek_boxplot.png')
    plt.close()

    # Yearly seasonality (dayofyear)
    # Group by day of year to see the annual cycle
    annual_cycle = temp_df.groupby('dayofyear')[target_col].agg(['mean', 'min', 'max'])

    plt.figure(figsize=(14, 7))
    plt.plot(annual_cycle.index, annual_cycle['mean'], color='blue', linewidth=2)
    plt.fill_between(annual_cycle.index, annual_cycle['min'],
                     annual_cycle['max'], color='blue', alpha=0.2)
    plt.title(f'Annual Cycle of {target_col}')
    plt.xlabel('Day of Year')
    plt.ylabel(target_col)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'reports/figures/{target_col}_annual_cycle.png')
    plt.close()

    # Yearly trends
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='year', y=target_col, data=temp_df.reset_index())
    plt.title(f'Yearly Distribution of {target_col}')
    plt.xlabel('Year')
    plt.ylabel(target_col)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'reports/figures/{target_col}_yearly_boxplot.png')
    plt.close()

    return yearly, monthly


#########################################################
# 4. Feature Analysis
#########################################################

def analyze_feature_relationships(df, target_col=None):
    """
    Analyze relationships between features

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to analyze
    target_col : str, optional
        Target column name (if None, use first column)
    """
    print("\n=== Feature Relationship Analysis ===")

    # If target_col not specified, use the first column
    if target_col is None:
        target_col = df.columns[1]  # Second column (after date)

    # Get numeric columns only
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    # Top correlated features
    correlations = numeric_df.corr()[target_col].sort_values(ascending=False)
    top_corr = correlations.drop(target_col).head(5)
    print(f"\nTop 5 features correlated with {target_col}:")
    print(top_corr)

    # Scatter plots for top correlations
    for col in top_corr.index:
        plt.figure(figsize=(8, 6))
        plt.scatter(df[col], df[target_col], alpha=0.5)
        plt.title(f'{target_col} vs {col} (Correlation: {correlations[col]:.3f})')
        plt.xlabel(col)
        plt.ylabel(target_col)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'reports/figures/{target_col}_vs_{col}_scatter.png')
        plt.close()

    # Pairplot of top features (if there are enough)
    if len(top_corr) >= 3:
        cols_to_plot = [target_col] + list(top_corr.index[:4])  # Top 4 + target
        try:
            pair_df = df[cols_to_plot].dropna()
            if len(pair_df) > 100:  # Only if we have enough data
                # Use a sample if dataset is very large
                if len(pair_df) > 1000:
                    pair_df = pair_df.sample(1000, random_state=42)

                pairplot = sns.pairplot(pair_df, height=2.5)
                plt.suptitle(f'Pairplot of {target_col} and Top Correlated Features', y=1.02)
                plt.tight_layout()
                plt.savefig('reports/figures/top_features_pairplot.png')
                plt.close()
        except Exception as e:
            print(f"Could not create pairplot: {e}")

    # PCA for feature dimensionality reduction
    try:
        # Remove target from PCA input
        X = numeric_df.drop(columns=[target_col] if target_col in numeric_df.columns else [])
        X = X.dropna()  # Remove rows with missing values

        if X.shape[0] > 10 and X.shape[1] >= 3:  # Need enough data and features
            # Standardize the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Apply PCA
            pca = PCA()
            pca_result = pca.fit_transform(X_scaled)

            # Explained variance
            explained_variance = pca.explained_variance_ratio_

            print("\nPCA Explained Variance Ratio:")
            for i, var in enumerate(explained_variance[:10]):
                print(f"PC{i + 1}: {var:.4f} ({var * 100:.2f}%)")

            # Cumulative explained variance
            cumulative_variance = np.cumsum(explained_variance)

            # Plot explained variance
            plt.figure(figsize=(10, 6))
            plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
            plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', color='red')
            plt.axhline(y=0.8, color='r', linestyle='-', alpha=0.3)
            plt.axhline(y=0.9, color='g', linestyle='-', alpha=0.3)
            plt.title('PCA Explained Variance')
            plt.xlabel('Principal Component')
            plt.ylabel('Explained Variance Ratio')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('reports/figures/pca_explained_variance.png')
            plt.close()

            # Feature contribution to PCs
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

            # Plot feature importance for PC1 and PC2
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.barh(X.columns, loadings[:, 0])
            plt.title('Feature Importance - PC1')
            plt.subplot(1, 2, 2)
            plt.barh(X.columns, loadings[:, 1])
            plt.title('Feature Importance - PC2')
            plt.tight_layout()
            plt.savefig('reports/figures/pca_feature_importance.png')
            plt.close()

            # PC1 vs PC2 scatter plot
            plt.figure(figsize=(10, 8))
            plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
            plt.title('PCA: PC1 vs PC2')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('reports/figures/pca_scatter.png')
            plt.close()
    except Exception as e:
        print(f"Error performing PCA: {e}")

    return correlations


#########################################################
# 5. Target Variable Analysis
#########################################################

def analyze_target_variable(df, target_col=None):
    """
    Analyze the target variable

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to analyze
    target_col : str, optional
        Target column name (if None, use first column)
    """
    print("\n=== Target Variable Analysis ===")

    # If target_col not specified, use the first column
    if target_col is None:
        target_col = df.columns[1]  # Second column (after date)

    # Basic statistics
    target = df[target_col].dropna()
    print(f"\nTarget variable: {target_col}")
    print(f"Number of observations: {len(target)}")
    print(f"Min: {target.min():.2f}")
    print(f"Max: {target.max():.2f}")
    print(f"Mean: {target.mean():.2f}")
    print(f"Median: {target.median():.2f}")
    print(f"Standard deviation: {target.std():.2f}")

    # Distribution plot
    plt.figure(figsize=(10, 6))
    sns.histplot(target, kde=True, bins=30)
    plt.title(f'Distribution of {target_col}')
    plt.xlabel(target_col)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'reports/figures/{target_col}_distribution.png')
    plt.close()

    # QQ plot to check normality
    plt.figure(figsize=(8, 8))
    stats.probplot(target, dist="norm", plot=plt)
    plt.title(f'QQ Plot for {target_col}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'reports/figures/{target_col}_qqplot.png')
    plt.close()

    # Box plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=target)
    plt.title(f'Box Plot of {target_col}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'reports/figures/{target_col}_boxplot.png')
    plt.close()

    # Check for extreme values
    threshold = 3
    z_scores = stats.zscore(target)
    outliers = (np.abs(z_scores) > threshold)

    print(f"\nOutliers (|z| > {threshold}): {outliers.sum()} values ({outliers.sum() / len(target) * 100:.2f}%)")

    if outliers.sum() > 0:
        # Plot outliers
        plt.figure(figsize=(12, 6))
        plt.scatter(range(len(target)), target, alpha=0.5)
        plt.scatter(np.where(outliers)[0], target[outliers], color='red', alpha=0.7)
        plt.title(f'Outliers in {target_col} (|z| > {threshold})')
        plt.ylabel(target_col)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'reports/figures/{target_col}_outliers.png')
        plt.close()

    return target.describe()


#########################################################
# 6. Feature Importance Using Random Forest
#########################################################

def analyze_feature_importance(df, target_col=None):
    """
    Analyze feature importance using Random Forest

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to analyze
    target_col : str, optional
        Target column name (if None, use first column)
    """
    print("\n=== Feature Importance Analysis ===")

    # If target_col not specified, use the first column
    if target_col is None:
        target_col = df.columns[1]  # Second column (after date)

    # Extract features and target
    X = df.select_dtypes(include=['float64', 'int64']).drop(columns=[target_col] if target_col in df.columns else [])
    y = df[target_col]

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Filter rows where target is available
    mask = ~y.isna()
    X_imputed = X_imputed[mask]
    y = y[mask]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42
    )

    # Train a Random Forest model
    print("\nTraining Random Forest model for feature importance...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Random Forest Performance:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R²: {r2:.4f}")

    # Get feature importance
    importances = rf.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    print("\nTop 10 most important features:")
    print(feature_importance.head(10))

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
    plt.title('Feature Importance from Random Forest')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/figures/feature_importance.png')
    plt.close()

    # Scatter plot of actual vs predicted values
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted {target_col}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/figures/actual_vs_predicted.png')
    plt.close()

    return feature_importance


#########################################################
# 7. Interactive Visualizations with Plotly
#########################################################

def create_interactive_plots(df, target_col=None):
    """
    Create interactive plots using Plotly

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to visualize
    target_col : str, optional
        Target column name (if None, use first column)
    """
    print("\n=== Creating Interactive Plots ===")

    # If target_col not specified, use the first column
    if target_col is None:
        target_col = df.columns[1]  # Second column (after date)

    # Time series plot
    fig = px.line(df, x='date', y=target_col, title=f'{target_col} Over Time')
    fig.update_xaxes(title='Date')
    fig.update_yaxes(title=target_col)
    fig.write_html('reports/figures/interactive_time_series.html')

    # Seasonal patterns
    # Add time components
    temp_df = df.copy()
    temp_df['year'] = temp_df['date'].dt.year
    temp_df['month'] = temp_df['date'].dt.month
    temp_df['day'] = temp_df['date'].dt.day

    # Monthly patterns by year
    fig = px.box(temp_df, x='month', y=target_col, color='year',
                 title=f'Monthly Distribution of {target_col} by Year')
    fig.update_xaxes(title='Month')
    fig.update_yaxes(title=target_col)
    fig.write_html('reports/figures/interactive_monthly_boxplot.html')

    # Correlation heatmap
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    fig = px.imshow(corr_matrix,
                    title='Feature Correlation Matrix',
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1)
    fig.write_html('reports/figures/interactive_correlation_heatmap.html')

    # Feature relationships
    # Get top correlated features
    correlations = corr_matrix[target_col].sort_values(ascending=False)
    top_features = correlations.drop(target_col).head(5).index.tolist()

    # Create scatterplot matrix
    if len(top_features) >= 2:
        for feature in top_features[:3]:  # Limit to top 3 for readability
            fig = px.scatter(df, x=feature, y=target_col, color='month',
                             title=f'{target_col} vs {feature}',
                             color_continuous_scale=px.colors.cyclical.IceFire)
            fig.update_xaxes(title=feature)
            fig.update_yaxes(title=target_col)
            fig.write_html(f'reports/figures/interactive_{target_col}_vs_{feature}.html')

    # 3D plot of top 3 features
    if len(top_features) >= 3:
        fig = px.scatter_3d(df, x=top_features[0], y=top_features[1], z=top_features[2],
                            color=target_col, title=f'3D Plot of Top Features',
                            color_continuous_scale='Viridis')
        fig.update_layout(scene=dict(
            xaxis_title=top_features[0],
            yaxis_title=top_features[1],
            zaxis_title=top_features[2]))
        fig.write_html('reports/figures/interactive_3d_features.html')

    # Create prediction vs actual plot if we have enough data
    try:
        # Extract features and target
        X = df.select_dtypes(include=['float64', 'int64']).drop(
            columns=[target_col] if target_col in df.columns else [])
        y = df[target_col]

        # Handle missing values
        X = X.fillna(X.median())
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]

        if len(X) > 100:  # Only if we have enough data
            # Train a quick random forest model
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            rf.fit(X_train, y_train)

            # Get predictions
            y_pred = rf.predict(X_test)

            # Create scatter plot of actual vs predicted
            pred_df = pd.DataFrame({
                'Actual': y_test,
                'Predicted': y_pred,
                'Date': df.loc[X_test.index, 'date'].values
            })

            fig = px.scatter(pred_df, x='Actual', y='Predicted',
                             hover_data=['Date'],
                             title=f'Actual vs Predicted {target_col}')

            # Add 45-degree line
            fig.add_trace(go.Scatter(
                x=[pred_df['Actual'].min(), pred_df['Actual'].max()],
                y=[pred_df['Actual'].min(), pred_df['Actual'].max()],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))

            fig.update_xaxes(title=f'Actual {target_col}')
            fig.update_yaxes(title=f'Predicted {target_col}')
            fig.write_html('reports/figures/interactive_prediction.html')

    except Exception as e:
        print(f"Error creating prediction plot: {e}")

    print("Interactive plots saved to reports/figures/")


#########################################################
# 8. Unsupervised Learning (Clustering)
#########################################################

def perform_clustering(df, target_col=None):
    """
    Perform clustering analysis

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to analyze
    target_col : str, optional
        Target column name (if None, use first column)
    """
    print("\n=== Clustering Analysis ===")

    # If target_col not specified, use the first column
    if target_col is None:
        target_col = df.columns[1]  # Second column (after date)

    # Get numeric columns only, excluding date
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    # Remove target column from features
    if target_col in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target_col])

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(numeric_df)
    X = pd.DataFrame(X_imputed, columns=numeric_df.columns[:X_imputed.shape[1]])

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine optimal number of clusters
    if X_scaled.shape[0] > 20:  # Need enough data
        # Try different k values
        k_range = range(2, min(11, X_scaled.shape[0] // 5 + 1))  # Up to 10 clusters or fewer if small dataset
        inertias = []
        silhouette_scores = []

        for k in k_range:
            # K-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            inertias.append(kmeans.inertia_)

            # Only calculate silhouette score if we have at least 3 points per cluster on average
            if X_scaled.shape[0] / k >= 3:
                silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
            else:
                silhouette_scores.append(0)

        # Plot elbow curve
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(k_range, inertias, 'o-')
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.grid(True, alpha=0.3)

        # Plot silhouette scores
        plt.subplot(1, 2, 2)
        plt.plot(k_range, silhouette_scores, 'o-')
        plt.title('Silhouette Method')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('reports/figures/clustering_optimal_k.png')
        plt.close()

        # Select best k
        best_k = k_range[np.argmax(silhouette_scores)] if any(silhouette_scores) else 3
        print(f"Optimal number of clusters based on silhouette score: {best_k}")

        # Perform clustering with optimal k
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

        # Add cluster labels to original data
        temp_df = df.copy()
        temp_df['cluster'] = cluster_labels

        # Analyze clusters
        print("\nCluster analysis:")
        for cluster in range(best_k):
            cluster_data = temp_df[temp_df['cluster'] == cluster]
            print(f"\nCluster {cluster} ({len(cluster_data)} samples):")
            if target_col in df.columns:
                cluster_target = cluster_data[target_col].dropna()
                if len(cluster_target) > 0:
                    print(f"  {target_col}: mean={cluster_target.mean():.2f}, std={cluster_target.std():.2f}")

            # Get top 3 most distinct features for this cluster
            cluster_means = X.loc[temp_df['cluster'] == cluster].mean()
            global_means = X.mean()
            diff = (cluster_means - global_means).abs()
            top_features = diff.nlargest(3).index.tolist()

            for feature in top_features:
                feature_vals = cluster_data[feature].dropna()
                if len(feature_vals) > 0:
                    print(f"  {feature}: mean={feature_vals.mean():.2f}, std={feature_vals.std():.2f}")

        # Visualize clusters using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Cluster')
        plt.title('Clusters Visualization (PCA)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('reports/figures/clusters_pca.png')
        plt.close()

        # Create interactive plot
        pca_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Cluster': cluster_labels,
            'Date': df['date'].values
        })

        if target_col in df.columns:
            pca_df[target_col] = df[target_col].values

        fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster',
                         hover_data=['Date'] + ([target_col] if target_col in df.columns else []),
                         title='Clusters Visualization (PCA)',
                         color_continuous_scale='Viridis')

        fig.update_xaxes(title='Principal Component 1')
        fig.update_yaxes(title='Principal Component 2')
        fig.write_html('reports/figures/interactive_clusters.html')

        # Analyze temperature by cluster and month
        if 'month' in temp_df.columns and target_col in df.columns:
            pivot = temp_df.pivot_table(index='month', columns='cluster', values=target_col, aggfunc='mean')

            plt.figure(figsize=(12, 8))
            pivot.plot(kind='bar')
            plt.title(f'{target_col} by Month and Cluster')
            plt.xlabel('Month')
            plt.ylabel(f'Average {target_col}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'reports/figures/{target_col}_by_month_cluster.png')
            plt.close()

        return temp_df
    else:
        print("Not enough data for clustering analysis")
        return None


#########################################################
# 9. Lag Features and Time Series Analysis
#########################################################

def analyze_lag_features(df, target_col=None):
    """
    Analyze lag features for time series prediction

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with date index
    target_col : str, optional
        Target column name (if None, use first column)
    """
    print("\n=== Lag Features Analysis ===")

    # If target_col not specified, use the first column
    if target_col is None:
        target_col = df.columns[1]  # Second column (after date)

    # Create a copy and set date as index
    ts_df = df.copy()
    ts_df.set_index('date', inplace=True)

    # Sort index
    ts_df = ts_df.sort_index()

    # Check if we have the target column
    if target_col not in ts_df.columns:
        print(f"Target column {target_col} not found in data")
        return None

    # Create lag features
    lag_features = pd.DataFrame(index=ts_df.index)
    lag_features[target_col] = ts_df[target_col]

    # Add lags from 1 to 7 days
    for lag in range(1, 8):
        lag_features[f'{target_col}_lag_{lag}'] = ts_df[target_col].shift(lag)

    # Add rolling window features
    windows = [3, 7, 14, 30]
    for window in windows:
        lag_features[f'{target_col}_rolling_mean_{window}'] = ts_df[target_col].rolling(window=window).mean()
        lag_features[f'{target_col}_rolling_std_{window}'] = ts_df[target_col].rolling(window=window).std()
        lag_features[f'{target_col}_rolling_max_{window}'] = ts_df[target_col].rolling(window=window).max()
        lag_features[f'{target_col}_rolling_min_{window}'] = ts_df[target_col].rolling(window=window).min()

    # Add seasonal features
    lag_features['month'] = lag_features.index.month
    lag_features['day'] = lag_features.index.day
    lag_features['dayofyear'] = lag_features.index.dayofyear
    lag_features['dayofweek'] = lag_features.index.dayofweek

    # Calculate correlation of lag features with target
    correlations = lag_features.corr()[target_col].sort_values(ascending=False).drop(target_col)

    print("\nTop lag features correlation with target:")
    print(correlations.head(10))

    # Plot top lag correlations
    plt.figure(figsize=(12, 8))
    correlations.head(15).plot(kind='bar')
    plt.title(f'Top Lag Features Correlation with {target_col}')
    plt.ylabel('Correlation')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/figures/lag_correlations.png')
    plt.close()

    # Plot autocorrelation
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plot_acf(ts_df[target_col].dropna(), ax=plt.gca(), lags=30)
    plt.title(f'Autocorrelation of {target_col}')

    plt.subplot(1, 2, 2)
    plot_pacf(ts_df[target_col].dropna(), ax=plt.gca(), lags=30)
    plt.title(f'Partial Autocorrelation of {target_col}')

    plt.tight_layout()
    plt.savefig('reports/figures/autocorrelation.png')
    plt.close()

    # Plot lag scatter plots
    plt.figure(figsize=(15, 10))
    for i, lag in enumerate(range(1, 7)):
        plt.subplot(2, 3, i + 1)
        lag_col = f'{target_col}_lag_{lag}'
        plt.scatter(lag_features[lag_col], lag_features[target_col], alpha=0.5)
        plt.xlabel(f'{target_col} (t-{lag})')
        plt.ylabel(f'{target_col} (t)')
        plt.title(f'Lag {lag} vs Target (corr: {correlations[lag_col]:.3f})')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/figures/lag_scatterplots.png')
    plt.close()

    # Save processed data with lag features
    lag_features.reset_index().to_csv('data/processed/data_with_lag_features.csv', index=False)
    print("Lag features saved to data/processed/data_with_lag_features.csv")

    return lag_features


#########################################################
# Main Function
#########################################################

def main():
    """
    Main function to run the full EDA pipeline
    """
    print("=== NYC Temperature EDA ===")

    # 1. Load datasets
    datasets = load_datasets()

    # 2. Clean datasets
    cleaned_datasets = clean_datasets(datasets)

    # 3. Merge datasets
    merged_data = merge_datasets(cleaned_datasets)

    # Define target column (assuming it's the first non-date column)
    target_col = merged_data.columns[1]  # Second column (after date)
    print(f"\nUsing {target_col} as target variable")

    # Extract month, day, etc. for analysis
    merged_data['year'] = merged_data['date'].dt.year
    merged_data['month'] = merged_data['date'].dt.month
    merged_data['day'] = merged_data['date'].dt.day
    merged_data['dayofyear'] = merged_data['date'].dt.dayofyear
    merged_data['dayofweek'] = merged_data['date'].dt.dayofweek

    # 4. Statistical summaries
    summarize_dataset(merged_data)

    # 5. Temporal analysis
    analyze_temporal_patterns(merged_data)

    # 6. Feature relationships
    analyze_feature_relationships(merged_data, target_col)

    # 7. Target variable analysis
    analyze_target_variable(merged_data, target_col)

    # 8. Feature importance
    analyze_feature_importance(merged_data, target_col)

    # 9. Interactive plots
    create_interactive_plots(merged_data, target_col)

    # 10. Unsupervised learning (clustering)
    cluster_data = perform_clustering(merged_data, target_col)

    # 11. Lag features analysis
    lag_features = analyze_lag_features(merged_data, target_col)

    print("\n=== EDA Complete ===")
    print("The analysis results have been saved in the reports/figures directory")
    print("Processed datasets have been saved in the data/processed directory")
    print("\nNext steps:")
    print("1. Feature engineering based on insights from the EDA")
    print("2. Build and evaluate different predictive models")
    print("3. Deploy the best model for temperature prediction")


if __name__ == "__main__":
    main()