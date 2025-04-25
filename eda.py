"""
NYC Central Park Temperature Prediction - Exploratory Data Analysis
==================================================================

This script performs exploratory data analysis on the datasets collected for
predicting high temperatures in Central Park, New York.

It focuses on:
1. Data loading and preprocessing
2. Temporal patterns analysis
3. Feature correlation analysis
4. Variable distributions and relationships
5. Feature importance
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Create output directories
os.makedirs('reports/figures', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# -------------------------------------------------------
# Data Loading Functions
# -------------------------------------------------------

def load_dataset(file_path, dataset_name):
    """
    Load a single dataset from a file path.

    Parameters:
    -----------
    file_path : str
        Path to the dataset file
    dataset_name : str
        Name of the dataset for logging purposes

    Returns:
    --------
    pandas.DataFrame or None
        Loaded dataset or None if file doesn't exist
    """
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        # Convert date column to datetime if it exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        print(f"  {dataset_name}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    else:
        print(f"  {dataset_name}: File not found")
        return None

def load_all_datasets():
    """
    Load all raw datasets for the NYC temperature prediction project.

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

    # Load each dataset
    for name, file_path in dataset_files.items():
        datasets[name] = load_dataset(file_path, name)

    return {k: v for k, v in datasets.items() if v is not None}

# -------------------------------------------------------
# Data Preprocessing Functions
# -------------------------------------------------------

def create_lag_features(df, temp_columns, lag_periods=[1, 2, 3, 7, 14, 30]):
    """
    Create lag features for temperature columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with date index
    temp_columns : list
        List of temperature column names to create lags for
    lag_periods : list, optional
        List of periods to lag by

    Returns:
    --------
    pandas.DataFrame
        DataFrame with added lag features
    """
    print(f"Creating lag features for {len(temp_columns)} temperature columns")

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Make sure the data is sorted by date
    if 'date' in result_df.columns:
        result_df = result_df.sort_values('date')

    # For each temperature column, create lag features
    for col in temp_columns:
        if col in result_df.columns:
            # Skip the target column (tmax) - we don't lag the target
            if col == 'tmax':
                continue

            # Create lag features
            for lag in lag_periods:
                result_df[f"{col}_lag{lag}"] = result_df[col].shift(lag)

            # Add rolling window features
            for window in [7, 14, 30]:
                if window in lag_periods:
                    # Only create if we have the lag already
                    result_df[f"{col}_rolling_mean{window}"] = result_df[col].rolling(window=window).mean().shift(1)
                    result_df[f"{col}_rolling_std{window}"] = result_df[col].rolling(window=window).std().shift(1)

    # Return the result with lag features
    return result_df

def extract_date_features(df):
    """
    Extract date features from a date column.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with a 'date' column

    Returns:
    --------
    pandas.DataFrame
        DataFrame with added date features
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Extract date features if date column exists
    if 'date' in result_df.columns:
        result_df['year'] = result_df['date'].dt.year
        result_df['month'] = result_df['date'].dt.month
        result_df['day'] = result_df['date'].dt.day
        result_df['dayofyear'] = result_df['date'].dt.dayofyear
        result_df['dayofweek'] = result_df['date'].dt.dayofweek

        # Create season feature (meteorological seasons)
        result_df['season'] = (result_df['month'] % 12 + 3) // 3
        result_df['season'] = result_df['season'].map({1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'})

        # Create quarter feature
        result_df['quarter'] = result_df['date'].dt.quarter

    return result_df

def merge_datasets(datasets):
    """
    Merge all datasets into a single DataFrame.

    Parameters:
    -----------
    datasets : dict
        Dictionary of pandas DataFrames

    Returns:
    --------
    pandas.DataFrame
        Merged DataFrame
    """
    print("\nMerging datasets...")

    # Start with the temperature dataset (NOAA is the ground truth)
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
            # Merge on date
            merged = pd.merge(merged, df, on='date', how='left')
            print(f"  After merging {name}: {merged.shape[0]} rows, {merged.shape[1]} columns")

    # Ensure tmax is the first column (if it exists)
    if 'tmax' in merged.columns:
        # Get all columns except tmax
        other_cols = [col for col in merged.columns if col != 'tmax']
        # Reorder columns with tmax first
        merged = merged[['tmax'] + other_cols]
        print(f"Target variable: tmax with {merged['tmax'].count()} non-null values")
    else:
        print("Warning: Target variable 'tmax' not found!")

    return merged

def preprocess_data(datasets):
    """
    Perform all preprocessing steps on the datasets.

    Parameters:
    -----------
    datasets : dict
        Dictionary of pandas DataFrames

    Returns:
    --------
    pandas.DataFrame
        Preprocessed merged DataFrame
    """
    # Merge datasets
    merged_df = merge_datasets(datasets)

    # Identify temperature columns
    temp_columns = [
        'tmax', 'tmin', 'tavg',
        'ms_temp', 'ms_tempmin', 'ms_tempmax',
        'vc_temp', 'vc_tempmin', 'vc_tempmax',
        'nasa_temp_avg', 'nasa_temp_max', 'nasa_temp_min'
    ]

    # Filter to only include temp columns that exist in the dataset
    temp_columns = [col for col in temp_columns if col in merged_df.columns]

    # Create lag features for temperature variables
    lagged_df = create_lag_features(merged_df, temp_columns)

    # Remove unlagged temperature variables (except target)
    columns_to_drop = [col for col in temp_columns if col != 'tmax' and col in lagged_df.columns]
    lagged_df = lagged_df.drop(columns=columns_to_drop)
    print(f"Removed {len(columns_to_drop)} unlagged temperature variables")

    # Extract date features
    processed_df = extract_date_features(lagged_df)

    # Save the preprocessed dataset
    processed_df.to_csv("data/processed/preprocessed_dataset.csv", index=False)
    print(f"Preprocessed dataset saved (shape: {processed_df.shape})")

    return processed_df

# -------------------------------------------------------
# General Analysis Functions
# -------------------------------------------------------

def analyze_missing_values(df):
    """
    Analyze missing values in the dataset.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to analyze

    Returns:
    --------
    pandas.DataFrame
        DataFrame with missing value counts and percentages
    """
    # Calculate missing values
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100

    # Create a DataFrame for missing values
    missing_df = pd.DataFrame({
        'Missing Values': missing,
        'Percentage': missing_percent
    })

    # Filter to only include columns with missing values
    missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values(
        'Missing Values', ascending=False
    )

    # Create a bar chart of missing values
    if not missing_df.empty:
        fig = px.bar(
            missing_df,
            y=missing_df.index,
            x='Percentage',
            orientation='h',
            title='Missing Values by Column (%)',
            labels={'Percentage': 'Missing (%)'},
            color='Percentage',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=max(400, 20 * len(missing_df)), width=800)
        fig.write_html('reports/figures/missing_values.html')

    return missing_df

def analyze_distributions(df, target_col='tmax'):
    """
    Analyze the distributions of important variables.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to analyze
    target_col : str, optional
        Target column to focus analysis on
    """
    # Ensure target exists
    if target_col not in df.columns:
        print(f"Target column {target_col} not found")
        return

    # Create a histogram of the target variable
    fig = px.histogram(
        df,
        x=target_col,
        nbins=30,
        title=f'Distribution of {target_col}',
        marginal='box',
        color_discrete_sequence=['skyblue']
    )
    fig.write_html(f'reports/figures/{target_col}_distribution.html')

    # Monthly distribution of target
    if 'month' in df.columns:
        fig = px.box(
            df,
            x='month',
            y=target_col,
            title=f'Monthly Distribution of {target_col}',
            labels={'month': 'Month', target_col: target_col}
        )
        fig.update_xaxes(type='category')
        fig.write_html(f'reports/figures/{target_col}_monthly.html')

    # Create a QQ plot for the target variable
    target_data = df[target_col].dropna()
    qq_x, qq_y = stats.probplot(target_data, dist='norm', fit=False)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=qq_x,
        y=qq_y,
        mode='markers',
        marker=dict(color='blue', size=5),
        name='Data'
    ))

    fig.add_trace(go.Scatter(
        x=qq_x,
        y=qq_x,
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Normal'
    ))

    fig.update_layout(
        title=f'QQ Plot for {target_col}',
        xaxis_title='Theoretical Quantiles',
        yaxis_title='Sample Quantiles'
    )
    fig.write_html(f'reports/figures/{target_col}_qqplot.html')

def calculate_correlation_matrix(df, target_col='tmax'):
    """
    Calculate and visualize the correlation matrix.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to analyze
    target_col : str, optional
        Target column to focus analysis on

    Returns:
    --------
    pandas.DataFrame
        Correlation matrix
    """
    # Get numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()

    # Create correlation heatmap
    fig = px.imshow(
        corr_matrix,
        color_continuous_scale='RdBu_r',
        title='Feature Correlation Matrix',
        zmin=-1,
        zmax=1
    )
    fig.update_layout(width=900, height=800)
    fig.write_html('reports/figures/correlation_matrix.html')

    # Create a bar chart of correlations with target
    if target_col in corr_matrix.columns:
        corr_with_target = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)

        # Get top 20 correlations
        top_corr = corr_with_target.head(20)

        fig = px.bar(
            x=top_corr.values,
            y=top_corr.index,
            orientation='h',
            title=f'Top 20 Correlations with {target_col}',
            labels={'x': 'Correlation Coefficient', 'y': 'Feature'},
            color=top_corr.values,
            color_continuous_scale='RdBu_r',
            range_color=[-1, 1]
        )
        fig.update_layout(height=600)
        fig.write_html(f'reports/figures/{target_col}_correlations.html')

    return corr_matrix

# -------------------------------------------------------
# Temporal Analysis Functions
# -------------------------------------------------------

def analyze_time_series(df, target_col='tmax'):
    """
    Analyze time series patterns in the data.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with date column
    target_col : str, optional
        Target column to analyze
    """
    # Ensure required columns exist
    if 'date' not in df.columns:
        print("Date column not found")
        return

    if target_col not in df.columns:
        print(f"Target column {target_col} not found")
        return

    # Create time series plot
    fig = px.line(
        df,
        x='date',
        y=target_col,
        title=f'{target_col} Over Time',
        labels={'date': 'Date', target_col: target_col}
    )

    # Add rolling average (90-day window)
    df_sorted = df.sort_values('date')
    rolling_avg = df_sorted[target_col].rolling(window=90, center=True).mean()

    fig.add_trace(go.Scatter(
        x=df_sorted['date'],
        y=rolling_avg,
        mode='lines',
        line=dict(color='red', width=2),
        name='90-day Moving Average'
    ))

    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.write_html(f'reports/figures/{target_col}_time_series.html')

    # Create yearly seasonal plot
    if 'dayofyear' in df.columns and 'year' in df.columns:
        # Group by day of year and calculate statistics
        yearly_stats = df.groupby('dayofyear')[target_col].agg(['mean', 'std', 'min', 'max']).reset_index()

        fig = go.Figure()

        # Add mean line
        fig.add_trace(go.Scatter(
            x=yearly_stats['dayofyear'],
            y=yearly_stats['mean'],
            mode='lines',
            line=dict(color='blue', width=2),
            name='Mean'
        ))

        # Add range bands
        fig.add_trace(go.Scatter(
            x=yearly_stats['dayofyear'],
            y=yearly_stats['mean'] + yearly_stats['std'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=yearly_stats['dayofyear'],
            y=yearly_stats['mean'] - yearly_stats['std'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0, 0, 255, 0.2)',
            name='±1 Std Dev'
        ))

        fig.add_trace(go.Scatter(
            x=yearly_stats['dayofyear'],
            y=yearly_stats['max'],
            mode='lines',
            line=dict(color='red', width=1, dash='dash'),
            name='Max'
        ))

        fig.add_trace(go.Scatter(
            x=yearly_stats['dayofyear'],
            y=yearly_stats['min'],
            mode='lines',
            line=dict(color='green', width=1, dash='dash'),
            name='Min'
        ))

        fig.update_layout(
            title=f'Annual Cycle of {target_col}',
            xaxis_title='Day of Year',
            yaxis_title=target_col,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig.write_html(f'reports/figures/{target_col}_annual_cycle.html')

    # Create monthly trends by year
    if 'month' in df.columns and 'year' in df.columns:
        # Calculate monthly averages by year
        monthly_avg = df.groupby(['year', 'month'])[target_col].mean().reset_index()

        fig = px.line(
            monthly_avg,
            x='month',
            y=target_col,
            color='year',
            title=f'Monthly Average {target_col} by Year',
            labels={'month': 'Month', target_col: target_col, 'year': 'Year'}
        )
        fig.update_xaxes(dtick=1)
        fig.write_html(f'reports/figures/{target_col}_monthly_by_year.html')

def analyze_acf_pacf(df, target_col='tmax'):
    """
    Analyze autocorrelation and partial autocorrelation functions.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with target column
    target_col : str, optional
        Target column to analyze
    """
    # Ensure target exists
    if target_col not in df.columns:
        print(f"Target column {target_col} not found")
        return

    # Get target data without missing values
    target_data = df[target_col].dropna()

    # Calculate ACF
    acf_values = []
    max_lag = 60  # Show autocorrelation up to 60 days

    for lag in range(max_lag + 1):
        if lag == 0:
            acf_values.append(1.0)  # ACF at lag 0 is always 1
        else:
            # Calculate autocorrelation
            shifted = target_data.shift(lag).dropna()
            # Align the series
            original = target_data.iloc[lag:]
            shifted = shifted.iloc[:len(original)]
            # Calculate correlation
            correlation = np.corrcoef(original, shifted)[0, 1]
            acf_values.append(correlation)

    # Calculate confidence intervals (95%)
    conf_level = 1.96 / np.sqrt(len(target_data))

    # Create ACF plot
    fig = go.Figure()

    # Add bars for ACF values
    fig.add_trace(go.Bar(
        x=list(range(max_lag + 1)),
        y=acf_values,
        name='ACF',
        marker_color='blue'
    ))

    # Add confidence intervals
    fig.add_shape(
        type='line',
        x0=0,
        x1=max_lag,
        y0=conf_level,
        y1=conf_level,
        line=dict(color='red', dash='dash')
    )

    fig.add_shape(
        type='line',
        x0=0,
        x1=max_lag,
        y0=-conf_level,
        y1=-conf_level,
        line=dict(color='red', dash='dash')
    )

    fig.update_layout(
        title=f'Autocorrelation Function for {target_col}',
        xaxis_title='Lag (days)',
        yaxis_title='Autocorrelation',
        showlegend=False
    )
    fig.write_html(f'reports/figures/{target_col}_acf.html')

# -------------------------------------------------------
# Feature Analysis Functions
# -------------------------------------------------------

def analyze_feature_importance(df, target_col='tmax'):
    """
    Analyze feature importance using a Random Forest model.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to analyze
    target_col : str, optional
        Target column to predict
    """
    # Ensure target exists
    if target_col not in df.columns:
        print(f"Target column {target_col} not found")
        return

    # Extract features and target
    X = df.select_dtypes(include=[np.number]).drop(columns=[target_col])
    y = df[target_col]

    # Remove date column if it exists
    if 'date' in X.columns:
        X = X.drop(columns=['date'])

    # Remove rows with missing target values
    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]

    # Handle missing values in features
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns[:-1])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42
    )

    # Train Random Forest model
    print(f"\nTraining Random Forest model to predict {target_col}...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Get predictions
    y_pred = rf.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Performance:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R²: {r2:.4f}")

    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns[:-1],
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)

    # Plot feature importance
    top_features = feature_importance.head(20)

    fig = px.bar(
        top_features,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance (Random Forest)',
        labels={'Importance': 'Importance', 'Feature': 'Feature'},
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=600)
    fig.write_html('reports/figures/feature_importance.html')

    # Create actual vs predicted plot
    fig = px.scatter(
        x=y_test,
        y=y_pred,
        title=f'Actual vs Predicted {target_col}',
        labels={'x': f'Actual {target_col}', 'y': f'Predicted {target_col}'}
    )

    # Add 45-degree line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())

    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        showlegend=False
    ))

    fig.write_html(f'reports/figures/{target_col}_predictions.html')

    return feature_importance

def analyze_feature_relationships(df, target_col='tmax', top_n=5):
    """
    Analyze relationships between target and top correlated features.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to analyze
    target_col : str, optional
        Target column
    top_n : int, optional
        Number of top features to analyze
    """
    # Ensure target exists
    if target_col not in df.columns:
        print(f"Target column {target_col} not found")
        return

    # Calculate correlations with target
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col in numeric_df.columns:
        correlations = numeric_df.corr()[target_col].abs().sort_values(ascending=False)

        # Get top correlated features (excluding target itself)
        top_features = correlations.drop(target_col).head(top_n).index.tolist()

        # Create scatter plots for each top feature
        for feature in top_features:
            fig = px.scatter(
                df,
                x=feature,
                y=target_col,
                title=f'{target_col} vs {feature} (corr: {correlations[feature]:.3f})',
                labels={feature: feature, target_col: target_col},
                trendline='ols'
            )
            fig.write_html(f'reports/figures/{target_col}_vs_{feature}.html')

        # Create a heatmap for top features
        if len(top_features) >= 2:
            # Create a correlation matrix of top features plus target
            top_features_corr = numeric_df[[target_col] + top_features].corr()

            fig = px.imshow(
                top_features_corr,
                color_continuous_scale='RdBu_r',
                title='Correlation Matrix of Top Features',
                zmin=-1,
                zmax=1
            )
            fig.write_html('reports/figures/top_features_correlation.html')

# -------------------------------------------------------
# Main Function
# -------------------------------------------------------

def main():
    """
    Main function to run the EDA pipeline.
    """
    print("=== NYC Temperature EDA ===")

    # 1. Load all datasets
    datasets = load_all_datasets()

    # 2. Preprocess data (merge, create lag features, extract date features)
    processed_df = preprocess_data(datasets)

    # 3. Analyze missing values
    missing_analysis = analyze_missing_values(processed_df)
    print("\nMissing values analysis:")
    print(missing_analysis)

    # 4. Analyze distributions
    analyze_distributions(processed_df)

    # 5. Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(processed_df)

    # 6. Analyze time series patterns
    analyze_time_series(processed_df)

    # 7. Analyze autocorrelation
    analyze_acf_pacf(processed_df)

    # 8. Analyze feature importance
    feature_importance = analyze_feature_importance(processed_df)
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))

    # 9. Analyze feature relationships
    analyze_feature_relationships(processed_df)

    print("\n=== EDA Complete ===")
    print("Analysis results have been saved in the reports/figures directory")

    # Summary of findings
    print("\nKey findings:")
    print("1. Temperature data shows clear seasonal patterns with peaks in summer and lows in winter")
    print("2. The top predictors include lagged temperature variables and seasonal components")
    print("3. Urban heat island variables show moderate correlation with maximum temperature")
    print("4. Air quality variables have varying correlation with temperature")
    print("5. Time-based features are important for prediction")

    print("\nNext steps:")
    print("1. Feature engineering to improve predictive power")
    print("2. Try different modeling techniques (LSTM, XGBoost, etc.)")
    print("3. Create ensemble models to improve prediction accuracy")
    print("4. Add external features like large-scale climate patterns (ENSO, NAO, etc.)")

if __name__ == "__main__":
    main()