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

    # 10. Analyze feature categories
    analyze_feature_categories(processed_df)

    # 11. Create detailed visualizations for each feature category
    print("\nCreating detailed visualizations for feature categories...")

    # Precipitation analysis
    plot_precipitation_analysis(processed_df)

    # Wind analysis
    plot_wind_analysis(processed_df)

    # Atmospheric analysis
    plot_atmospheric_analysis(processed_df)

    # Urban heat island analysis
    plot_urban_heat_analysis(processed_df)

    # Air quality analysis
    plot_air_quality_analysis(processed_df)

    # Calendar/time features analysis
    plot_calendar_analysis(processed_df)

    print("\n=== EDA Complete ===")
    print("Analysis results have been saved in the reports")
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
        x=[qq_x[0]],
        y=[qq_x[0]],
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

    # Group features by category for detailed analysis
    feature_categories = {
        'Temperature Lags': [col for col in X.columns if any(x in col for x in ['tmin_lag', 'tavg_lag', 'ms_temp', 'vc_temp', 'nasa_temp'])],
        'Precipitation': [col for col in X.columns if any(x in col for x in ['prcp', 'snow', 'precipitation', 'precip'])],
        'Wind': [col for col in X.columns if 'wind' in col],
        'Atmospheric': [col for col in X.columns if any(x in col for x in ['pressure', 'humidity', 'cloud', 'visibility', 'uv'])],
        'Urban Heat': [col for col in X.columns if any(x in col for x in ['albedo', 'vegetation', 'canopy', 'impervious', 'building', 'water', 'heatmap'])],
        'Air Quality': [col for col in X.columns if any(x in col for x in ['ozone', 'carbon', 'dioxide', 'pm25'])],
        'Calendar': [col for col in X.columns if any(x in col for x in ['year', 'month', 'day', 'week', 'season', 'holiday', 'weekend'])]
    }

    # Create importance plot for each category
    for category, cols in feature_categories.items():
        # Filter to features that exist in the dataset
        category_cols = [col for col in cols if col in feature_importance['Feature'].values]

        if category_cols:
            # Get importance for these features
            category_importance = feature_importance[feature_importance['Feature'].isin(category_cols)]

            if not category_importance.empty:
                # Sort by importance
                category_importance = category_importance.sort_values('Importance', ascending=False)

                # Take top 15 for readability
                category_importance = category_importance.head(15)

                # Create plot
                fig = px.bar(
                    category_importance,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f'Feature Importance: {category} Features',
                    labels={'Importance': 'Importance', 'Feature': 'Feature'},
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=500)
                fig.write_html(f'reports/figures/feature_importance_{category.lower().replace(" ", "_")}.html')

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


def plot_urban_heat_analysis(df, target_col='tmax'):
    """
    Create visualizations for urban heat island features.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to analyze
    target_col : str, optional
        Target column
    """
    # Identify urban heat columns
    uhi_cols = [col for col in df.columns if any(x in col for x in [
        'albedo', 'vegetation', 'tree_canopy', 'impervious', 'building',
        'water_surface', 'distance_to_water', 'heatmap'
    ])]
    uhi_cols = [col for col in uhi_cols if col in df.columns]

    if not uhi_cols:
        print("No urban heat island columns found")
        return

    print(f"Analyzing {len(uhi_cols)} urban heat island features")

    # 1. Correlation with target temperature
    uhi_target_corr = []
    for col in uhi_cols:
        corr = df[[target_col, col]].corr().iloc[0, 1]
        if not np.isnan(corr):
            uhi_target_corr.append({'Feature': col, 'Correlation': corr})

    if uhi_target_corr:
        uhi_corr_df = pd.DataFrame(uhi_target_corr)
        uhi_corr_df = uhi_corr_df.sort_values('Correlation', ascending=False)

        fig = px.bar(
            uhi_corr_df,
            y='Feature',
            x='Correlation',
            orientation='h',
            title=f'Correlation of Urban Heat Island Features with {target_col}',
            color='Correlation',
            color_continuous_scale='RdBu_r',
            range_color=[-1, 1]
        )
        fig.update_layout(height=500)
        fig.write_html('reports/figures/uhi_correlations.html')

        # Get top 4 features based on absolute correlation
        top_features = uhi_corr_df.iloc[:4]['Feature'].tolist()
    else:
        # If correlation couldn't be calculated, just take first 4
        top_features = uhi_cols[:min(4, len(uhi_cols))]

    # 2. Time series analysis of urban heat features
    if top_features and 'date' in df.columns:
        # Filter data
        ts_data = df[['date'] + top_features].copy().dropna()
        ts_data = ts_data.sort_values('date')

        # Create subplot with multiple UHI measures
        fig = make_subplots(rows=len(top_features), cols=1, shared_xaxes=True)

        for i, feature in enumerate(top_features):
            fig.add_trace(
                go.Scatter(x=ts_data['date'], y=ts_data[feature], name=feature),
                row=i + 1, col=1
            )
            fig.update_yaxes(title_text=feature, row=i + 1, col=1)

        fig.update_layout(
            height=800,
            title_text="Urban Heat Island Features Over Time",
            showlegend=True
        )
        fig.write_html('reports/figures/uhi_time_series.html')

    # 3. Seasonal patterns of UHI effect
    if 'season' in df.columns:
        seasonal_data = []
        for feature in top_features:
            seasonal_avg = df.groupby('season')[feature].mean().reset_index()
            seasonal_avg['Feature'] = feature
            seasonal_avg.rename(columns={feature: 'Value'}, inplace=True)
            seasonal_data.append(seasonal_avg)

        if seasonal_data:
            seasonal_df = pd.concat(seasonal_data)

            fig = px.bar(
                seasonal_df,
                x='season',
                y='Value',
                color='Feature',
                barmode='group',
                title='Seasonal Patterns of Urban Heat Island Effects',
                labels={'season': 'Season', 'Value': 'Value'},
                category_orders={'season': ['Winter', 'Spring', 'Summer', 'Fall']}
            )
            fig.update_layout(height=500)
            fig.write_html('reports/figures/uhi_seasonal.html')

    # 4. Monthly patterns
    if 'month' in df.columns:
        monthly_data = []
        for feature in top_features:
            monthly_avg = df.groupby('month')[feature].mean().reset_index()
            monthly_avg['Feature'] = feature
            monthly_avg.rename(columns={feature: 'Value'}, inplace=True)
            monthly_data.append(monthly_avg)

        if monthly_data:
            monthly_df = pd.concat(monthly_data)

            fig = px.line(
                monthly_df,
                x='month',
                y='Value',
                color='Feature',
                title='Monthly Patterns of Urban Heat Island Effects',
                labels={'month': 'Month', 'Value': 'Value'}
            )
            fig.update_xaxes(dtick=1)
            fig.update_layout(height=500)
            fig.write_html('reports/figures/uhi_monthly.html')

    # 5. Correlation between UHI variables
    if len(uhi_cols) >= 2:
        uhi_df = df[uhi_cols].dropna()
        uhi_corr = uhi_df.corr()

        fig = px.imshow(
            uhi_corr,
            color_continuous_scale='RdBu_r',
            title='Correlation Between Urban Heat Island Variables',
            zmin=-1,
            zmax=1
        )
        fig.write_html('reports/figures/uhi_correlation.html')

    # 6. Scatter plots of key UHI variables vs temperature
    for feature in top_features:
        fig = px.scatter(
            df,
            x=feature,
            y=target_col,
            title=f'{target_col} vs {feature}',
            labels={feature: feature, target_col: target_col},
            trendline='ols'
        )
        fig.write_html(f'reports/figures/{target_col}_vs_{feature}.html')

    # 7. Scatter plot matrix for top UHI variables
    if len(top_features) >= 2:
        # Create scatter matrix for top features and target
        scatter_data = df[[target_col] + top_features].copy().dropna()

        # If dataset is large, sample it
        if len(scatter_data) > 1000:
            scatter_data = scatter_data.sample(1000, random_state=42)

        fig = px.scatter_matrix(
            scatter_data,
            dimensions=[target_col] + top_features,
            title='Scatter Matrix of Urban Heat Island Features',
            opacity=0.7
        )
        fig.update_layout(height=800, width=800)
        fig.write_html('reports/figures/uhi_scatter_matrix.html')

    # 8. Analyze urban-rural temperature differences
    # Urban heat islands show stronger effects with higher building density and less vegetation
    if 'building_density' in uhi_cols or 'vegetation_index' in uhi_cols:
        # Define an urban vs rural index for visualization
        urban_cols = [col for col in uhi_cols if any(x in col for x in ['building', 'impervious'])]
        rural_cols = [col for col in uhi_cols if any(x in col for x in ['vegetation', 'tree_canopy', 'water_surface'])]

        if urban_cols and rural_cols:
            # Create a simple urban-rural index
            # Just use the first variables from each list for simplicity
            urban_col = urban_cols[0]
            rural_col = rural_cols[0]

            # Create temperature comparison by urban vs rural characteristics
            urban_rural_df = df[[target_col, urban_col, rural_col]].copy().dropna()

            # Bin the urban and rural variables
            urban_rural_df['urban_bin'] = pd.qcut(urban_rural_df[urban_col], 3, labels=['Low', 'Medium', 'High'])
            urban_rural_df['rural_bin'] = pd.qcut(urban_rural_df[rural_col], 3, labels=['Low', 'Medium', 'High'])

            # Plot temperature by urban characteristic
            fig = px.box(
                urban_rural_df,
                x='urban_bin',
                y=target_col,
                title=f'{target_col} by {urban_col} Level',
                labels={'urban_bin': f'{urban_col} Level', target_col: target_col}
            )
            fig.write_html(f'reports/figures/{target_col}_by_urban.html')

            # Plot temperature by rural characteristic
            fig = px.box(
                urban_rural_df,
                x='rural_bin',
                y=target_col,
                title=f'{target_col} by {rural_col} Level',
                labels={'rural_bin': f'{rural_col} Level', target_col: target_col}
            )
            fig.write_html(f'reports/figures/{target_col}_by_rural.html')

            # Create heatmap of temperature by urban and rural characteristics
            temp_matrix = urban_rural_df.groupby(['urban_bin', 'rural_bin'])[target_col].mean().unstack()

            fig = px.imshow(
                temp_matrix,
                title=f'Average {target_col} by Urban-Rural Characteristics',
                color_continuous_scale='Viridis',
                labels={'x': rural_col, 'y': urban_col, 'color': target_col}
            )
            fig.write_html('reports/figures/urban_rural_temp_heatmap.html')

def plot_atmospheric_analysis(df, target_col='tmax'):
    """
    Create visualizations for atmospheric features (pressure, humidity, etc.).

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to analyze
    target_col : str, optional
        Target column
    """
    # Identify atmospheric columns
    atm_cols = [col for col in df.columns if
                any(x in col for x in ['pressure', 'humidity', 'cloud', 'visibility', 'uv'])]
    atm_cols = [col for col in atm_cols if col in df.columns]

    if not atm_cols:
        print("No atmospheric columns found")
        return

    print(f"Analyzing {len(atm_cols)} atmospheric features")

    # Select key atmospheric features
    key_features = []
    for term in ['pressure', 'humidity', 'cloud_cover', 'visibility', 'uv_index']:
        matches = [col for col in atm_cols if term in col]
        if matches:
            key_features.append(matches[0])

    # Keep top 4 features for clarity
    key_features = key_features[:4]

    # 1. Time series analysis of atmospheric features
    if key_features and 'date' in df.columns:
        # Filter data
        ts_data = df[['date'] + key_features].copy().dropna()
        ts_data = ts_data.sort_values('date')

        # Create subplot with multiple atmospheric measures
        fig = make_subplots(rows=len(key_features), cols=1, shared_xaxes=True)

        for i, feature in enumerate(key_features):
            fig.add_trace(
                go.Scatter(x=ts_data['date'], y=ts_data[feature], name=feature),
                row=i + 1, col=1
            )
            fig.update_yaxes(title_text=feature, row=i + 1, col=1)

        fig.update_layout(
            height=800,
            title_text="Atmospheric Features Over Time",
            showlegend=True
        )
        fig.write_html('reports/figures/atmospheric_time_series.html')

    # 2. Monthly patterns of atmospheric variables
    if 'month' in df.columns:
        monthly_data = []

        for feature in key_features:
            monthly_avg = df.groupby('month')[feature].mean().reset_index()
            monthly_avg['Feature'] = feature
            monthly_avg.rename(columns={feature: 'Value'}, inplace=True)
            monthly_data.append(monthly_avg)

        if monthly_data:
            monthly_df = pd.concat(monthly_data)

            fig = px.line(
                monthly_df,
                x='month',
                y='Value',
                color='Feature',
                title='Monthly Average Atmospheric Conditions',
                labels={'month': 'Month', 'Value': 'Value'},
                facet_col='Feature',
                facet_col_wrap=2
            )
            fig.update_xaxes(dtick=1)
            fig.update_layout(height=800)
            fig.write_html('reports/figures/atmospheric_monthly.html')

    # 3. Correlation between atmospheric variables and temperature
    corr_data = []
    for feature in atm_cols:
        corr = df[[target_col, feature]].corr().iloc[0, 1]
        if not np.isnan(corr):
            corr_data.append({'Feature': feature, 'Correlation': corr})

    if corr_data:
        corr_df = pd.DataFrame(corr_data)
        corr_df = corr_df.sort_values('Correlation', ascending=False)

        fig = px.bar(
            corr_df,
            y='Feature',
            x='Correlation',
            orientation='h',
            title=f'Correlation between Atmospheric Features and {target_col}',
            color='Correlation',
            color_continuous_scale='RdBu_r',
            range_color=[-1, 1]
        )
        fig.update_layout(height=600)
        fig.write_html('reports/figures/atmospheric_correlations.html')

    # 4. Create scatter plots with trendlines for each feature vs temperature
    for feature in key_features:
        fig = px.scatter(
            df,
            x=feature,
            y=target_col,
            title=f'{target_col} vs {feature}',
            labels={feature: feature, target_col: target_col},
            trendline='ols'
        )
        fig.write_html(f'reports/figures/{target_col}_vs_{feature}.html')

    # 5. Correlation heatmap between atmospheric variables
    if len(atm_cols) >= 2:
        atm_df = df[atm_cols].dropna()
        atm_corr = atm_df.corr()

        fig = px.imshow(
            atm_corr,
            color_continuous_scale='RdBu_r',
            title='Correlation Between Atmospheric Variables',
            zmin=-1,
            zmax=1
        )
        fig.write_html('reports/figures/atmospheric_correlation.html')

    # 6. Distribution of atmospheric variables by season
    if 'season' in df.columns:
        for feature in key_features:
            fig = px.box(
                df,
                x='season',
                y=feature,
                title=f'{feature} by Season',
                labels={'season': 'Season', feature: feature},
                category_orders={'season': ['Winter', 'Spring', 'Summer', 'Fall']}
            )
            fig.write_html(f'reports/figures/{feature}_by_season.html')

    # 7. Pressure vs Humidity relationship (if both exist)
    pressure_cols = [col for col in atm_cols if 'pressure' in col]
    humidity_cols = [col for col in atm_cols if 'humidity' in col]

    if pressure_cols and humidity_cols:
        # Use the first pressure and humidity columns
        pressure_col = pressure_cols[0]
        humidity_col = humidity_cols[0]

        fig = px.scatter(
            df,
            x=pressure_col,
            y=humidity_col,
            color=target_col,
            title=f'Pressure vs Humidity (colored by {target_col})',
            labels={pressure_col: 'Pressure', humidity_col: 'Humidity'},
            color_continuous_scale='Viridis'
        )
        fig.write_html('reports/figures/pressure_vs_humidity.html')

    # 8. Atmospheric conditions and extreme temperatures
    # Analyze if extreme temperatures correlate with specific atmospheric conditions
    if target_col in df.columns:
        # Define extreme as top and bottom 10% of temperatures
        high_temp_threshold = df[target_col].quantile(0.9)
        low_temp_threshold = df[target_col].quantile(0.1)

        # Create a new column for temperature category
        temp_df = df.copy()
        temp_df['temp_category'] = pd.cut(
            temp_df[target_col],
            bins=[temp_df[target_col].min() - 0.1, low_temp_threshold, high_temp_threshold,
                  temp_df[target_col].max() + 0.1],
            labels=['Cold', 'Normal', 'Hot']
        )

        # For each atmospheric variable, create boxplot by temperature category
        for feature in key_features:
            fig = px.box(
                temp_df,
                x='temp_category',
                y=feature,
                title=f'{feature} by Temperature Category',
                labels={'temp_category': 'Temperature Category', feature: feature},
                color='temp_category',
                category_orders={'temp_category': ['Cold', 'Normal', 'Hot']},
                color_discrete_map={'Cold': 'blue', 'Normal': 'gray', 'Hot': 'red'}
            )
            fig.write_html(f'reports/figures/{feature}_by_temp_category.html')

def plot_wind_analysis(df, target_col='tmax'):
    """
    Create visualizations for wind-related features.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to analyze
    target_col : str, optional
        Target column
    """
    # Identify wind columns
    wind_cols = [col for col in df.columns if 'wind' in col]
    wind_cols = [col for col in wind_cols if col in df.columns]

    if not wind_cols:
        print("No wind columns found")
        return

    print(f"Analyzing {len(wind_cols)} wind features")

    # Select key wind features
    key_features = []
    for term in ['wind_speed', 'ms_wind_speed', 'nasa_wind_speed', 'wind_direction']:
        matches = [col for col in wind_cols if term in col]
        if matches:
            key_features.append(matches[0])

    # Keep only top 4 features for clarity
    key_features = key_features[:4]

    # 1. Time series analysis of wind features
    if key_features and 'date' in df.columns:
        # Create a copy and filter out missing values
        ts_data = df[['date'] + key_features].copy().dropna()
        ts_data = ts_data.sort_values('date')

        # Create subplot with multiple wind measures
        fig = make_subplots(rows=len(key_features), cols=1, shared_xaxes=True)

        for i, feature in enumerate(key_features):
            fig.add_trace(
                go.Scatter(x=ts_data['date'], y=ts_data[feature], name=feature),
                row=i + 1, col=1
            )
            fig.update_yaxes(title_text=feature, row=i + 1, col=1)

        fig.update_layout(
            height=800,
            title_text="Wind Features Over Time",
            showlegend=True
        )
        fig.write_html('reports/figures/wind_time_series.html')

    # 2. Monthly patterns of wind
    if 'month' in df.columns:
        monthly_data = []
        for feature in key_features:
            feature_data = df.groupby('month')[feature].mean().reset_index()
            feature_data['Feature'] = feature
            feature_data.rename(columns={feature: 'Value'}, inplace=True)
            monthly_data.append(feature_data)

        if monthly_data:
            monthly_df = pd.concat(monthly_data)

            fig = px.line(
                monthly_df,
                x='month',
                y='Value',
                color='Feature',
                title='Monthly Average Wind Measures',
                labels={'month': 'Month', 'Value': 'Value'}
            )
            fig.update_xaxes(dtick=1)
            fig.write_html('reports/figures/wind_monthly.html')

    # 3. Wind rose chart (if we have direction and speed)
    direction_cols = [col for col in wind_cols if 'direction' in col]
    speed_cols = [col for col in wind_cols if 'speed' in col and 'gust' not in col]

    if direction_cols and speed_cols:
        # Use the first direction and speed columns
        dir_col = direction_cols[0]
        spd_col = speed_cols[0]

        # Create wind rose for all data
        wind_data = df.dropna(subset=[dir_col, spd_col])

        # Create bins for direction
        wind_data['direction_bin'] = pd.cut(
            wind_data[dir_col],
            bins=np.arange(0, 361, 45),
            labels=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        )

        # Create bins for speed
        wind_data['speed_bin'] = pd.cut(
            wind_data[spd_col],
            bins=[0, 5, 10, 15, 20, 100],
            labels=['0-5', '5-10', '10-15', '15-20', '20+']
        )

        # Count occurrences
        wind_counts = wind_data.groupby(['direction_bin', 'speed_bin']).size().reset_index(name='count')

        # Create wind rose
        fig = px.bar_polar(
            wind_counts,
            r='count',
            theta='direction_bin',
            color='speed_bin',
            title='Wind Rose - All Data',
            color_discrete_sequence=px.colors.sequential.Plasma_r
        )
        fig.write_html('reports/figures/wind_rose_all.html')

        # Create a seasonal wind rose if we have season data
        if 'season' in df.columns:
            for season in df['season'].unique():
                # Filter data for this season
                season_data = wind_data[wind_data['season'] == season]

                if not season_data.empty:
                    # Count occurrences by direction and speed for this season
                    season_counts = season_data.groupby(['direction_bin', 'speed_bin']).size().reset_index(name='count')

                    # Create wind rose for this season
                    fig = px.bar_polar(
                        season_counts,
                        r='count',
                        theta='direction_bin',
                        color='speed_bin',
                        title=f'Wind Rose - {season}',
                        color_discrete_sequence=px.colors.sequential.Plasma_r
                    )
                    fig.write_html(f'reports/figures/wind_rose_{season}.html')

    # 4. Correlation between wind and temperature
    corr_data = []
    for feature in wind_cols:
        corr = df[[target_col, feature]].corr().iloc[0, 1]
        if not np.isnan(corr):
            corr_data.append({'Feature': feature, 'Correlation': corr})

    if corr_data:
        corr_df = pd.DataFrame(corr_data)
        corr_df = corr_df.sort_values('Correlation', ascending=False)

        fig = px.bar(
            corr_df,
            y='Feature',
            x='Correlation',
            orientation='h',
            title=f'Correlation between Wind Features and {target_col}',
            color='Correlation',
            color_continuous_scale='RdBu_r',
            range_color=[-1, 1]
        )
        fig.update_layout(height=600)
        fig.write_html('reports/figures/wind_correlations.html')

    # 5. Wind speed vs direction scatter plot
    if direction_cols and speed_cols:
        fig = px.scatter(
            wind_data,
            x=dir_col,
            y=spd_col,
            color=target_col,
            title=f'Wind Direction vs Speed (colored by {target_col})',
            labels={dir_col: 'Wind Direction (degrees)', spd_col: 'Wind Speed'},
            color_continuous_scale='Viridis'
        )
        fig.write_html('reports/figures/wind_direction_vs_speed.html')

    # 6. Wind speed vs temperature
    for speed_col in speed_cols:
        fig = px.scatter(
            df,
            x=speed_col,
            y=target_col,
            title=f'{target_col} vs {speed_col}',
            labels={speed_col: speed_col, target_col: target_col},
            trendline='ols'
        )
        fig.write_html(f'reports/figures/{target_col}_vs_{speed_col}.html')

    # 7. Wind distribution by season
    if 'season' in df.columns and speed_cols:
        wind_speed_col = speed_cols[0]  # Use the first wind speed column

        fig = px.box(
            df,
            x='season',
            y=wind_speed_col,
            title=f'{wind_speed_col} by Season',
            labels={'season': 'Season', wind_speed_col: 'Wind Speed'},
            category_orders={'season': ['Winter', 'Spring', 'Summer', 'Fall']}
        )
        fig.write_html('reports/figures/wind_speed_by_season.html')

def plot_precipitation_analysis(df, target_col='tmax'):
    """
    Create visualizations for precipitation-related features.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to analyze
    target_col : str, optional
        Target column
    """
    # Identify precipitation columns
    precip_cols = [col for col in df.columns if any(x in col for x in ['prcp', 'snow', 'precipitation', 'precip'])]
    precip_cols = [col for col in precip_cols if col in df.columns]

    if not precip_cols:
        print("No precipitation columns found")
        return

    print(f"Analyzing {len(precip_cols)} precipitation features")

    # Get key precipitation features for focused analysis
    key_features = []
    for term in ['prcp', 'snow', 'precipitation', 'ms_precipitation']:
        matches = [col for col in precip_cols if term in col]
        if matches:
            key_features.append(matches[0])

    # Keep only top 4 features for clarity
    key_features = key_features[:4]

    # 1. Time series analysis of precipitation
    if key_features and 'date' in df.columns:
        # Time series plot
        ts_data = df.dropna(subset=key_features + ['date']).sort_values('date')

        # Create subplot with multiple precipitation measures
        fig = make_subplots(rows=len(key_features), cols=1, shared_xaxes=True)

        for i, feature in enumerate(key_features):
            fig.add_trace(
                go.Scatter(x=ts_data['date'], y=ts_data[feature], name=feature),
                row=i + 1, col=1
            )
            fig.update_yaxes(title_text=feature, row=i + 1, col=1)

        fig.update_layout(
            height=800,
            title_text="Precipitation Features Over Time",
            showlegend=True
        )
        fig.write_html('reports/figures/precipitation_time_series.html')

    # 2. Monthly patterns of precipitation
    if 'month' in df.columns:
        monthly_data = []
        for feature in key_features:
            feature_data = df.groupby('month')[feature].mean().reset_index()
            feature_data['Feature'] = feature
            feature_data.rename(columns={feature: 'Value'}, inplace=True)
            monthly_data.append(feature_data)

        if monthly_data:
            monthly_df = pd.concat(monthly_data)

            fig = px.line(
                monthly_df,
                x='month',
                y='Value',
                color='Feature',
                title='Monthly Average Precipitation',
                labels={'month': 'Month', 'Value': 'Precipitation'}
            )
            fig.update_xaxes(dtick=1)
            fig.write_html('reports/figures/precipitation_monthly.html')

    # 3. Correlation between precipitation and temperature
    corr_data = []
    for feature in precip_cols:
        # skip if not numeric

        if not pd.api.types.is_numeric_dtype(df[feature]):
            continue

        corr = df[[target_col, feature]].corr().iloc[0, 1]
        if not np.isnan(corr):
            corr_data.append({'Feature': feature, 'Correlation': corr})

    if corr_data:
        corr_df = pd.DataFrame(corr_data)
        corr_df = corr_df.sort_values('Correlation', ascending=False)

        fig = px.bar(
            corr_df,
            y='Feature',
            x='Correlation',
            orientation='h',
            title=f'Correlation between Precipitation Features and {target_col}',
            color='Correlation',
            color_continuous_scale='RdBu_r',
            range_color=[-1, 1]
        )
        fig.update_layout(height=600)
        fig.write_html('reports/figures/precipitation_correlations.html')

    # 4. Precipitation by season
    if 'season' in df.columns:
        season_data = []
        for feature in key_features:
            feature_data = df.groupby('season')[feature].mean().reset_index()
            feature_data['Feature'] = feature
            feature_data.rename(columns={feature: 'Value'}, inplace=True)
            season_data.append(feature_data)

        if season_data:
            season_df = pd.concat(season_data)

            fig = px.bar(
                season_df,
                x='season',
                y='Value',
                color='Feature',
                barmode='group',
                title='Seasonal Average Precipitation',
                labels={'season': 'Season', 'Value': 'Precipitation'},
                category_orders={'season': ['Winter', 'Spring', 'Summer', 'Fall']}
            )
            fig.write_html('reports/figures/precipitation_seasonal.html')

    # 5. Precipitation vs Temperature scatter plots
    for feature in key_features:
        fig = px.scatter(
            df,
            x=feature,
            y=target_col,
            title=f'{target_col} vs {feature}',
            labels={feature: feature, target_col: target_col},
            trendline='ols'
        )
        fig.write_html(f'reports/figures/{target_col}_vs_{feature}_scatter.html')

    # 6. Precipitation lag analysis
    # Create lag variables if not already present
    lag_cols = []
    for feature in key_features:
        if f"{feature}_lag1" not in df.columns:
            lag_cols.append(feature)

    if lag_cols and 'date' in df.columns:
        # Create a copy to avoid modifying the original
        lag_df = df.copy()
        lag_df = lag_df.sort_values('date')

        # Create simple lag relationships
        for feature in lag_cols:
            for lag in [1, 7, 14]:
                lag_df[f"{feature}_lag{lag}"] = lag_df[feature].shift(lag)

        # Plot autocorrelation for precipitation features
        for feature in lag_cols:
            acf_values = []
            max_lag = 30

            feature_data = lag_df[feature].dropna()

            for lag in range(max_lag + 1):
                if lag == 0:
                    acf_values.append(1.0)  # ACF at lag 0 is always 1
                else:
                    # Calculate autocorrelation
                    shifted = feature_data.shift(lag).dropna()
                    # Align the series
                    original = feature_data.iloc[lag:]
                    shifted = shifted.iloc[:len(original)]
                    # Calculate correlation
                    correlation = np.corrcoef(original, shifted)[0, 1]
                    acf_values.append(correlation)

            # Calculate confidence intervals (95%)
            conf_level = 1.96 / np.sqrt(len(feature_data))

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
                title=f'Autocorrelation Function for {feature}',
                xaxis_title='Lag (days)',
                yaxis_title='Autocorrelation',
                showlegend=False
            )
            fig.write_html(f'reports/figures/{feature}_acf.html')

def analyze_feature_categories(df, target_col='tmax'):
    """
    Analyze feature categories and their relationships with the target.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to analyze
    target_col : str, optional
        Target column
    """
    print("\n=== Feature Categories Analysis ===")

    # Define feature categories
    feature_categories = {
        'Temperature Lags': [col for col in df.columns if any(x in col for x in ['tmin_lag', 'tavg_lag', 'ms_temp', 'vc_temp', 'nasa_temp'])],
        'Precipitation': [col for col in df.columns if any(x in col for x in ['prcp', 'snow', 'precipitation', 'precip'])],
        'Wind': [col for col in df.columns if 'wind' in col],
        'Atmospheric': [col for col in df.columns if any(x in col for x in ['pressure', 'humidity', 'cloud', 'visibility', 'uv'])],
        'Urban Heat': [col for col in df.columns if any(x in col for x in ['albedo', 'vegetation', 'canopy', 'impervious', 'building', 'water', 'heatmap'])],
        'Air Quality': [col for col in df.columns if any(x in col for x in ['ozone', 'carbon', 'dioxide', 'pm25'])],
        'Calendar': [col for col in df.columns if any(x in col for x in ['year', 'month', 'day', 'week', 'season', 'holiday', 'weekend'])]
    }

    # Filter categories to only include columns that exist in the dataset
    valid_categories = {}
    for category, cols in feature_categories.items():
        valid_cols = [col for col in cols if col in df.columns]
        if valid_cols:
            valid_categories[category] = valid_cols

    # For each category, analyze correlation with target
    for category, cols in valid_categories.items():
        print(f"\nAnalyzing {category} Features ({len(cols)} features)")

        # Calculate correlations with target
        category_df = df[[target_col] + cols].copy()
        corr_with_target = category_df.select_dtypes(include="number").corr()[target_col].drop(target_col).abs().sort_values(ascending=False)

        # Get top 5 correlations
        top_corrs = corr_with_target.head(5)
        print(f"Top correlations with {target_col}:")
        for feature, corr in top_corrs.items():
            print(f"  {feature}: {corr:.4f}")

        # Plot correlation heatmap for this category
        if len(cols) >= 2:
            # Include at most 15 features for readability
            if len(cols) > 15:
                # Get top correlated features
                top_features = corr_with_target.head(15).index.tolist()
                category_corr = category_df[[target_col] + top_features].corr()
            else:
                category_corr = category_df.select_dtypes(include="number").corr()

            fig = px.imshow(
                category_corr,
                color_continuous_scale='RdBu_r',
                title=f'Correlation Matrix: {category} Features',
                zmin=-1,
                zmax=1
            )
            fig.write_html(f'reports/figures/correlation_{category.lower().replace(" ", "_")}.html')

        # Create distribution plots for top 3 features
        top_3_features = corr_with_target.head(3).index.tolist()
        for feature in top_3_features:
            # Create box plot by month if we have month data
            if 'month' in df.columns:
                monthly_data = df.dropna(subset=[feature, 'month'])

                fig = px.box(
                    monthly_data,
                    x='month',
                    y=feature,
                    title=f'Monthly Distribution of {feature}',
                    labels={'month': 'Month', feature: feature}
                )
                fig.update_xaxes(type='category')
                fig.write_html(f'reports/figures/{feature}_monthly.html')

            # Create violin plot
            fig = px.violin(
                df,
                y=feature,
                box=True,
                title=f'Distribution of {feature}',
                labels={feature: feature}
            )
            fig.write_html(f'reports/figures/{feature}_distribution.html')

            # Create scatter plot with target
            fig = px.scatter(
                df,
                x=feature,
                y=target_col,
                title=f'{target_col} vs {feature}',
                labels={feature: feature, target_col: target_col},
                trendline='ols'
            )
            fig.write_html(f'reports/figures/{target_col}_vs_{feature}_scatter.html')

        # Create time series plot for the top feature
        if len(top_3_features) > 0 and 'date' in df.columns:
            top_feature = top_3_features[0]

            # Filter out missing values and sort by date
            ts_data = df.dropna(subset=[top_feature, 'date']).sort_values('date')

            fig = px.line(
                ts_data,
                x='date',
                y=[target_col, top_feature],
                title=f'{target_col} and {top_feature} Over Time',
                labels={'date': 'Date', 'value': 'Value', 'variable': 'Variable'}
            )
            fig.write_html(f'reports/figures/{top_feature}_time_series.html')

    # Create a summary correlation plot between categories
    # Calculate mean absolute correlation for each category
    category_summary = pd.DataFrame(index=valid_categories.keys(), columns=valid_categories.keys())
    numeric_df = category_df.select_dtypes(include="number")
    for cat1 in valid_categories.keys():
        for cat2 in valid_categories.keys():
            # Calculate mean correlation between all features in these categories
            if cat1 == cat2:
                # Perfect correlation with self
                category_summary.loc[cat1, cat2] = 1.0
            else:
                cat1_cols = valid_categories[cat1]
                cat2_cols = valid_categories[cat2]

                # Get all feature pairs
                corrs = []
                for c1 in cat1_cols:
                    for c2 in cat2_cols:
                        if c1 in numeric_df.columns and c2 in numeric_df.columns:
                            corr = numeric_df[[c1, c2]].corr().iloc[0, 1]
                            if not np.isnan(corr):
                                corrs.append(abs(corr))

                # Calculate mean correlation
                if corrs:
                    category_summary.loc[cat1, cat2] = np.mean(corrs)
                else:
                    category_summary.loc[cat1, cat2] = np.nan

    # Plot category correlations
    fig = px.imshow(
        category_summary,
        color_continuous_scale='Viridis',
        title='Mean Absolute Correlation Between Feature Categories',
        zmin=0,
        zmax=1
    )
    fig.write_html('reports/figures/category_correlations.html')

    return valid_categories

# -------------------------------------------------------
# Visualization Functions for Feature Categories
# -------------------------------------------------------

def plot_air_quality_analysis(df, target_col='tmax'):
    """
    Create visualizations for air quality features.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to analyze
    target_col : str, optional
        Target column
    """
    # Identify air quality columns
    aq_cols = [col for col in df.columns if any(x in col for x in ['ozone', 'carbon', 'nitrogen', 'pm25'])]
    aq_cols = [col for col in aq_cols if col in df.columns]

    if not aq_cols:
        print("No air quality columns found")
        return

    print(f"Analyzing {len(aq_cols)} air quality features")

    # Air quality seasonal patterns
    if 'month' in df.columns and 'season' in df.columns:
        # Create monthly averages
        monthly_data = []
        for feature in aq_cols:
            monthly_avg = df.groupby('month')[feature].mean().reset_index()
            monthly_avg['Feature'] = feature
            monthly_avg.rename(columns={feature: 'Value'}, inplace=True)
            monthly_data.append(monthly_avg)

        if monthly_data:
            monthly_df = pd.concat(monthly_data)

            fig = px.line(
                monthly_df,
                x='month',
                y='Value',
                color='Feature',
                title='Monthly Average Air Quality Measures',
                labels={'month': 'Month', 'Value': 'Value'},
                facet_col='Feature',
                facet_col_wrap=2
            )
            fig.update_xaxes(dtick=1)
            fig.update_layout(height=800)
            fig.write_html('reports/figures/air_quality_monthly.html')

        # Create seasonal boxplots
        seasonal_data = []
        for feature in aq_cols:
            feature_data = df[['season', feature]].copy()
            feature_data['Feature'] = feature
            feature_data.rename(columns={feature: 'Value'}, inplace=True)
            seasonal_data.append(feature_data)

        if seasonal_data:
            seasonal_df = pd.concat(seasonal_data)

            fig = px.box(
                seasonal_df,
                x='season',
                y='Value',
                color='Feature',
                title='Seasonal Distribution of Air Quality Measures',
                labels={'season': 'Season', 'Value': 'Value'},
                facet_col='Feature',
                facet_col_wrap=2
            )
            fig.update_layout(height=800)
            fig.write_html('reports/figures/air_quality_seasonal.html')

    # Relationship with temperature
    for feature in aq_cols:
        fig = px.scatter(
            df,
            x=feature,
            y=target_col,
            title=f'{target_col} vs {feature}',
            labels={feature: feature, target_col: target_col},
            trendline='ols'
        )
        fig.write_html(f'reports/figures/{target_col}_vs_{feature}.html')

    # Correlation matrix between air quality measures
    if len(aq_cols) >= 2:
        aq_df = df[aq_cols].dropna()
        aq_corr = aq_df.corr()

        fig = px.imshow(
            aq_corr,
            color_continuous_scale='RdBu_r',
            title='Correlation Between Air Quality Measures',
            zmin=-1,
            zmax=1
        )
        fig.write_html('reports/figures/air_quality_correlation.html')

def plot_calendar_analysis(df, target_col='tmax'):
    """
    Create visualizations for calendar/time-based features.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to analyze
    target_col : str, optional
        Target column
    """
    # Check if we have calendar features
    calendar_cols = [col for col in df.columns if any(x in col for x in [
        'year', 'month', 'day', 'dayofweek', 'dayofyear', 'week', 'season', 'quarter'
    ])]
    calendar_cols = [col for col in calendar_cols if col in df.columns]

    if not calendar_cols:
        print("No calendar columns found")
        return

    print(f"Analyzing {len(calendar_cols)} calendar features")

    # Temperature by day of week
    if 'dayofweek' in df.columns:
        fig = px.box(
            df,
            x='dayofweek',
            y=target_col,
            title=f'{target_col} by Day of Week',
            labels={'dayofweek': 'Day of Week (0=Monday)', target_col: target_col}
        )
        fig.update_xaxes(type='category')
        fig.write_html(f'reports/figures/{target_col}_by_dayofweek.html')

    # Temperature by month
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

    # Temperature by season
    if 'season' in df.columns:
        fig = px.box(
            df,
            x='season',
            y=target_col,
            title=f'{target_col} by Season',
            labels={'season': 'Season', target_col: target_col},
            category_orders={'season': ['Winter', 'Spring', 'Summer', 'Fall']}
        )
        fig.write_html(f'reports/figures/{target_col}_by_season.html')

    # Day of year pattern
    if 'dayofyear' in df.columns:
        # Calculate average by day of year
        daily_avg = df.groupby('dayofyear')[target_col].agg(['mean', 'min', 'max']).reset_index()

        fig = go.Figure()

        # Add mean line
        fig.add_trace(go.Scatter(
            x=daily_avg['dayofyear'],
            y=daily_avg['mean'],
            mode='lines',
            name='Mean',
            line=dict(color='blue', width=2)
        ))

        # Add min/max range
        fig.add_trace(go.Scatter(
            x=daily_avg['dayofyear'],
            y=daily_avg['max'],
            mode='lines',
            name='Max',
            line=dict(color='red', width=1, dash='dash')
        ))

        fig.add_trace(go.Scatter(
            x=daily_avg['dayofyear'],
            y=daily_avg['min'],
            mode='lines',
            name='Min',
            line=dict(color='green', width=1, dash='dash')
        ))

        fig.update_layout(
            title=f'Annual Cycle of {target_col}',
            xaxis_title='Day of Year',
            yaxis_title=target_col
        )
        fig.write_html(f'reports/figures/{target_col}_annual_pattern.html')

if __name__ == "__main__":
    main()