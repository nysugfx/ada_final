"""
Central Park Temperature Prediction - Data Setup Script
===============================================
This script helps set up a project to predict high temperatures in Central Park, New York
using historical NOAA data and additional factors that influence urban temperatures.

The project uses:
1. NOAA's Climate Data Online (CDO) API to gather historical temperature data
2. Additional data sources for factors that influence urban temperatures
3. Python libraries focusing on Plotly for visualization

Station ID for Central Park, NY: GHCND:USW00094728
"""

import os
import time

import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Set your NOAA API token here (you'll need to request one from NOAA)
NOAA_TOKEN = os.getenv("NOAA_TOKEN") # Request from https://www.ncdc.noaa.gov/cdo-web/token

# Constants for data retrieval
BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
CENTRAL_PARK_STATION = "GHCND:USW00094728"  # Central Park station ID
DATASET_ID = "GHCND"  # Global Historical Climatology Network - Daily
DATA_TYPES = ["TMAX"]  # Maximum temperature (target variable)


def create_project_folders():
    """Create necessary folders for the project"""
    folders = ['data/raw', 'data/processed', 'models', 'notebooks', 'reports', 'figures']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print("Project folder structure created.")


def fetch_noaa_data(start_date, end_date, datatype_id="TMAX", limit=1000):
    """
    Fetch data from NOAA's Climate Data Online API

    Parameters:
    -----------
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    datatype_id : str
        Data type ID to fetch (default: TMAX - maximum temperature)
    limit : int
        Number of results to return per request (max 1000)

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the requested data
    """
    headers = {
        "token": NOAA_TOKEN
    }

    params = {
        "datasetid": DATASET_ID,
        "stationid": CENTRAL_PARK_STATION,
        "datatypeid": datatype_id,
        "startdate": start_date,
        "enddate": end_date,
        "limit": limit,
        "units": "standard"
    }

    url = f"{BASE_URL}/data"

    all_data = []
    offset = 0

    while True:
        params["offset"] = offset
        time.sleep(0.25)
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            break

        data = response.json()

        if "results" not in data or len(data["results"]) == 0:
            break

        all_data.extend(data["results"])

        if len(data["results"]) < limit:
            break

        offset += limit

    df = pd.DataFrame(all_data)

    if not df.empty:
        # Convert date strings to datetime
        df['date'] = pd.to_datetime(df['date'])

        # If the datatype is temperature, convert from tenths of a degree C to degrees F
        # if datatype_id in ["TMAX", "TMIN", "TAVG"]:
        #     df['value'] = df['value'] / 10 * 9 / 5 + 32

    return df


def get_metadata():
    """Get metadata about available datatypes and stations"""
    headers = {"token": NOAA_TOKEN}

    # Get information about the Central Park station
    station_url = f"{BASE_URL}/stations/{CENTRAL_PARK_STATION}"
    station_response = requests.get(station_url, headers=headers)

    if station_response.status_code == 200:
        station_data = station_response.json()
        print(f"Station information for {station_data.get('name', 'Unknown')}:")
        print(f"  Location: {station_data.get('latitude', 'Unknown')}, {station_data.get('longitude', 'Unknown')}")
        print(f"  Elevation: {station_data.get('elevation', 'Unknown')}")
    else:
        print(f"Error fetching station metadata: {station_response.status_code}")

    # Get available datatypes for this station
    datatypes_url = f"{BASE_URL}/datatypes"
    params = {
        "stationid": CENTRAL_PARK_STATION,
        "limit": 100
    }

    datatypes_response = requests.get(datatypes_url, headers=headers, params=params)

    if datatypes_response.status_code == 200:
        datatypes_data = datatypes_response.json()
        print(f"\nAvailable data types for this station:")
        for datatype in datatypes_data.get("results", []):
            print(f"  {datatype['id']}: {datatype['name']}")
    else:
        print(f"Error fetching datatypes: {datatypes_response.status_code}")


def fetch_full_dataset(start_year, end_year, save=True):
    """
    Fetch multiple years of data and combine them into a single dataset

    Parameters:
    -----------
    start_year : int
        First year to fetch data for
    end_year : int
        Last year to fetch data for (inclusive)
    save : bool
        Whether to save the resulting dataset to a CSV file

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing all fetched data
    """
    all_data = []

    for year in range(start_year, end_year + 1):
        print(f"Fetching data for {year}...")
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        # Fetch maximum temperature (TMAX)
        tmax_df = fetch_noaa_data(start_date, end_date, "TMAX")
        if not tmax_df.empty:
            tmax_df = tmax_df.rename(columns={"value": "tmax"})
            tmax_df = tmax_df[["date", "tmax"]]
            all_data.append(tmax_df)

        # Add other data types as needed
        # For example, to get precipitation data:
        # prcp_df = fetch_noaa_data(start_date, end_date, "PRCP")
        # if not prcp_df.empty:
        #     prcp_df = prcp_df.rename(columns={"value": "prcp"})
        #     prcp_df = prcp_df[["date", "prcp"]]
        #     all_data.append(prcp_df)

    if not all_data:
        print("No data was fetched.")
        return pd.DataFrame()

    # Combine all dataframes
    df = pd.concat(all_data, ignore_index=True)

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    if save:
        output_path = "data/raw/central_park_temperature_data.csv"
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")

    return df


def add_features(df):
    """
    Add additional features to the dataset that may help with prediction

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing temperature data

    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional features
    """
    # Extract date features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    # Calculate moving averages for different windows
    df['tmax_ma3'] = df['tmax'].rolling(window=3, min_periods=1).mean()
    df['tmax_ma7'] = df['tmax'].rolling(window=7, min_periods=1).mean()
    df['tmax_ma30'] = df['tmax'].rolling(window=30, min_periods=1).mean()

    # Add seasonal features using sine and cosine transformations
    df['season_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['season_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

    # Here you would add additional features from other data sources
    # - Urban heat island factors (could be static values for Central Park)
    # - Satellite data on ground cover/vegetation
    # - Pollution data
    # - Weather forecasts from previous days

    return df


def visualize_temperature_data(df):
    """
    Create visualizations of the temperature data

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing temperature data
    """
    # Time series plot of high temperatures
    fig1 = px.line(df, x='date', y='tmax', title='Daily Maximum Temperature in Central Park, NY')
    fig1.update_layout(
        xaxis_title='Date',
        yaxis_title='Maximum Temperature (°F)',
        template='plotly_white'
    )
    fig1.write_html('figures/temperature_timeseries.html')

    # Monthly temperature distributions by year (boxplot)
    df_monthly = df.copy()
    df_monthly['year_month'] = df_monthly['date'].dt.strftime('%Y-%m')
    df_monthly['month_name'] = df_monthly['date'].dt.strftime('%b')

    fig2 = px.box(df_monthly, x='month_name', y='tmax', color='year',
                  title='Monthly Distribution of Maximum Temperatures by Year')
    fig2.update_layout(
        xaxis_title='Month',
        yaxis_title='Maximum Temperature (°F)',
        xaxis={'categoryorder': 'array', 'categoryarray': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']},
        template='plotly_white'
    )
    fig2.write_html('figures/monthly_temperature_distributions.html')

    # Seasonal pattern visualization
    df_seasonal = df.groupby(['month', 'day'])['tmax'].mean().reset_index()
    # Convert month and day to a date format for easier plotting
    df_seasonal['date'] = df_seasonal.apply(lambda x: datetime(2000, int(x['month']), int(x['day'])), axis=1)
    df_seasonal = df_seasonal.sort_values('date')

    fig3 = px.line(df_seasonal, x='date', y='tmax',
                   title='Average Maximum Temperature by Day of Year')
    fig3.update_layout(
        xaxis_title='Day of Year',
        yaxis_title='Average Maximum Temperature (°F)',
        xaxis_tickformat='%b %d',
        template='plotly_white'
    )
    fig3.write_html('figures/seasonal_pattern.html')

    print("Visualizations saved to figures/ directory")


def main():
    """Main function to execute the script"""
    # Create project structure
    create_project_folders()

    # Get information about the data
    get_metadata()

    # Example: Fetch data for the last 5 years
    current_year = datetime.now().year
    start_year = current_year - 5

    print(f"\nFetching temperature data from {start_year} to {current_year}...")

    # In a real project, you might want to fetch more historical data
    df = fetch_full_dataset(start_year, current_year)

    if not df.empty:
        # Add features
        df_with_features = add_features(df)

        # Save processed data
        df_with_features.to_csv("data/processed/central_park_temp_with_features.csv", index=False)

        # Create visualizations
        visualize_temperature_data(df)

        print("\nNext steps:")
        print("1. Get a NOAA API token from https://www.ncdc.noaa.gov/cdo-web/token")
        print("2. Add your token to the script")
        print("3. Run this script to fetch the data")
        print("4. Expand the dataset with additional features based on urban heat island factors")
        print("5. Create predictive models in the notebooks/ directory")
    else:
        print("No data was fetched. Please check your API token and try again.")


if __name__ == "__main__":
    main()