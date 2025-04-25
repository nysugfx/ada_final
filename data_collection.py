import os
import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Create necessary directories
def create_project_structure():
    """Create the project directory structure"""
    directories = [
        'data/raw',
        'data/processed',
        'data/interim',
        'notebooks',
        'models',
        'reports/figures'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print("Project directory structure created.")

#########################################################
# 1. NOAA Temperature Data (Primary Dataset)
#########################################################

def fetch_noaa_temperature_data(token, station_id="GHCND:USW00094728", start_year=2015, end_year=2023):
    """
    Fetch NOAA temperature data for Central Park station

    Parameters:
    -----------
    token : str
        NOAA API token
    station_id : str
        Station identifier for Central Park
    start_year : int
        Start year for data collection
    end_year : int
        End year for data collection

    Returns:
    --------
    pandas.DataFrame
        Combined temperature data
    """
    base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
    headers = {"token": token}

    # Data types to fetch
    data_types = ["TMAX", "TMIN", "TAVG", "PRCP", "SNOW"]

    all_data = {}

    # Fetch data for each year and data type
    for year in range(start_year, end_year + 1):
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        for data_type in data_types:
            print(f"Fetching {data_type} data for {year}...")

            params = {
                "datasetid": "GHCND",
                "stationid": station_id,
                "datatypeid": data_type,
                "startdate": start_date,
                "enddate": end_date,
                "limit": 1000,
                "units": "standard"
            }

            all_records = []
            offset = 0

            while True:
                params["offset"] = offset
                response = requests.get(base_url, headers=headers, params=params)
                time.sleep(0.35)
                if response.status_code != 200:
                    print(f"Error: {response.status_code}")
                    print(response.text)
                    break

                data = response.json()

                if "results" not in data or len(data["results"]) == 0:
                    break

                all_records.extend(data["results"])

                if len(data["results"]) < params["limit"]:
                    break

                offset += params["limit"]

            # Process the data
            if all_records:
                df = pd.DataFrame(all_records)
                df['date'] = pd.to_datetime(df['date'])

                # Convert temperature from tenths of degrees C to F
                if data_type in ["TMAX", "TMIN", "TAVG"]:
                    df['value'] = df['value'] / 10 * 9/5 + 32

                # Rename value column to the data type
                df = df.rename(columns={"value": data_type.lower()})

                # Keep only date and value
                df = df[['date', data_type.lower()]]

                if data_type in all_data:
                    all_data[data_type] = pd.concat([all_data[data_type], df])
                else:
                    all_data[data_type] = df

    # Merge all data types
    merged_data = all_data["TMAX"]

    for data_type in all_data:
        if data_type != "TMAX":
            merged_data = pd.merge(merged_data, all_data[data_type], on="date", how="outer")

    # Sort by date
    merged_data = merged_data.sort_values("date").reset_index(drop=True)

    # Save raw data
    merged_data.to_csv("data/raw/noaa_temperature_data.csv", index=False)

    print(f"NOAA temperature data collected with {len(merged_data)} rows and {merged_data.columns.size} columns")
    return merged_data

#########################################################
# 2. Weather API Data (Additional Weather Variables)
#########################################################

def fetch_visual_crossing_weather_data(api_key, location="Central Park, NY", start_date="2015-01-01", end_date="2023-12-31"):
    """
    Fetch historical weather data from Visual Crossing Weather API
    Split into smaller queries to avoid quota limits

    Parameters:
    -----------
    api_key : str
        Visual Crossing API key
    location : str
        Location to fetch weather for
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format

    Returns:
    --------
    pandas.DataFrame
        Weather data including humidity, wind, etc.
    """
    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"

    params = {
        "key": api_key,
        "include": "days",
        "elements": "datetime,tempmax,tempmin,temp,humidity,precip,precipprob,preciptype,windspeed,winddir,pressure,cloudcover,visibility,uvindex,conditions",
        "unitGroup": "us"
    }

    # Convert date strings to datetime objects
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Split into 90-day chunks to stay within API limits
    all_daily_data = []
    current_start = start

    print(f"Fetching weather data from Visual Crossing for {location} in chunks...")

    while current_start <= end:
        # Calculate end of current chunk (90 days or end date, whichever comes first)
        current_end = min(current_start + pd.Timedelta(days=90), end)

        # Format dates for API
        current_start_str = current_start.strftime('%Y-%m-%d')
        current_end_str = current_end.strftime('%Y-%m-%d')

        # Construct URL for this chunk
        url = f"{base_url}/{location}/{current_start_str}/{current_end_str}"

        print(f"  Fetching chunk: {current_start_str} to {current_end_str}...")

        try:
            response = requests.get(url, params=params)

            if response.status_code != 200:
                print(f"  Error: {response.status_code}")
                print(f"  {response.text}")
            else:
                data = response.json()

                # Extract daily data
                for day in data.get("days", []):
                    all_daily_data.append({
                        "date": day.get("datetime"),
                        "vc_tempmax": day.get("tempmax"),
                        "vc_tempmin": day.get("tempmin"),
                        "vc_temp": day.get("temp"),
                        "humidity": day.get("humidity"),
                        "precipitation": day.get("precip"),
                        "precip_probability": day.get("precipprob"),
                        "precip_type": day.get("preciptype"),
                        "wind_speed": day.get("windspeed"),
                        "wind_direction": day.get("winddir"),
                        "pressure": day.get("pressure"),
                        "cloud_cover": day.get("cloudcover"),
                        "visibility": day.get("visibility"),
                        "uv_index": day.get("uvindex"),
                        "conditions": day.get("conditions")
                    })

                print(f"  Successfully retrieved {len(data.get('days', []))} days of data")

            # Move to next chunk
            current_start = current_end + pd.Timedelta(days=1)

            # Add a pause to avoid hitting rate limits
            time.sleep(1)

        except Exception as e:
            print(f"  Error fetching chunk: {e}")
            current_start = current_end + pd.Timedelta(days=1)

    # Create DataFrame from collected data
    if all_daily_data:
        df = pd.DataFrame(all_daily_data)
        df['date'] = pd.to_datetime(df['date'])

        # Save raw data
        df.to_csv("data/raw/visual_crossing_weather_data.csv", index=False)

        print(f"Visual Crossing weather data collected with {len(df)} rows and {df.columns.size} columns")
        return df
    else:
        print("No data was collected from Visual Crossing")
        return None


def fetch_meteostat_weather_data(lat=40.7829, lon=-73.9654, start_date="2015-01-01", end_date="2023-12-31", alt=48.0):
    """
    Fetch weather data using meteostat Python library instead of Visual Crossing API
    Meteostat doesn't require an API key and has generous usage limits

    Parameters:
    -----------
    lat : float
        Latitude for Central Park
    lon : float
        Longitude for Central Park
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    alt : float
        Altitude in meters

    Returns:
    --------
    pandas.DataFrame
        Weather data including temperature, humidity, etc.
    """
    from meteostat import Point, Daily, Hourly
    import pandas as pd

    # Create Point for Central Park
    point = Point(lat, lon, alt)

    # Convert dates to datetime
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    print(f"Fetching Meteostat weather data for Central Park...")

    # Fetch daily data
    data = Daily(point, start, end)
    daily_data = data.fetch()

    # Reset index to make date a column
    daily_data = daily_data.reset_index()

    # Rename columns to be consistent with our schema
    column_map = {
        'time': 'date',
        'tavg': 'ms_temp',  # Average temperature (°C)
        'tmin': 'ms_tempmin',  # Minimum temperature (°C)
        'tmax': 'ms_tempmax',  # Maximum temperature (°C)
        'prcp': 'ms_precipitation',  # Precipitation (mm)
        'snow': 'ms_snow',  # Snow depth (mm)
        'wdir': 'ms_wind_direction',  # Wind direction (°)
        'wspd': 'ms_wind_speed',  # Wind speed (km/h)
        'wpgt': 'ms_wind_gust',  # Wind gust (km/h)
        'pres': 'ms_pressure',  # Pressure (hPa)
        'tsun': 'ms_sunshine',  # Sunshine (minutes)
        'coco': 'ms_weather_condition'  # Weather condition code
    }

    daily_data = daily_data.rename(columns=column_map)

    # Convert temperature from C to F
    if 'ms_temp' in daily_data.columns:
        daily_data['ms_temp'] = daily_data['ms_temp'] * 9 / 5 + 32
    if 'ms_tempmin' in daily_data.columns:
        daily_data['ms_tempmin'] = daily_data['ms_tempmin'] * 9 / 5 + 32
    if 'ms_tempmax' in daily_data.columns:
        daily_data['ms_tempmax'] = daily_data['ms_tempmax'] * 9 / 5 + 32

    # Convert precipitation from mm to inches
    if 'ms_precipitation' in daily_data.columns:
        daily_data['ms_precipitation'] = daily_data['ms_precipitation'] * 0.0393701

    # Convert snow from mm to inches
    if 'ms_snow' in daily_data.columns:
        daily_data['ms_snow'] = daily_data['ms_snow'] * 0.0393701

    # Convert wind speed from km/h to mph
    if 'ms_wind_speed' in daily_data.columns:
        daily_data['ms_wind_speed'] = daily_data['ms_wind_speed'] * 0.621371

    # Save raw data
    daily_data.to_csv("data/raw/meteostat_weather_data.csv", index=False)

    print(f"Meteostat weather data collected with {len(daily_data)} rows and {daily_data.columns.size} columns")
    return daily_data



#########################################################
# 3. EPA Air Quality Data
#########################################################

def fetch_epa_air_quality_data(api_key=None, email="adf2157@columbia.edu", start_date="2015-01-01",
                               end_date="2023-12-31"):
    """
    Fetch EPA Air Quality System (AQS) data using the byBox query with different parameters

    Parameters:
    -----------
    api_key : str
        EPA API key
    email : str
        Email for EPA API access
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format

    Returns:
    --------
    pandas.DataFrame
        Air quality data
    """
    # If no API key provided, use synthetic data
    if not api_key:
        print("No EPA API key provided. Generating synthetic air quality data...")
        return generate_synthetic_air_quality_data(start_date, end_date)

    # Since specific sites aren't returning data, try using byCounty instead
    # This should give us all sites in New York County (Manhattan)
    base_url = "https://aqs.epa.gov/data/api/dailyData/byCounty"

    # List of parameters to collect
    # 44201 - Ozone
    # 42101 - Carbon Monoxide
    # 42602 - Nitrogen Dioxide
    # 88101 - PM2.5
    parameters = ["44201", "42101", "42602", "88101"]

    # New York County (Manhattan) = state code 36, county code 061
    state_code = "36"
    county_code = "061"

    all_data = []

    for param in parameters:
        print(f"Fetching EPA data for parameter {param} in New York County...")

        # Break the date range into yearly chunks (API limitation)
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        current_start = start

        while current_start <= end:
            current_end = min(datetime(current_start.year, 12, 31), end)

            params = {
                "email": email,
                "key": api_key,
                "param": param,
                "bdate": current_start.strftime("%Y%m%d"),
                "edate": current_end.strftime("%Y%m%d"),
                "state": state_code,
                "county": county_code
            }

            try:
                print(f"  Requesting data for {current_start.year}...")
                response = requests.get(base_url, params=params)

                if response.status_code != 200:
                    print(f"  Error: {response.status_code}")
                    print(f"  {response.text}")
                else:
                    data = response.json()

                    # Check if there's data
                    if "Data" in data and len(data["Data"]) > 0:
                        print(f"  Found {len(data['Data'])} records")
                        for record in data["Data"]:
                            record["parameter"] = param
                            all_data.append(record)
                    else:
                        print(f"  No data found in response: {data.get('Header', {})}")

            except Exception as e:
                print(f"  Error fetching EPA data: {e}")

            current_start = current_end + timedelta(days=1)
            time.sleep(1)  # Pause to avoid rate limits

    df = pd.DataFrame(all_data)

    if "date_local" in df.columns:
        df["date"] = pd.to_datetime(df["date_local"])

    # For simplicity, average values by date and parameter
    if "parameter" in df.columns and "arithmetic_mean" in df.columns and len(df) > 0:
        pivot_df = df.pivot_table(
            index="date",
            columns="parameter",
            values="arithmetic_mean",
            aggfunc="mean"
        ).reset_index()

        # Rename columns with parameter names
        param_names = {
            "44201": "ozone",
            "42101": "carbon_monoxide",
            "42602": "nitrogen_dioxide",
            "88101": "pm25"
        }

        pivot_df = pivot_df.rename(columns=param_names)

        # Save raw data
        pivot_df.to_csv("data/raw/epa_air_quality_data.csv", index=False)

        print(f"EPA air quality data collected with {len(pivot_df)} rows and {pivot_df.columns.size} columns")
        return pivot_df

#########################################################
# 4. NASA Solar and Earth Data
#########################################################

def fetch_nasa_power_data(lat=40.7829, lon=-73.9654, start_date="20150101", end_date="20231231"):
    """
    Fetch NASA POWER (Prediction of Worldwide Energy Resources) data
    for solar radiation and related variables

    Parameters:
    -----------
    lat : float
        Latitude for Central Park
    lon : float
        Longitude for Central Park
    start_date : str
        Start date in YYYYMMDD format
    end_date : str
        End date in YYYYMMDD format

    Returns:
    --------
    pandas.DataFrame
        Solar radiation and related data
    """
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"

    params = {
        "parameters": "T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,WS2M,ALLSKY_SFC_SW_DWN,ALLSKY_SFC_LW_DWN",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "JSON"
    }

    print("Fetching NASA POWER data...")

    response = requests.get(base_url, params=params)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

    data = response.json()

    # Extract the data
    if "properties" in data and "parameter" in data["properties"]:
        parameters = data["properties"]["parameter"]

        # Convert to dataframe
        records = []

        for date_str in parameters["T2M"]:
            record = {
                "date": pd.to_datetime(date_str),
                "nasa_temp_avg": parameters["T2M"].get(date_str),
                "nasa_temp_max": parameters["T2M_MAX"].get(date_str),
                "nasa_temp_min": parameters["T2M_MIN"].get(date_str),
                "nasa_humidity": parameters["RH2M"].get(date_str),
                "nasa_precipitation": parameters["PRECTOTCORR"].get(date_str),
                "nasa_wind_speed": parameters["WS2M"].get(date_str),
                "solar_radiation": parameters["ALLSKY_SFC_SW_DWN"].get(date_str),
                "longwave_radiation": parameters["ALLSKY_SFC_LW_DWN"].get(date_str)
            }

            records.append(record)

        df = pd.DataFrame(records)

        # Save raw data
        df.to_csv("data/raw/nasa_power_data.csv", index=False)

        print(f"NASA POWER data collected with {len(df)} rows and {df.columns.size} columns")
        return df

    print("Failed to extract NASA POWER data.")
    return None

#########################################################
# 5. Urban Heat Island Data
#########################################################

def create_urban_heat_island_features():
    """
    Create synthetic urban heat island features based on research

    Note: In a real project, you would gather these from sources like:
    - NYC Open Data
    - Satellite imagery analysis
    - EPA's Smart Location Database
    - Urban Atlases

    Returns:
    --------
    pandas.DataFrame
        Urban heat island features by date
    """
    # Create date range
    date_range = pd.date_range(start="2015-01-01", end="2023-12-31")

    # Base values for NYC/Central Park area
    base_values = {
        "albedo": 0.25,  # Reflectivity (higher is more reflective)
        "vegetation_index": 0.65,  # Normalized Difference Vegetation Index for Central Park
        "tree_canopy_percent": 0.60,  # Approximate percentage of tree coverage
        "impervious_surface_percent": 0.20,  # Roads, paths, buildings
        "building_density": 0.10,  # Lower in the park, higher in surrounding areas
        "water_surface_percent": 0.05,  # Ponds, lakes, etc.
        "distance_to_water_km": 2.0,  # Approximate distance to Hudson/East River
        "heatmap_intensity": 0.4  # Relative intensity on urban heat maps (0-1)
    }

    # Seasonal variations (multipliers for certain variables)
    month_to_season = {
        1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer', 9: 'fall', 10: 'fall',
        11: 'fall', 12: 'winter'
    }

    seasonal_factors = {
        'winter': {
            'vegetation_index': 0.7,  # Lower in winter
            'albedo': 1.2  # Higher when snow is present
        },
        'spring': {
            'vegetation_index': 0.9,  # Increasing
            'albedo': 1.0  # Normal
        },
        'summer': {
            'vegetation_index': 1.1,  # Higher in summer
            'albedo': 0.9  # Lower due to darker surfaces
        },
        'fall': {
            'vegetation_index': 1.0,  # Normal
            'albedo': 1.0  # Normal
        }
    }

    # Create records with some annual trend and seasonal variations
    records = []

    for date in date_range:
        # Get seasonal factors
        season = month_to_season[date.month]
        season_factor = seasonal_factors[season]

        # Create mild annual trends (e.g., increasing building density, decreasing vegetation)
        year_factor = (date.year - 2015) / (2023 - 2015)

        record = {
            "date": date,
            "albedo": base_values["albedo"] * season_factor.get("albedo", 1.0),
            "vegetation_index": base_values["vegetation_index"] * season_factor.get("vegetation_index", 1.0) * (1 - 0.05 * year_factor),
            "tree_canopy_percent": base_values["tree_canopy_percent"] * (1 - 0.03 * year_factor),
            "impervious_surface_percent": base_values["impervious_surface_percent"] * (1 + 0.10 * year_factor),
            "building_density": base_values["building_density"] * (1 + 0.15 * year_factor),
            "water_surface_percent": base_values["water_surface_percent"],
            "distance_to_water_km": base_values["distance_to_water_km"],
            "heatmap_intensity": base_values["heatmap_intensity"] * (1 + 0.20 * year_factor)
        }

        # Add small random variations
        for key in record:
            if key != "date":
                record[key] *= 1 + np.random.normal(0, 0.02)  # 2% random variation

        records.append(record)

    df = pd.DataFrame(records)

    # Save raw data
    df.to_csv("data/raw/urban_heat_island_data.csv", index=False)

    print(f"Urban heat island features created with {len(df)} rows and {df.columns.size} columns")
    return df


#########################################################
# 7. Merge All Datasets
#########################################################

def merge_all_datasets(datasets):
    """
    Merge all collected datasets into a single dataframe

    Parameters:
    -----------
    datasets : dict
        Dictionary of dataframes with keys as dataset names

    Returns:
    --------
    pandas.DataFrame
        Merged dataset
    """
    # Start with the main temperature dataset
    if "temperature" not in datasets:
        print("Error: Temperature dataset is required")
        return None

    merged_df = datasets["temperature"]

    # Merge other datasets
    for name, df in datasets.items():
        if name != "temperature":
            print(f"Merging {name} dataset...")
            merged_df = pd.merge(merged_df, df, on="date", how="left")

    # Handle missing values
    # For this example, we'll just count and report them
    missing_values = merged_df.isnull().sum()
    print("\nMissing values in merged dataset:")
    print(missing_values[missing_values > 0])

    # Save the merged dataset
    merged_df.to_csv("data/processed/merged_dataset.csv", index=False)

    print(f"\nFinal merged dataset created with {len(merged_df)} rows and {merged_df.columns.size} columns")
    return merged_df

#########################################################
# 8. Data Summary and Visualization
#########################################################

def summarize_dataset(df):
    """
    Create a summary of the dataset

    Parameters:
    -----------
    df : pandas.DataFrame
        Merged dataset
    """
    # Basic statistics
    print("\nDataset Summary:")
    print(f"Time period: {df['date'].min()} to {df['date'].max()}")
    print(f"Total features: {df.columns.size - 1}")  # Excluding date
    print(f"Total observations: {len(df)}")

    # Check for missing values
    missing = df.isnull().sum()
    print("\nFeatures with missing values:")
    print(missing[missing > 0])

    # Correlation with target variable
    if "tmax" in df.columns:
        correlations = df.corr()["tmax"].sort_values(ascending=False)
        print("\nTop 10 features correlated with maximum temperature:")
        print(correlations.head(11))  # 11 because it includes the target itself

        # Save correlations
        correlations.to_csv("data/processed/feature_correlations.csv")

    # Create correlation matrix visualization
    corr_matrix = df.select_dtypes(include=[np.number]).corr()

    fig = px.imshow(
        corr_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1
    )

    fig.write_html("reports/figures/correlation_matrix.html")

    # Visualize the target variable over time
    if "tmax" in df.columns:
        fig = px.line(
            df,
            x="date",
            y="tmax",
            title="Maximum Temperature Over Time"
        )

        fig.write_html("reports/figures/temperature_timeseries.html")

    print("\nDataset summary and visualizations created")

#########################################################
# Main Execution Function
#########################################################

def collect_all_data(noaa_token=None, visualcrossing_key=None, epa_key=None):
    """
    Execute the complete data collection pipeline

    Parameters:
    -----------
    noaa_token : str
        NOAA API token
    visualcrossing_key : str
        Visual Crossing API key
    epa_key : str
        EPA API key

    Returns:
    --------
    pandas.DataFrame
        Final merged dataset
    """
    # Create project structure
    create_project_structure()

    # Dictionary to hold all datasets
    datasets = {}

    # 1. Collect NOAA temperature data
    # if noaa_token:
    #     temp_data = fetch_noaa_temperature_data(noaa_token)
    #     datasets["temperature"] = temp_data
    # else:
    #     print("NOAA token not provided. Skipping temperature data collection.")
    #     # Create a date range as a fallback
    #     date_range = pd.date_range(start="2015-01-01", end="2023-12-31")
    #     datasets["temperature"] = pd.DataFrame({"date": date_range})
    #
    # # 2. Collect Visual Crossing weather data
    # weather_data = fetch_meteostat_weather_data()
    # if weather_data is not None:
    #     datasets["weather"] = weather_data
    # else:
    #     print("Visual Crossing API key not provided. Skipping weather data collection.")

    # 3. Collect EPA air quality data
    if epa_key:
        air_quality_data = fetch_epa_air_quality_data(epa_key)
        if air_quality_data is not None:
            datasets["air_quality"] = air_quality_data
    else:
        print("EPA API key not provided. Skipping air quality data collection.")

    # 4. Collect NASA POWER data
    nasa_data = fetch_nasa_power_data()
    if nasa_data is not None:
        datasets["nasa"] = nasa_data

    # 5. Create urban heat island features
    uhi_data = create_urban_heat_island_features()
    datasets["urban_heat"] = uhi_data


    # 7. Merge all datasets
    merged_dataset = merge_all_datasets(datasets)

    # 8. Summarize the dataset
    if merged_dataset is not None:
        summarize_dataset(merged_dataset)

    return merged_dataset

# Example usage
if __name__ == "__main__":
    # Replace with your actual API keys
    noaa_token = os.getenv("NOAA_TOKEN")
    visualcrossing_key = os.getenv("VISUALCROSSING_KEY")
    epa_key = os.getenv("EPA_TOKEN")

    # Collect all data
    dataset = collect_all_data(noaa_token, visualcrossing_key, epa_key)

    print("\nData collection complete!")
    print("The dataset is ready for exploratory data analysis and modeling.")
