# NYC Central Park Temperature Prediction Dataset Guide

This repository contains a collection of datasets used for predicting high temperatures in Central Park, New York City. The data comes from various sources including NOAA, Visual Crossing Weather, NASA, and urban heat island measurements.

## Dataset Overview

The project combines the following datasets:
1. NOAA temperature data
2. Visual Crossing weather data
3. Urban heat island data
4. NASA POWER meteorological data
5. Meteostat weather data
6. Holiday/event data
7. EPA air quality data
8. Central Park specific temperature data

## Variables and Descriptions

### Temperature Variables (tmax is the target, please lag the rest)

| Variable | Source | Unit | Description |
|----------|--------|------|-------------|
| `tmax` | NOAA | °F | Maximum daily temperature |
| `tmin` | NOAA | °F | Minimum daily temperature |
| `tavg` | NOAA | °F | Average daily temperature |
| `ms_temp` | Meteostat | °F | Average daily temperature from Meteostat |
| `ms_tempmin` | Meteostat | °F | Minimum daily temperature from Meteostat |
| `ms_tempmax` | Meteostat | °F | Maximum daily temperature from Meteostat |
| `vc_temp` | Visual Crossing | °F | Average daily temperature from Visual Crossing |
| `vc_tempmin` | Visual Crossing | °F | Minimum daily temperature from Visual Crossing |
| `vc_tempmax` | Visual Crossing | °F | Maximum daily temperature from Visual Crossing |
| `nasa_temp_avg` | NASA POWER | °F | Average daily temperature from NASA |
| `nasa_temp_max` | NASA POWER | °F | Maximum daily temperature from NASA |
| `nasa_temp_min` | NASA POWER | °F | Minimum daily temperature from NASA |

### Precipitation Variables

| Variable | Source | Unit | Description |
|----------|--------|------|-------------|
| `prcp` | NOAA | inches | Daily precipitation amount |
| `snow` | NOAA | inches | Daily snowfall amount |
| `ms_precipitation` | Meteostat | inches | Daily precipitation from Meteostat |
| `ms_snow` | Meteostat | inches | Daily snowfall from Meteostat |
| `precipitation` | Visual Crossing | inches | Daily precipitation from Visual Crossing |
| `precip_probability` | Visual Crossing | % | Probability of precipitation |
| `precip_type` | Visual Crossing | category | Type of precipitation (rain, snow, etc.) |
| `nasa_precipitation` | NASA POWER | inches | Daily precipitation from NASA |

### Wind Variables

| Variable | Source | Unit | Description |
|----------|--------|------|-------------|
| `wind_speed` | Visual Crossing | mph | Average daily wind speed |
| `wind_direction` | Visual Crossing | degrees | Average daily wind direction (0-360°) |
| `ms_wind_speed` | Meteostat | mph | Average daily wind speed from Meteostat |
| `ms_wind_direction` | Meteostat | degrees | Average daily wind direction from Meteostat |
| `ms_wind_gust` | Meteostat | mph | Maximum daily wind gust from Meteostat |
| `nasa_wind_speed` | NASA POWER | mph | Average daily wind speed from NASA |

### Atmospheric Variables

| Variable | Source | Unit | Description |
|----------|--------|------|-------------|
| `pressure` | Visual Crossing | hPa | Sea level atmospheric pressure |
| `ms_pressure` | Meteostat | hPa | Sea level atmospheric pressure from Meteostat |
| `humidity` | Visual Crossing | % | Relative humidity |
| `nasa_humidity` | NASA POWER | % | Relative humidity from NASA |
| `cloud_cover` | Visual Crossing | % | Cloud cover percentage |
| `visibility` | Visual Crossing | miles | Visibility distance |
| `uv_index` | Visual Crossing | index (0-11+) | UV radiation index |
| `conditions` | Visual Crossing | category | General weather conditions description |

### Solar Radiation Variables

| Variable | Source | Unit | Description |
|----------|--------|------|-------------|
| `solar_radiation` | NASA POWER | kWh/m²/day | Daily total solar radiation |
| `longwave_radiation` | NASA POWER | W/m² | Downward longwave radiative flux |
| `ms_sunshine` | Meteostat | minutes | Daily sunshine duration |

### Urban Heat Island Variables

| Variable | Source | Unit | Description |
|----------|--------|------|-------------|
| `albedo` | Synthetic/Model | ratio (0-1) | Surface reflectivity (higher means more reflective) |
| `vegetation_index` | Synthetic/Model | ratio (0-1) | Normalized Difference Vegetation Index (NDVI) |
| `tree_canopy_percent` | Synthetic/Model | % | Percentage of area covered by tree canopy |
| `impervious_surface_percent` | Synthetic/Model | % | Percentage of impervious surfaces (roads, buildings) |
| `building_density` | Synthetic/Model | ratio (0-1) | Density of buildings in the area |
| `water_surface_percent` | Synthetic/Model | % | Percentage of water surfaces |
| `distance_to_water_km` | Calculated | km | Distance to nearest major water body |
| `heatmap_intensity` | Synthetic/Model | ratio (0-1) | Relative heat intensity from urban heat maps |

### Air Quality Variables

| Variable | Source | Unit | Description |
|----------|--------|------|-------------|
| `ozone` | EPA | ppm | Ground-level ozone concentration |
| `carbon_monoxide` | EPA | ppm | Carbon monoxide concentration |
| `nitrogen_dioxide` | EPA | ppb | Nitrogen dioxide concentration |
| `pm25` | EPA | μg/m³ | Fine particulate matter (diameter < 2.5 μm) |

### Calendar Variables

| Variable | Source | Unit | Description |
|----------|--------|------|-------------|
| `date` | Generated | date | Calendar date |
| `year` | Generated | year | Year |
| `month` | Generated | month (1-12) | Month of year |
| `day` | Generated | day (1-31) | Day of month |
| `dayofyear` | Generated | day (1-366) | Day of year |
| `dayofweek` | Generated | day (0-6) | Day of week (0=Monday, 6=Sunday) |
| `is_weekend` | Generated | binary (0/1) | Whether day is a weekend |
| `is_holiday` | Generated | binary (0/1) | Whether day is a holiday |
| `holiday_name` | Generated | category | Name of holiday if applicable |
| `week_of_year` | Generated | week (1-53) | Week of year |
| `season` | Generated | category | Season (winter, spring, summer, fall) |

## Data Sources

1. **NOAA GHCN-Daily** - Official temperature records from the National Oceanic and Atmospheric Administration's Global Historical Climatology Network Daily dataset. Data for Central Park station (GHCND:USW00094728) includes daily maximum, minimum temperatures, and precipitation.

2. **Visual Crossing Weather API** - Commercial weather data service that provides historical weather data including temperature, precipitation, humidity, wind, and other atmospheric conditions.

3. **NASA POWER** (Prediction of Worldwide Energy Resources) - NASA's dataset providing meteorological data primarily used for renewable energy, agriculture, and sustainability applications.

4. **Meteostat** - Open source weather data platform providing historical weather data compiled from various official sources worldwide.

5. **EPA Air Quality System (AQS)** - Environmental Protection Agency's repository of ambient air quality data, including measurements for criteria pollutants.

6. **Urban Heat Island Data** - Synthetic or modeled data capturing the factors that contribute to urban heat island effects.

## Target Variable

The primary target variable for prediction is `tmax` (maximum daily temperature in Central Park) from the NOAA dataset, measured in degrees Fahrenheit.

## Using This Data

When working with this dataset:

1. Check for missing values in each source
2. Consider the different time scales and measurement units
3. For temperature prediction, `tmax` from the NOAA dataset is considered the "ground truth"

## Data Preparation Notes

- Some datasets required unit conversion (e.g., Celsius to Fahrenheit, mm to inches)
- Temporal alignment was necessary to ensure all measurements correspond to the same time periods
- Urban heat island factors show seasonal variation that needs to be considered in modeling
- Date-based features were generated to capture seasonality and temporal patterns
- 