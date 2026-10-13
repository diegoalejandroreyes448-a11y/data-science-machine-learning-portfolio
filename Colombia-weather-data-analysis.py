
# ============================================================
# PROJECT: COLOMBIA WEATHER DATA ANALYSIS
# Description: Collect, clean, validate, and export weather data
# Data source: Open-Meteo API
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import requests
import pandas as pd

print("Libraries loaded successfully.")


# ============================================================
# 2. DEFINE CITIES TO ANALYZE
# ============================================================

cities = {
    "Bogota": {"lat": 4.7110, "lon": -74.0721},
    "Medellin": {"lat": 6.2442, "lon": -75.5812},
    "Cali": {"lat": 3.4516, "lon": -76.5320},
    "Barranquilla": {"lat": 10.9639, "lon": -74.7964},
    "Cartagena": {"lat": 10.3910, "lon": -75.4794},
}

print(f"Weather forecasts will be collected for {len(cities)} cities.")


# ============================================================
# 3. CREATE WEATHER API FUNCTION
# ============================================================

def get_weather_forecast(city_name, latitude, longitude):
    """
    Retrieve a 7-day weather forecast for a city
    using the Open-Meteo API.
    """

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": (
            "temperature_2m_max,"
            "temperature_2m_min,"
            "precipitation_sum,"
            "windspeed_10m_max,"
            "weathercode"
        ),
        "timezone": "auto",
        "forecast_days": 7,
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    return response.json()


# ============================================================
# 4. COLLECT WEATHER DATA FROM ALL CITIES
# ============================================================

rows = []

for city_name, coordinates in cities.items():

    try:
        data = get_weather_forecast(
            city_name,
            coordinates["lat"],
            coordinates["lon"]
        )

        daily_data = data["daily"]

        for i, date in enumerate(daily_data["time"]):
            rows.append({
                "city": city_name,
                "date": date,
                "max_temperature": daily_data["temperature_2m_max"][i],
                "min_temperature": daily_data["temperature_2m_min"][i],
                "precipitation_mm": daily_data["precipitation_sum"][i],
                "max_wind_speed_kmh": daily_data["windspeed_10m_max"][i],
                "weather_code": daily_data["weathercode"][i],
            })

        print(f"Weather data collected for {city_name}.")

    except (requests.RequestException, KeyError, TypeError) as error:
        print(f"Could not retrieve data for {city_name}: {error}")


# Convert collected records into a DataFrame
df_raw = pd.DataFrame(rows)

if df_raw.empty:
    raise RuntimeError(
        "No weather data was collected. Check your internet connection "
        "and the Open-Meteo API."
    )

print("\nRaw dataset preview:")
print(df_raw.head(10))


# ============================================================
# 5. TRANSLATE WEATHER CODES
# ============================================================

weather_codes = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


# ============================================================
# 6. CLEAN AND TRANSFORM THE DATA
# ============================================================

df = df_raw.copy()

# Convert dates into datetime objects
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Translate weather codes into readable descriptions
df["weather_condition"] = (
    df["weather_code"]
    .map(weather_codes)
    .fillna("Unknown")
)

# Derive the daily temperature range
df["temperature_range"] = (
    df["max_temperature"] - df["min_temperature"]
).round(1)

# Round numeric values for presentation
numeric_columns = [
    "max_temperature",
    "min_temperature",
    "precipitation_mm",
    "max_wind_speed_kmh",
]

df[numeric_columns] = df[numeric_columns].round(1)

# Remove records without essential values
df = df.dropna(
    subset=[
        "city",
        "date",
        "max_temperature",
        "min_temperature",
    ]
)

# Add the weekday name in English
df["weekday"] = df["date"].dt.day_name()

# Select and order final dataset columns
final_columns = [
    "city",
    "date",
    "weekday",
    "weather_condition",
    "max_temperature",
    "min_temperature",
    "temperature_range",
    "precipitation_mm",
    "max_wind_speed_kmh",
]

df_final = df[final_columns].copy()

print("\nCleaned dataset preview:")
print(df_final.head(10))


# ============================================================
# 7. VALIDATE THE DATASET
# ============================================================

print("\nDataset dimensions:")
print(df_final.shape)

print("\nData types:")
print(df_final.dtypes)

print("\nMissing values by column:")
print(df_final.isnull().sum())

print("\nStatistical summary:")
print(df_final.describe(include="all"))


# ============================================================
# 8. SUMMARIZE WEATHER BY CITY
# ============================================================

city_summary = (
    df_final.groupby("city")
    .agg(
        average_max_temperature=("max_temperature", "mean"),
        average_min_temperature=("min_temperature", "mean"),
        total_precipitation_mm=("precipitation_mm", "sum"),
        average_wind_speed_kmh=("max_wind_speed_kmh", "mean"),
    )
    .round(2)
    .sort_values("average_max_temperature", ascending=False)
)

print("\nWeather summary by city:")
print(city_summary)


# ============================================================
# 9. IDENTIFY THE WETTEST FORECAST DAYS
# ============================================================

wettest_days = (
    df_final.sort_values("precipitation_mm", ascending=False)
    .head(10)
)

print("\nTop 10 days with the highest forecast precipitation:")
print(wettest_days[
    ["city", "date", "precipitation_mm", "weather_condition"]
])


# ============================================================
# 10. EXPORT DATASETS TO CSV
# ============================================================

df_final.to_csv(
    "colombia_weather_data.csv",
    index=False,
    encoding="utf-8-sig"
)

city_summary.to_csv(
    "colombia_weather_city_summary.csv",
    encoding="utf-8-sig"
)

print("\nFiles exported successfully:")
print("- colombia_weather_data.csv")
print("- colombia_weather_city_summary.csv")


# ============================================================
# END OF PROJECT
# ============================================================
