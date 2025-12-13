import cdsapi

dataset = "derived-era5-single-levels-daily-statistics"
request = {
    "product_type": "reanalysis",
    "variable": [
        "total_precipitation",
        "clear_sky_direct_solar_radiation_at_surface",
        "convective_precipitation",
        "convective_rain_rate",
        "instantaneous_large_scale_surface_precipitation_fraction",
        "large_scale_rain_rate",
        "large_scale_precipitation",
        "large_scale_precipitation_fraction",
        "maximum_total_precipitation_rate_since_previous_post_processing",
        "minimum_total_precipitation_rate_since_previous_post_processing",
        "precipitation_type",
        "total_column_rain_water"
    ],
    "year": "2024",
    "month": ["01"],
    "day": ["01"],
    "daily_statistic": "daily_mean",
    "time_zone": "utc+03:00",
    "frequency": "1_hourly",
    "format": "grib"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download("era5_daily_2025_10.nc")
