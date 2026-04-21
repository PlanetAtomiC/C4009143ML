import pandas as pd
import numpy as np
from Config import (dataset_path, Sheff_code, Lat_min, Lat_max, Lon_min, Lon_max, Numeric_object_cols, log)

def load_sheffield_data():

    try:
        dataframe = pd.read_csv(dataset_path)

    except FileNotFoundError:
        log.error("Dataset not found at: %s", dataset_path)
        raise

    sheffield = dataframe[dataframe['local_authority_district'] == Sheff_code].copy()
    log.info("Loaded %d Sheffield rows from dataset.", len(sheffield))
    return sheffield


def remove_invalid_coordinates(dataframe):
    dataframe = dataframe.dropna(subset=['latitude', 'longitude'])
    dataframe = dataframe[(dataframe['latitude'].between(Lat_min, Lat_max)) & (dataframe['longitude'].between(Lon_min, Lon_max))]
    log.info("After coordinate filtering: %d rows remain.", len(dataframe))
    return dataframe

def impute_missing_values(dataframe):
    #Replace DfT values with proper NaN before any imputation
    dataframe = dataframe.replace(-1, pd.NA)
    dataframe = dataframe.drop_duplicates()

    #Forward fill severity adjustments
    dataframe['collision_adjusted_severity_serious'] = dataframe['collision_adjusted_severity_serious'].ffill()
    dataframe['collision_adjusted_severity_slight']  = dataframe['collision_adjusted_severity_slight'].ffill()

    #Mean imputation for standard numeric columns
    dataframe = dataframe.fillna(dataframe.mean(numeric_only=True))

    for col in Numeric_object_cols:
        dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
    dataframe[Numeric_object_cols] = dataframe[Numeric_object_cols].fillna(dataframe[Numeric_object_cols].mean())

    for col in dataframe.select_dtypes(include='str').columns:
        dataframe[col] = dataframe[col].fillna(dataframe[col].mode()[0])

    nulls_remaining = dataframe.isnull().sum().sum()
    log.info("Imputation complete. Total nulls remaining: %d", nulls_remaining)
    return dataframe

def fix_data_types(dataframe):
    dataframe['date']  = pd.to_datetime(dataframe['date'], format='%d/%m/%Y')
    dataframe['time']  = pd.to_datetime(dataframe['time'], format='%H:%M').dt.time
    dataframe['month'] = dataframe['date'].dt.month
    dataframe['hour']  = pd.to_datetime(dataframe['time'].astype(str), format='%H:%M:%S').dt.hour
    return dataframe

def engineer_features(dataframe):
    dataframe['is_rush_hour'] = dataframe['hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 18) else 0)
    dataframe['is_weekend'] = dataframe['day_of_week'].apply(lambda x: 1 if x in [1, 7] else 0)
    dataframe['is_dark'] = dataframe['light_conditions'].apply(lambda x: 1 if x >= 4 else 0)
    dataframe['is_bad_weather'] = dataframe['weather_conditions'].apply(lambda x: 1 if x in [2, 3, 5, 6, 7] else 0)
    dataframe['severity_label'] = dataframe['collision_severity'].map({1: 'Fatal', 2: 'Serious', 3: 'Slight'})
    log.info("Feature engineering complete.")
    return dataframe


def run_preprocessing():
    dataframe = load_sheffield_data()
    dataframe = remove_invalid_coordinates(dataframe)
    dataframe = impute_missing_values(dataframe)
    dataframe = fix_data_types(dataframe)
    dataframe = engineer_features(dataframe)

    log.info("Preprocessing complete. Final shape: %s", dataframe.shape)
    return dataframe