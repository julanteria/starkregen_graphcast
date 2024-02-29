import xarray as xr
import numpy as np
import h5netcdf
import cdsapi
import os

def load_and_compute_dataset(path: str) -> xr.Dataset:
    """
    Load and compute a dataset from a NetCDF file.

    Parameters:
    - path (str): The file path to the dataset.

    Returns:
    - xr.Dataset: The loaded and computed xarray dataset.
    """
    with open(path, "rb") as f:
        dataset = xr.load_dataset(f).compute()
    return dataset

def combine_datasets(ds_with_pressure: xr.Dataset, ds_without_pressure: xr.Dataset, example_batch: xr.Dataset) -> xr.Dataset:
    """
    Combine datasets with and without pressure level data, adjusting variables to match an example batch.

    Parameters:
    - ds_with_pressure (xr.Dataset): Dataset containing pressure level data.
    - ds_without_pressure (xr.Dataset): Dataset not containing pressure level data.
    - example_batch (xr.Dataset): The example batch dataset to match variables to.

    Returns:
    - xr.Dataset: The example batch dataset with combined data.
    """
    # Rename variables in the pressure level dataset to match those in the example batch if needed
    rename_dict_pressure = {
        'u_wind': 'u_component_of_wind',
        'v_wind': 'v_component_of_wind',
        'vertical_velocity': 'vertical_velocity',  
        'specific_humidity': 'specific_humidity', 
    }
    ds_with_pressure = ds_with_pressure.rename(rename_dict_pressure)

    # Add batch dimension to pressure level data and assign to example batch
    for var in ds_with_pressure.data_vars:
        if var in example_batch:
            data_with_batch = ds_with_pressure[var].expand_dims('batch', axis=0)
            example_batch[var].values = data_with_batch.values

    # Process non-pressure level data
    rename_dict_no_pressure = {
        'z': 'geopotential_at_surface',
        'lsm': 'land_sea_mask',
        't2m': '2m_temperature',
        'msl': 'mean_sea_level_pressure',
        'v10': '10m_v_component_of_wind',
        'u10': '10m_u_component_of_wind',
        'tp': 'total_precipitation_6hr',
        'tisr': 'toa_incident_solar_radiation',
    }
    ds_without_pressure = ds_without_pressure.rename(rename_dict_no_pressure)

    # For variables without a batch dimension, directly assign values to the example batch
    for var in ds_without_pressure.data_vars:
        if var in example_batch:
            if var in ['geopotential_at_surface', 'land_sea_mask']:
                data = ds_without_pressure[var].isel(time=0).drop('time')
            else:
                data = ds_without_pressure[var].expand_dims('batch', axis=0) if 'batch' not in ds_without_pressure[var].dims else ds_without_pressure[var]
            example_batch[var].values = data.values

    return example_batch

def adjust_datetime(ds_with_pressure: xr.Dataset, ds_without_pressure: xr.Dataset, example_batch: xr.Dataset) -> xr.Dataset:
    """
    Adjust the datetime dimension of the example batch to match the datasets.

    Parameters:
    - ds_with_pressure (xr.Dataset): Dataset with pressure data, for reference datetime.
    - ds_without_pressure (xr.Dataset): Dataset without pressure data, for reference datetime.
    - example_batch (xr.Dataset): The example batch dataset to adjust datetime on.

    Returns:
    - xr.Dataset: The example batch with adjusted datetime.
    """
    day_date1 = ds_with_pressure.time.values[0].astype('datetime64[D]')
    day_date2 = ds_without_pressure.time.values[0].astype('datetime64[D]')
    
    assert day_date1 == day_date2, "Dates are different"
    
    original_datetimes = example_batch.datetime.values
    changed_datetimes = np.array([day_date1 + (dt - dt.astype('datetime64[D]')) for dt in original_datetimes])
    example_batch.datetime.values = changed_datetimes
    
    return example_batch

def save_dataset(dataset: xr.Dataset, path: str) -> None:
    """
    Save a dataset to a NetCDF file using the h5netcdf engine.

    Parameters:
    - dataset (xr.Dataset): The dataset to save.
    - path (str): The file path where the dataset will be saved.
    """
    dataset.to_netcdf(path, engine='h5netcdf')



def get_data_from_copernicus(data_type: str, times: list, days: list, months: list, years: list, 
                             save_path: str = "", pressure_levels: int = 37, product_type: str = "reanalysis", format: str = "netcdf") -> None:
    """
    Uses the cdsapi to download data from the Copernicus ERA5 dataset.
    
    Parameters:
    - data_type: The type of data to retrieve (e.g., 'single-levels', 'pressure-levels').
    - times: A list of times in 'HH:MM' format.
    - days: A list of days as strings.
    - months: A list of months as strings.
    - years: A list of years as strings.
    - pressure_levels: The number of pressure levels to retrieve data for (only relevant for 'pressure-levels').
    - product_type: The type of product to retrieve.
    - format: The format of the retrieved data.
    """

    c = cdsapi.Client()

    esri_vars_for_prediction_without_pressure = [
    'land_sea_mask',
    '2m_temperature',
    'mean_sea_level_pressure',
    'total_precipitation',
    'toa_incident_solar_radiation',
    'geopotential',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    ]
    
    esri_vars_for_prediction_with_pressure = [
    'temperature',
    'specific_humidity',
    'u_component_of_wind',
    'v_component_of_wind',
    'vertical_velocity',
    'geopotential',
    ] 

    if data_type == 'pressure-levels':
        variables = esri_vars_for_prediction_with_pressure
        if pressure_levels == 37:
            pressure_levels = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
        elif pressure_levels == 13:
            pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        else:
            raise ValueError("Invalid number of pressure levels. Please choose 13 or 37.")

        request_parameters = {
            "pressure_level": pressure_levels,
            "variable": variables,
        }
    elif data_type == 'single-levels':
        variables = esri_vars_for_prediction_without_pressure
        request_parameters = {
            "variable": variables,
        }
    else:
        raise ValueError("Invalid data type. Please choose 'single-levels' or 'pressure-levels'.")

    # Common parameters for both types
    request_parameters.update({
        "product_type": product_type,
        "format": format,
        "year": years,
        "month": months,
        "day": days,
        "time": times,
    })

    file_name = f'reanalysis-era5-{data_type}_data_for_prediction.nc'
    save_path = os.path.join(save_path, file_name)
    c.retrieve(
        f'reanalysis-era5-{data_type}',
        request_parameters,
        file_name
    )
    print(f"Data retrieved and saved as {save_path}")