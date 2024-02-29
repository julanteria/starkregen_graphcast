from data_utils import load_and_compute_dataset, combine_datasets, adjust_datetime, save_dataset
from model_utils import GraphCastModel

# Load and compute input data
src_input_data_pressure = "data/copernicus_data/20_02_pressure_level_data.nc"
src_input_data_no_pressure = "data/copernicus_data/20_02_no_pressure_level_data.nc"
src_example_batch = "data/datasets/source-era5_date-2022-01-01_res-0.25_levels-37_steps-01.nc"

ds_with_pressure = load_and_compute_dataset(src_input_data_pressure)
ds_without_pressure = load_and_compute_dataset(src_input_data_no_pressure)
example_batch = load_and_compute_dataset(src_example_batch)

# Combine datasets
combined_input_batch = combine_datasets(ds_with_pressure, ds_without_pressure, example_batch)
# fix dates
combined_input_batch= adjust_datetime(ds_with_pressure, ds_without_pressure, combined_input_batch)

print(combined_input_batch)

# Save combined dataset
combined_input_batch_path = "data/copernicus_data/TEST_input_pressure_and_no_pressure_combined_20_02.nc"
save_dataset(combined_input_batch, combined_input_batch_path)


# Load the model
src_diffs_stddev_by_level = "data/stats/diffs_stddev_by_level.nc"
src_mean_by_level = "data/stats/mean_by_level.nc"
src_stddev_by_level = "data/stats/stddev_by_level.nc"
src_model = "data/params/params_GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"

model = GraphCastModel(src_model, src_diffs_stddev_by_level, src_mean_by_level, src_stddev_by_level, combined_input_batch, number_of_predictions=2)

print("Model Evaluation Inputs")
print(model.eval_inputs)
print()
print()
print("Model Evaluation Targets")
print(model.eval_targets)
print()
print()
print("Model Evaluation Forcings")
print(model.eval_forcings)
print()
print()

prediciton = model.make_predictions()
print("Prediction")
print(prediciton)

# Save the predictions
prediciton_path = "data/predictions/TEST_predictions_20_02.nc"
save_dataset(prediciton, prediciton_path)