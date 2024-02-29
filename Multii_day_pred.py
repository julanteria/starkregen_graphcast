# %%
#! pip install h5netcdf

# %%
import dataclasses
import os
import xarray
import matplotlib.pyplot as plt
import sys
sys.path.append('graphcast')
from graphcast import graphcast, checkpoint, normalization, autoregressive, casting, data_utils, rollout
import jax
#jax.config.update('jax_platform_name', 'gpu')
print("JAX is using: ", jax.devices())
import haiku as hk
import numpy as np
import functools
import time
import pandas as pd


# %%

src_diffs_stddev_by_level = "data/stats/diffs_stddev_by_level.nc"
src_mean_by_level = "data/stats/mean_by_level.nc"
src_stddev_by_level = "data/stats/stddev_by_level.nc"

with open(src_diffs_stddev_by_level, "rb") as f:
    diffs_stddev_by_level = xarray.load_dataset(f).compute()
with open(src_mean_by_level, "rb") as f:
    mean_by_level = xarray.load_dataset(f).compute()
with open(src_stddev_by_level, "rb") as f:
    stddev_by_level = xarray.load_dataset(f).compute()

src = "data/params/params_GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
with open(src, "rb",) as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)




params = ckpt.params
state = {}

model_config = ckpt.model_config
task_config = ckpt.task_config
task_config = dataclasses.replace(task_config, input_duration="18h")
print("Model description:/n", ckpt.description, "/n")
print("Model license:/n", ckpt.license, "/n")

#example_batch_src = "/Users/jules/Documents/github/starkregen_graphcast/data/datasets/source-era5_date-2022-01-01_res-0.25_levels-37_steps-01.nc"
my_own_src = "data/copernicus_data/input_pressure_and_no_pressure_combined_20_02.nc"
#my_own_src = "data/copernicus_data/input_pressure_and_no_pressure_combined_13_06.nc"
with open(my_own_src, "rb") as f:
    example_batch = xarray.load_dataset(f).compute()


# %%
print(ckpt.task_config)
# change input duration to 18h
task_config = dataclasses.replace(task_config, input_duration="18h")
print(task_config)

# %%
# double the time dimension
#doubled_example_batch = xarray.concat([example_batch, example_batch], dim='time')


# %%
# For the eval_inputs step=1
eval_steps = 1
eval_inputs, _, __ = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("6h", f"{eval_steps*6}h"), **dataclasses.asdict(task_config))



# For eval_targets and eval_forcings step=x
eval_steps = 10
___, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("6h", f"{eval_steps*6}h"), **dataclasses.asdict(task_config))
#slice 1d in 6h steps
#eval_targets = eval_targets.sel(lead_time=slice("0h", f"{eval_steps*6}h"))



# %%
# sel example_batch to one time step


# %%
eval_targets

# %%
# Create eval_targets


for i in range(3):
    eval_targets = xarray.concat([eval_targets, eval_targets], dim='time')

print(eval_targets)

for i in range(3):
    eval_forcings = xarray.concat([eval_forcings, eval_forcings], dim='time')

print((eval_forcings.time.size))





# %%
# Assuming `eval_forcings` is your xarray Dataset
# Generate new time values as before
# 10 tage in 6h steps
hours = np.arange(6,25*6, 6)
print(len(hours))


nanoseconds = hours * 1e9 * 3600
time_deltas = np.array(nanoseconds, dtype='timedelta64[ns]')

# Check if the size matches; if not, you may need to adjust your approach
print(eval_forcings.time.size == time_deltas.size)

# If the sizes match, you can directly replace the time coordinate
if eval_forcings.time.size == time_deltas.size:
    eval_forcings['time'] = time_deltas
    print("Time coordinates updated successfully.")
else:
    print("Size mismatch. Time coordinates not updated.")

# %%
# Assuming `eval_forcings` is your xarray Dataset
# Generate new time values as before
hours = np.arange(6, 25*6, 6)
nanoseconds = hours * 1e9 * 3600
time_deltas = np.array(nanoseconds, dtype='timedelta64[ns]')

# Check if the size matches; if not, you may need to adjust your approach
print(eval_targets.time.size == time_deltas.size)



# If the sizes match, you can directly replace the time coordinate
if eval_targets.time.size == time_deltas.size:
    eval_targets['time'] = time_deltas
    print("Time coordinates updated successfully.")
else:
    print("Size mismatch. Time coordinates not updated.")

# %%
"""eval_inputs.time.values
# convert timedelta int to string
print(np.array(eval_targets.time.values) / 1e9 / (3600))
print(eval_targets.time.values)
#print([6, 12, 18, 24, 30, 36]*1e9*3600)
x = [i*1e9*3600 for i in range(6, 36, 6)]
#print(np.array(x) / 1e9 / (3600))
x = np.array(np.array(x), dtype='timedelta64[s]')
eval_forcings.assign_coords(time=x)
print(eval_forcings.time.values)
"""


# %%
def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig):
  """Constructs and wraps the GraphCast Predictor."""
  # Deeper one-step predictor.
  predictor = graphcast.GraphCast(model_config, task_config)

  # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
  # from/to float32 to/from BFloat16.
  predictor = casting.Bfloat16Cast(predictor)

  # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
  # BFloat16 happens after applying normalization to the inputs/targets.
  predictor = normalization.InputsAndResiduals(
      predictor,
      diffs_stddev_by_level=diffs_stddev_by_level,
      mean_by_level=mean_by_level,
      stddev_by_level=stddev_by_level)

  # Wraps everything so the one-step model can produce trajectories.
  predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
  return predictor



def run_forward(model_config, task_config, inputs, targets_template, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  return predictor(inputs, targets_template=targets_template, forcings=forcings)


def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
  def _aux(params, state, i, t, f):
    (loss, diagnostics), next_state = loss_fn.apply(
        params, state, jax.random.PRNGKey(0), model_config, task_config,
        i, t, f)
    return loss, (diagnostics, next_state)
  (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
      _aux, has_aux=True)(params, state, inputs, targets, forcings)
  return loss, diagnostics, next_state, grads

# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn):
  return functools.partial(
      fn, model_config=model_config, task_config=task_config)

# Always pass params and state, so the usage below are simpler
def with_params(fn):
  return functools.partial(fn, params=params, state=state)

# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
  return lambda **kw: fn(**kw)[0]

# Transform the function with Haiku
run_forward = hk.transform_with_state(run_forward)

init_jitted = jax.jit(with_configs(run_forward.init))

#grads_fn_jitted = with_params(jax.jit(with_configs(grads_fn)))
run_forward_jitted = drop_state(with_params(jax.jit(with_configs(
    run_forward.apply))))



# %%

start = time.time()
predictions = rollout.chunked_prediction(
    run_forward_jitted,
    rng=jax.random.PRNGKey(0),
    inputs=eval_inputs,
    targets_template=eval_targets * np.nan,
    forcings=eval_forcings,
    num_steps_per_chunk=1,
    verbose=True)

end = time.time()

# %%
print("Predction finished in: ", end-start, " s   or  ", (end-start)/60, " min")
print("Predictions: ")
print(predictions)


# %%
# For NetCDF format
predictions.to_netcdf(f'possible_10step_steps_predictions_13_06.nc')

