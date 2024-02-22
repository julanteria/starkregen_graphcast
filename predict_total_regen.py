"""# Clone graphcast and tree repositories
! git clone https://github.com/google-deepmind/graphcast.git
! git clone https://github.com/google-deepmind/tree.git

# Setup environment
! conda env create -n graphcast python=3.11
! conda activate graphcast
! pip install -r graphcast/requirements.txt

# Build dm-tree from source
! python dm-tree/setup.py build
! python dm-tree/setup.py install
! rm -rf dm-tree

# Data setup
! mkdir -p data/params data/stats data/datasets
! pip install gsutil
! gsutil -m cp "gs://dm_graphcast/params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz" .
! gsutil -m cp "gs://dm_graphcast/dataset/source-era5_date-2022-01-01_res-0.25_levels-37_steps-01.nc" data/datasets/
! gsutil -m cp "gs://dm_graphcast/stats/diffs_stddev_by_level.nc" "gs://dm_graphcast/stats/mean_by_level.nc" "gs://dm_graphcast/stats/stddev_by_level.nc" data/stats
! mv "GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz" "data/params/GraphCast_ERA5_1979-2017_Resolution-0.25_PressureLevels-37_Mesh-2to6_PrecipitationInputOutput.npz"
"""

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

# Set JAX to use CPU only
os.environ['JAX_PLATFORM_NAME'] = 'cuda'

sys.path.append('graphcast')

# JAX configuration
print("JAX is using: ", jax.devices())


src_diffs_stddev_by_level = "data/stats/diffs_stddev_by_level.nc"
src_mean_by_level = "data/stats/mean_by_level.nc"
src_stddev_by_level = "data/stats/stddev_by_level.nc"

with open(src_diffs_stddev_by_level, "rb") as f:
    diffs_stddev_by_level = xarray.load_dataset(f).compute()
with open(src_mean_by_level, "rb") as f:
    mean_by_level = xarray.load_dataset(f).compute()
with open(src_stddev_by_level, "rb") as f:
    stddev_by_level = xarray.load_dataset(f).compute()

src = "/home/jul/Documents/git/starkregen_graphcast/data/params/params_GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz"
with open(src, "rb",) as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)

params = ckpt.params
state = {}

model_config = ckpt.model_config
task_config = ckpt.task_config
print("Model description:/n", ckpt.description, "/n")
print("Model license:/n", ckpt.license, "/n")

example_batch_src = "/home/jul/Documents/git/starkregen_graphcast/data/datasets/dataset_source-era5_date-2022-01-01_res-1.0_levels-13_steps-01.nc"
with open(example_batch_src, "rb") as f:
    example_batch = xarray.load_dataset(f).compute()



eval_steps = 1
eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("6h", f"{eval_steps*6}h"),
    **dataclasses.asdict(task_config))




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



print("Inputs:  ", eval_inputs.dims.mapping)
print("Targets: ", eval_targets.dims.mapping)
#print("Forcings:", eval_forcings.dims.mapping)


# Example specific coordinates
specific_lon = 12.0
specific_lat = 51.5

# Selecting the data for specific lon and lat in eval_inputs
eval_inputs_specific = eval_inputs.sel(lon=specific_lon, lat=specific_lat, method='nearest')

# Selecting the data for specific lon and lat in eval_targets
eval_targets_specific = eval_targets.sel(lon=specific_lon, lat=specific_lat, method='nearest')

# Selecting the data for specific lon and lat in eval_forcings
eval_forcings_specific = eval_forcings.sel(lon=specific_lon, lat=specific_lat, method='nearest')


predictions = rollout.chunked_prediction(
    run_forward_jitted,
    rng=jax.random.PRNGKey(0),
    inputs=eval_inputs,
    targets_template=eval_targets * np.nan,
    forcings=eval_forcings)
print(predictions)


start = time.time()
start = str(str(start)[:10])

# For NetCDF format
predictions.to_netcdf(f'{start}_predicted_dataset_medium_model.nc')
