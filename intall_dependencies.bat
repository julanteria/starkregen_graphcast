@echo off

:: Clone the graphcast repository
git clone https://github.com/google-deepmind/graphcast.git

:: Initialize Conda in the script
:: For Windows, assuming Anaconda is installed and added to PATH

:: Create and activate a new conda environment
call conda create -n graphcastwin python=3.11 -y
call conda activate graphcastwin

:: Install Python requirements
python graphcast/setup.py install
call conda install ipykernel -y

:: GPU
:: Uncomment the next line if GPU support is needed and CUDA is installed
:: pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

:: Install dm-tree (No OS-specific steps needed for Windows)
pip install dm-tree

:: Create data directories
mkdir data\params
mkdir data\stats
mkdir data\datasets

:: Install gsutil
pip install gsutil

:: Download data using gsutil
gsutil -m cp "gs://dm_graphcast/params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz" .
gsutil -m cp "gs://dm_graphcast/dataset/source-era5_date-2022-01-01_res-0.25_levels-37_steps-01.nc" data\datasets\
gsutil -m cp "gs://dm_graphcast/stats/diffs_stddev_by_level.nc" "gs://dm_graphcast/stats/mean_by_level.nc" "gs://dm_graphcast/stats/stddev_by_level.nc" data\stats\

:: Move downloaded parameter file to the appropriate directory
move "GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz" "data\params\GraphCast_ERA5_1979-2017_Resolution-0.25_PressureLevels-37_Mesh-2to6_PrecipitationInputOutput.npz"

:: End of script
