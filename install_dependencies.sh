#!/bin/bash

# Clone the graphcast repository
git clone https://github.com/google-deepmind/graphcast.git

# Initialize Conda in the script
# Replace this with the path to your Conda installation if it's different
source /opt/anaconda3/etc/profile.d/conda.sh


# Create and activate a new conda environment
yes | conda create -n graphcast python=3.11
conda activate graphcast

# Install Python requirements
# Replace 'install' with the appropriate command for your setup script
python graphcast/setup.py install
conda install ipykernel

# multiline comment

# Clone dm-tree, build and install it
git clone https://github.com/google-deepmind/tree.git
cd tree
python setup.py build
python setup.py install
cd ..
#rm -rf tree
#rm -rf dm_tree.egg-info

# Create data directories
mkdir -p data/params
mkdir -p data/stats
mkdir -p data/datasets

# Install gsutil
pip install gsutil

# Download data using gsutil
gsutil -m cp "gs://dm_graphcast/params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz" .
gsutil -m cp "gs://dm_graphcast/dataset/source-era5_date-2022-01-01_res-0.25_levels-37_steps-01.nc" data/datasets/
gsutil -m cp "gs://dm_graphcast/stats/diffs_stddev_by_level.nc" "gs://dm_graphcast/stats/mean_by_level.nc" "gs://dm_graphcast/stats/stddev_by_level.nc" data/stats

# Move downloaded parameter file to the appropriate directory
mv "GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz" "data/params/GraphCast_ERA5_1979-2017_Resolution-0.25_PressureLevels-37_Mesh-2to6_PrecipitationInputOutput.npz"
# multiline comment end
