#!/bin/bash

# Clone the graphcast repository
git clone https://github.com/google-deepmind/graphcast.git

# Initialize Conda in the script
# Get the current operating system
OS=$(uname)

# Define the universal user path
USER_HOME_PATH=$(eval echo ~$USER)

# Check if OS is macOS or Linux and source the appropriate conda.sh script
if [[ "$OS" == "Darwin" ]]; then
    # macOS
    source "/opt/anaconda3/etc/profile.d/conda.sh"
elif [[ "$OS" == "Linux" ]]; then
    # Linux
    source "$USER_HOME_PATH/anaconda3/etc/profile.d/conda.sh"
else
    echo "Unsupported operating system."
fi



# Create and activate a new conda environment
yes | conda create -n graphcast python=3.11
conda activate graphcast

# Install Python requirements
# Replace 'install' with the appropriate command for your setup script
python graphcast/setup.py install
yes | conda install ipykernel

# GPU
#pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html




# Check if OS is macOS or Linux and source the appropriate conda.sh script
if [[ "$OS" == "Darwin" ]]; then
    # macOS
    # Clone dm-tree, build and install it
    git clone https://github.com/google-deepmind/tree.git
    cd tree
    python setup.py build
    python setup.py install
    cd ..
    rm -rf tree
    rm -rf dm_tree.egg-info
elif [[ "$OS" == "Linux" ]]; then
    # Linux
    pip install dm-tree
else
    echo "Unsupported operating system."
fi

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
