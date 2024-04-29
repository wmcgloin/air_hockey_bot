#!/bin/bash

# Set the directory and environment names
DIR="air_hockey_bot"
ENV_NAME="airhockey"

# Check if the directory already exists
if [ -d "$DIR" ]; then
    echo "The directory $DIR already exists. Please remove it or use a different directory."
    exit 1
fi

# Clone the specific part of the repository
git clone --depth 1 --filter=blob:none https://github.com/wmcgloin/air_hockey_bot.git --sparse
cd $DIR
git sparse-checkout init --cone
git sparse-checkout set play

# Initialize Conda for your shell
eval "$(conda shell.bash hook)"

# Check if the Conda environment already exists
if conda info --envs | grep -qw $ENV_NAME; then
    echo "A Conda environment named '$ENV_NAME' already exists. Please remove it or use a different name."
    exit 1
fi

# Create and activate the Conda environment
conda create -y -n $ENV_NAME python=3.9
conda activate $ENV_NAME

# Install Python dependencies
# pip install -r play/requirements.txt
pip install gymnasium==0.28.1 numpy==1.24.4 pygame==2.1.0 torch==1.12.1

# Run the application
cd play
python play.py
