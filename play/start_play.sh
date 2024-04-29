#!/bin/bash

# Cloning only the play folder
git clone --depth 1 --filter=blob:none https://github.com/wmcgloin/air_hockey_bot.git --sparse
cd air_hockey_bot
git sparse-checkout init --cone
git sparse-checkout set play

# Initialize Conda for your shell
eval "$(conda shell.bash hook)"


conda create -y -n airhockey python=3.9
conda activate airhockey
pip install -r requirements.txt

python play.py