#!/bin/bash

git clone --depth 1 --filter=blob:none https://github.com/wmcgloin/air_hockey_bot.git --sparse
cd air_hockey_bot
git sparse-checkout init --cone
git sparse-checkout set play

conda create -y -n airhockey python=3.9
conda activate airhockey
pip install -y -r requirements.txt

python play.py