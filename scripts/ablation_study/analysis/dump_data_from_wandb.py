"""Script for extracting run data."""

import pandas as pd
from tqdm import tqdm

import wandb

api = wandb.Api(timeout=60)

# Project is specified by <entity/project-name>
runs = api.runs("media-bias-group/experiment_0_sentiment")

row_list = []
lr_dict = {}
max_epoch_dict = {}
patience_dict = {}

for run in tqdm(runs):
    run_dict = {}
    sweep = run.sweep.name
    run_dict.update({"sweep": sweep})
    run_dict.update(run.summary._json_dict)

    run_dict.update(run.config)  # hyperparameters

    row_list.append(run_dict)

df = pd.DataFrame(row_list)

df.to_csv("logging/experiment0/all_data.csv", index=False)
