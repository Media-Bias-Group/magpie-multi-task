import pandas as pd 
import wandb
import dataframe_image as dfi
from tqdm import tqdm
import json

api = wandb.Api(timeout=60)

METRIC_CLS = 'eval_f1'
METRIC_REG = 'eval_MSE'
LOSS = 'eval_loss'

# Project is specified by <entity/project-name>
runs = api.runs("media-bias-group/hyperparam-tuning")

row_list = []
lr_dict = {}
max_epoch_dict = {}
patience_dict = {}

for run in tqdm(runs):
    run_dict = {}
    st_id = run.sweep.name

    run_dict.update({'subtask':st_id})
    if st_id + '_eval_f1' in run.summary._json_dict.keys():
        run_dict.update({'metric':run.summary._json_dict[st_id + '_eval_f1']})
    elif st_id + '_eval_R2' in run.summary._json_dict.keys():
        run_dict.update({'metric':run.summary._json_dict[st_id + '_eval_R2']})

    run_dict.update(run.config) #hyperparameters

    row_list.append(run_dict)

df = pd.DataFrame(row_list)

# best hyperparameters w respect to F1 score ( / MSE)
df_optimal = df.loc[df.reset_index().groupby(['subtask'])['metric'].idxmax()]
df_optimal.to_csv('logging/hyperparameters.csv',index=False)
dfi.export(df_optimal,'logging/image.png')

# create final parameters
for i,row in df_optimal.iterrows():
    lr_dict.update({row['subtask'] : row['lr']})
    max_epoch_dict.update({row['subtask']: row['max_epoch']})
    patience_dict.update({row['subtask'] : row['patience']})

# save final parameters to our config file
with open('config.py','a') as f:
    f.write('\nhead_specific_lr = ')
    f.write(json.dumps(lr_dict))
    f.write('\nhead_specific_patience = ')
    f.write(json.dumps(patience_dict))
    f.write('\nhead_specific_max_epoch = ')
    f.write(json.dumps(max_epoch_dict))
