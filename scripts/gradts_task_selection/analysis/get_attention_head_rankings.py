from storage import storage_client
import os
from scipy.stats import kendalltau
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


path = "model_files/gradts/"

matrices = os.listdir(path)
babe_mat = torch.load(path + "10001_att_mat.pt")
row_list = []
for m in matrices:
    mat = torch.load(path + m)
    corr,_ = kendalltau(babe_mat.reshape(12*12,-1),mat.reshape(12*12,-1))
    row_list.append({'subtask':m[:-11],'corr':corr})

df = pd.DataFrame(row_list).sort_values(by='corr',ascending=False)

# show top 5
fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
sns.heatmap(babe_mat, vmin=babe_mat.min(), vmax=babe_mat.max(),cmap="Blues",ax=axes[0],cbar=False)
axes[0].set_title('BABE')

for i in range(1,3):
    st = str( df.iloc[i]['subtask'])
    mat = torch.load(path + st + "_att_mat.pt")
    sns.heatmap(mat, vmin=mat.min(), vmax=mat.max(),cmap="Blues",ax=axes[i],cbar=False)
    axes[i].set_title(st)

df.to_csv('att.csv')