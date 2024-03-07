import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import torch
import os
import sys
import logging

import numpy as np

import psutil


def save_img(plt, name, path):
    path = os.path.join(path, name)
    
    plt.savefig(path, dpi=300)

def plot(image_name, 
        points, 
        vectors=[],
        targets =[],
        values = [],
        show_normals=False, 
        vector_length=1.0,
        save=False,
        show=True,
        name='',
        path = '',
        blank=False
        ):    
    
    # Set the figure size
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(image_name, fontsize=16, y=0.05)
    
    if not blank:
        gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])
    else:
        gs = GridSpec(1, 2, width_ratios=[1, 0.05])
    
    x, y, z = points[:, :3].T
    
    if len(values) == 0:
        values = np.random.rand(len(x))
    
    if len(targets) == 0:
        targets = np.random.rand(len(x))
    
    
    color_map = plt.get_cmap('coolwarm')
    norm = Normalize(vmin=np.min(targets), vmax=np.max(targets))
    
    colors = color_map(norm(targets))
    
    ax1 = plt.subplot(gs[0], projection='3d')
    ax1.scatter(x, y, z, s=4, linewidths=0, c=colors, cmap=color_map, edgecolors='k')
    ax1.set_title('Ground truth [kWh/m2]', pad=20, loc='center')
    ax1.set_xlim(-0.75, 0.75)
    ax1.set_ylim(-0.75, 0.75)
    ax1.set_zlim(-0.25, 1.25)
    ax1.grid(False)
    ax1.set_proj_type('ortho')
    
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.w_xaxis.line.set_visible(False)
    ax1.w_yaxis.line.set_visible(False)
    ax1.w_zaxis.line.set_visible(False)
    ax1.set_axis_off()
    
    colors = color_map(norm(values))
    
    if not blank:
        ax2 = plt.subplot(gs[1], projection='3d')  # 1 row, 2 columns, second plot
        ax2.scatter(x, y, z, s=4, linewidths=0, c=colors, cmap=color_map, edgecolors='k')
        ax2.set_title('Prediction [kWh/m2]', pad=20, loc='center')
        ax2.set_xlim(-0.75, 0.75)
        ax2.set_ylim(-0.75, 0.75)
        ax2.set_zlim(-0.25, 1.25)
        ax2.grid(False)
        ax2.set_proj_type('ortho')

        ax2.xaxis.pane.fill = False
        ax2.yaxis.pane.fill = False
        ax2.zaxis.pane.fill = False
        ax2.w_xaxis.line.set_visible(False)
        ax2.w_yaxis.line.set_visible(False)
        ax2.w_zaxis.line.set_visible(False)
        ax2.set_axis_off()
    
    if not blank:
        norm = Normalize(vmin=0, vmax=1000)
        # Create a ScalarMappable for colorbar
        sm = ScalarMappable(cmap=color_map, norm=norm)
        sm.set_array([])  # Dummy array for the colorbar
    
        ax3 = plt.subplot(gs[2])
        plt.subplots_adjust(left=0.05, right=0.85, wspace=0.2)
        # Add the colorbar to the far right
        cbar = plt.colorbar(sm, cax=ax3, ax=ax3, orientation='vertical', pad=0.05)
        cbar.set_label('Irradiance [kWh/m2]')
    
    plt.tight_layout()
    
    if show:
        matplotlib.use('TkAgg')
        
        plt.show()
    
    if save:
       save_img(plt, name, path) 

def from_file(path):
    array = np.load(path)

    plot(0, array, vectors=[], targets=array[:, 6]. T, show_normals=False)

def from_sample(sample, idx, values, show, save, name, path):
    plot(name, 
        sample['pos'][idx, :, :].numpy(),
        vectors=[],
        targets = sample['y'][idx, :].numpy(),
        values = values,
        show_normals=False, 
        vector_length=1.0,
        save=save,
        show=show,
        name=name,
        path = path
        )

def binned_cm(target, 
            prediction, 
            min, 
            max, 
            bins, 
            name='',
            path = '',
            show=False, 
            save=True
            ):
    bin_edges = torch.linspace(min, max, steps=bins+1)

    names = [str(int(bin_edges[i].item())) + '-' + str(int(bin_edges[i+1].item())) for i in range(len(bin_edges[:-1]))]

    # Index calculation target
    diff = target.unsqueeze(1) - bin_edges.unsqueeze(0)
    cumsum = torch.cumsum(diff >= 0, dim=1)
    target_idxs = torch.argmax(cumsum, dim=1)

    # Index calculation data
    diff = prediction.unsqueeze(1) - bin_edges.unsqueeze(0)
    cumsum = torch.cumsum(diff >= 0, dim=1)
    prediction_idxs = torch.argmax(cumsum, dim=1)

    cf_matrix = confusion_matrix(target_idxs, prediction_idxs, labels=range(0, bins))
    
    df_cm = pd.DataFrame(cf_matrix, columns=np.unique(names), index = np.unique(names))

    df_cm.index.name = 'Actual Irradiance [kWh/m2]'
    df_cm.columns.name = 'Predicted Irradiance [kWh/m2]'

    f, ax = plt.subplots(figsize=(18, 18))

    cmap = sn.color_palette("flare", as_cmap=True)
    sn.heatmap(df_cm, cbar=True, annot=True, cmap=cmap, square=True, fmt='.0f',
                annot_kws={'size': 10}, norm=LogNorm(), vmin=None, vmax=None, linewidths=1, linecolor='black')
    plt.title(name)
    
    if show:
        matplotlib.use('TkAgg')
        plt.show()
        matplotlib.use('Agg')
    
    if save:
        logging.info(f"Saving confusion matrix in {path}")
        save_img(plt, name, path)
    
    return cf_matrix, bin_edges, names