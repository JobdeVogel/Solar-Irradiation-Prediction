import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
import os
import sys

import numpy as np

import psutil


def save_img(plt, name, path):
    path = os.path.join(path, name)
    
    plt.savefig(path, dpi=125)

def plot(idx, 
        points, 
        vectors=[],
        targets =[],
        values = [],
        show_normals=False, 
        vector_length=1.0,
        save=False,
        show=True,
        name='',
        path = ''
        ):    
    
    # Set the figure size
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])
    
    x, y, z = points[:, :3].T
    
    if len(values) == 0:
        values = np.random.rand(len(x))
    
    if len(targets) == 0:
        targets = np.random.rand(len(x))
    
    
    color_map = plt.get_cmap('coolwarm')
    norm = Normalize(vmin=np.min(targets), vmax=np.max(targets))
    
    colors = color_map(norm(targets))
    
    ax1 = plt.subplot(gs[0], projection='3d')
    ax1.scatter(x, y, z, s=10, linewidths=0, c=colors, cmap=color_map, edgecolors='k')
    ax1.set_title('Ground truth [kWh/m2]', pad=20, loc='center')
    ax1.set_xlim(-50, 50)
    ax1.set_ylim(-50, 50)
    ax1.set_zlim(0, 100)
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
    
    ax2 = plt.subplot(gs[1], projection='3d')  # 1 row, 2 columns, second plot
    ax2.scatter(x, y, z, s=10, linewidths=0, c=colors, cmap=color_map, edgecolors='k')
    ax2.set_title('Prediction [kWh/m2]', pad=20, loc='center')
    ax2.set_xlim(-50, 50)
    ax2.set_ylim(-50, 50)
    ax2.set_zlim(0, 100)
    ax2.grid(False)
    ax2.set_proj_type('ortho')
    
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False
    ax2.w_xaxis.line.set_visible(False)
    ax2.w_yaxis.line.set_visible(False)
    ax2.w_zaxis.line.set_visible(False)
    ax2.set_axis_off()
    
    
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
        plt.show()
    
    if save:
       save_img(plt, name, path) 

def from_file(path):
    array = np.load(path)
    print(array.shape)

    plot(0, array, vectors=[], targets=array[:, 6]. T, show_normals=False)
    
if __name__ == '__main__':
    matplotlib.use('TkAgg')
    
    path = "D:\\Master Thesis Data\\IrradianceNet\dset300_s\\10-286-590-LoD12-3D\\irradiance_sample_0_augmentation_0.npy"
    from_file(path)