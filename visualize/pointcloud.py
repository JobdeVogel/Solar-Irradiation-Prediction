import matplotlib

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
import os

def save_img(plt, name, path):
    path = os.path.join(path, name)
    
    plt.savefig(path, dpi=125)

def plot(idx, 
        points, 
        vectors=[], 
        show_normals=False, 
        vector_length=1.0
        ):    
    
    fig = plt.figure(figsize=(12,7))
    gs = GridSpec(1, 2, width_ratios=[1, 0.0])
    
    ax = fig.add_subplot(gs[0], projection='3d')

    x, y, z = points[:, :3].T

    img = ax.scatter(x, y, z, s=10, linewidths=0)

    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)
    ax.set_zlim(0, 300)

    ax.set_position([0, 0.05, 1, 2])

    cbar_ax = fig.add_subplot(gs[1])

    if show_normals:
        vx, vy, vz = vectors.T
        ax.quiver(x, y, z, vx, vy, vz, linewidths=0.5, length=vector_length, normalize=True)

    ax.grid(False)
    ax.axis('off')
    
    cbar_ax.grid(False)
    cbar_ax.axis('off')

    # ax.view_init(elev=35, azim=65)

    ax.set_title(str(idx), y=-0.01)

    # save_img(plt, str(idx), "C:\\Users\\Job de Vogel\\Desktop\\results")

    plt.show()