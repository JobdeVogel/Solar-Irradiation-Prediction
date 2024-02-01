import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
import os

def save_img(plt, name, path):
    path = os.path.join(path, name)

    plt.savefig(path, dpi=600)

def denormalize(values, clamp_min=0, clamp_max=1000):       
    values = (values + 1) / 2
    values *= (clamp_max - clamp_min)
    
    return values

def save_img(plt, name, path):
    path = os.path.join(path, name)
    
    plt.savefig(path, dpi=600)

def visualize(points, targets, name=None, path=None, save=False, show=True):
    """_summary_

    Args:
        points (np.array(npoints, 3)): _description_
        targets (np.array(npoints)): _description_
        name (_type_): _description_
        path (_type_): _description_
        save (bool, optional): _description_. Defaults to False.
        show (bool, optional): _description_. Defaults to True.
    """
             
    # Transform data back to kWh
    targets = denormalize(targets)
                    
    color_map = plt.get_cmap('coolwarm')
    norm = Normalize(0, 1000)
    colors = color_map(norm(targets))
    
    fig = plt.figure(figsize=(12,7))
    gs = GridSpec(1, 2, width_ratios=[1, 0.1])
    
    ax = fig.add_subplot(gs[0], projection='3d')

    x, y, z = points.T
    
    img = ax.scatter(x, y, z, s=10, c=colors, linewidths=0)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 1)

    # ax.set_position([0, 0.05, 1, 2])
    norm = Normalize(vmin=0, vmax=1000)
    
    # Create a ScalarMappable to map the color values to a colormap
    sm = ScalarMappable(cmap=color_map, norm=norm)
    sm.set_array([])  # An empty array is needed here
    
    cbar_ax = fig.add_subplot(gs[1])
    cbar = plt.colorbar(sm, ax=cbar_ax, label='Irradiance [kWh/m2]')
    
    ax.grid(False)
    ax.axis('off')
    
    cbar_ax.grid(False)
    cbar_ax.axis('off')

    # ax.view_init(elev=35, azim=45, roll=0)
    save_name = 'hoi'
    ax.set_title(save_name, y=-0.01)
    
    if save and name != None and path != None:
        save_img(fig, name, path)
    
    if show:
        plt.show()
