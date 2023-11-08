import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
import torch

import os

def get_im_data(dataset, idx):
    data, targets = dataset[idx]
    
    if data.shape[1] > 3:
        points, meta = data[:, :3], data[:, 3:]

        return points.numpy(), meta.numpy(), targets.numpy()
    else:
        points = data[:, :3]

        return points.numpy(), np.array([]), targets.numpy()

def save_img(plt, name, path):
    path = os.path.join(path, name)
    
    plt.savefig(path, dpi=600)

def compute_errors(targets, predictions):
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()
    
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    
    return targets - predictions
    
def plot(points, vectors, targets, vector_length=0.001, show_normals=False, show=False, save=False, save_name='temp', save_path='./images', error=False):
    if not error:
        color_map = plt.get_cmap('coolwarm')
    else:
        color_map = plt.get_cmap('seismic')

    norm = Normalize(vmin=min(targets), vmax=max(targets))

    colors = color_map(norm(targets))[:, :3]

    fig = plt.figure(figsize=(12,7))
    gs = GridSpec(1, 2, width_ratios=[1, 0.1])
    
    ax = fig.add_subplot(gs[0], projection='3d')

    x, y, z = points.T

    img = ax.scatter(x, y, z, s=10, c=colors, linewidths=0)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 1)

    ax.set_position([0, 0.05, 1, 1.4])
    
    if not error:
        norm = Normalize(vmin=0, vmax=1000)
    else:
        norm = Normalize(vmin=-1, vmax=1)
    
    # Create a ScalarMappable to map the color values to a colormap
    sm = ScalarMappable(cmap=color_map, norm=norm)
    sm.set_array([])  # An empty array is needed here

    cbar_ax = fig.add_subplot(gs[1])

    if not error:
        cbar = plt.colorbar(sm, ax=cbar_ax, label='Irradiance [kWh/m2]')
    else:
        cbar = plt.colorbar(sm, ax=cbar_ax, label='Errors')

    if show_normals:
        vx, vy, vz = vectors.T
        ax.quiver(x, y, z, vx, vy, vz, colors=colors, linewidths=0.5, length=vector_length, normalize=True)

    ax.grid(False)
    ax.axis('off')
    
    cbar_ax.grid(False)
    cbar_ax.axis('off')

    ax.view_init(elev=35, azim=45, roll=0)

    ax.set_title(save_name, y=-0.01)

    if save:
        save_img(plt, save_name, save_path)

    if show:
        plt.show()

    return plt

if __name__ == '__main__':
    from pointnet.dataset import IrradianceDataset
    
    dataset = IrradianceDataset(
            root="D:\\Master Thesis Data\\raw",
            split='train',
            dtype=np.float32,
            slice=100,
            normals=True,
            npoints=2500,
            transform=True,
            resample=False,
            preload=True
        )

    points, vectors, targets = get_im_data(dataset, 0)
    my_plot = plot(points, vectors, targets, show=True, save=True, save_name='temp', save_path='C:\\Users\\Job de Vogel\\Desktop')