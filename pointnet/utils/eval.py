import os
import torch
import time
from torchvision.io import read_image
from torchvision.utils import make_grid

import numpy as np

import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def visualize_pointcloud(pointcloud: np.array, color_values: np.array, path: str, visualize=False):   
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25, origin=[0, 0, 0])
    
    color_map = plt.get_cmap('coolwarm')
    
    norm = Normalize(vmin=min(color_values), vmax=max(color_values))

    # Map normalized values to colors
    colors = color_map(norm(color_values))[:, :3]       
    
    pcd.colors = o3d.utility.Vector3dVector(colors)    
    
    # Create a visualizer and set view parameters
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(pcd)
    visualizer.add_geometry(coord_frame)

    # Set the view parameters
    view_control = visualizer.get_view_control()
    view_control.set_zoom(0.8)  # Adjust zoom (e.g., zoom_factor > 1 for zoom-in)
    view_control.set_front([1, 1, 1])  # Set the front direction as a 3D vector
    view_control.set_lookat([0, 0, 0])  # Set the point to look at as a 3D vector
    view_control.set_up([0, 0, 1])  # Set the up direction as a 3D vector
    view_control.change_field_of_view(-10)

    # Update and render
    visualizer.poll_events()
    visualizer.update_renderer()
    
    visualizer.capture_screen_image(path)
    
    if visualize:
        visualizer.run()
    
    return read_image(path)

def eval_image(points, classifier, name, device, path=None):
    if path == None:
        path = './images'
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    eval_points = torch.unsqueeze(points, dim=0)

    eval_points = eval_points.transpose(2, 1)
    
    eval_points_cuda = eval_points.to(device)
    
    with torch.no_grad():        
        classifier = classifier.eval()
        
        pred, _, _ = classifier(eval_points_cuda)
        pred = pred.to('cpu')
    
    start = time.perf_counter()

    path = os.path.join(path, name)
    pixel_array = visualize_pointcloud(points.numpy(), pred.detach().numpy(), path)
    
    grid = make_grid(pixel_array)
    print(f'Saved evaluation image in {round(time.perf_counter() - start, 2)}s')
    
    return grid