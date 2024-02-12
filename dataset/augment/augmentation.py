import Rhino.Geometry as rg
from math import pi
import System

import sys

def augment(meshes, pointcloud, normalcloud, number, 
                rot_axis = rg.Vector3d(0,0,1), 
                rot_point = rg.Point3d(0,0,0),
                detailed_meshes=[None, None, None]):

    pointcloud = rg.PointCloud(System.Array[rg.Point3d](pointcloud))
    pointclouds = []
    
    normalcloud = rg.PointCloud(System.Array[rg.Point3d]([rg.Point3d(normal.X, normal.Y, normal.Z) for normal in normalcloud]))
    normalclouds = []
    
    angles = [i * 0.5 * pi for i in range(number)]
    
    augmented_meshes = []
    augmented_detailed_meshes = []

    for angle in angles:
        temp_meshes = []
        temp_detailed_meshes = []
        
        for mesh, detailed_mesh in zip(meshes, detailed_meshes):
            augmented_mesh = mesh.Duplicate()
            augmented_mesh.Rotate(angle, rot_axis, rot_point)
            
            temp_meshes.append(augmented_mesh)  
            
            if detailed_meshes != None:
                augmented_mesh = detailed_mesh.Duplicate()
                augmented_mesh.Rotate(angle, rot_axis, rot_point)
                
                temp_detailed_meshes.append(augmented_mesh)

        augmented_meshes.append(temp_meshes)
        augmented_detailed_meshes.append(temp_detailed_meshes)

        temp_pointcloud = pointcloud.Duplicate()
        temp_normalcloud = normalcloud.Duplicate()

        temp_pointcloud.Rotate(angle, rot_axis, rot_point)
        temp_normalcloud.Rotate(angle, rot_axis, rot_point)
        
        pointclouds.append(temp_pointcloud)
        normalclouds.append(temp_normalcloud)
    
    pointclouds = [cloud.GetPoints() for cloud in pointclouds]
    normalclouds = [cloud.GetPoints() for cloud in normalclouds]
    
    for cloud in normalclouds:
        cloud = System.Array[rg.Vector3f]([rg.Vector3f(point.X, point.Y, point.Z) for point in cloud])
    
    return augmented_meshes, pointclouds, normalclouds, augmented_detailed_meshes