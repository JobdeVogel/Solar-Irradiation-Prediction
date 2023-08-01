import Rhino.Geometry as rg
from math import pi
import System

def augment(mesh, pointcloud, number, 
                rot_axis = rg.Vector3d(0,0,1), 
                rot_point = rg.Point3d(0,0,0)
                ):

    pointcloud = System.Array[rg.Point3d](pointcloud)

    meshes = [mesh.Duplicate() for _ in range(number)]
    pointclouds = [rg.PointCloud(pointcloud) for _ in range(number)]

    angles = [i * 0.5 * pi for i in range(number)]
    
    for mesh, pointcloud, angle in zip(meshes, pointclouds, angles):
        mesh.Rotate(angle, rot_axis, rot_point)
        pointcloud.Rotate(angle, rot_axis, rot_point)
    
    pointclouds = [pointcloud.GetPoints() for pointcloud in pointclouds]
    return meshes, pointclouds

def augment(meshes, pointcloud, normalcloud, number, 
                rot_axis = rg.Vector3d(0,0,1), 
                rot_point = rg.Point3d(0,0,0),
                detailed_meshes=[None, None, None]):

    pointcloud = System.Array[rg.Point3d](pointcloud)
    pointclouds = [rg.PointCloud(pointcloud) for _ in range(number)]
    
    normalcloud = System.Array[rg.Point3d]([rg.Point3d(normal.X, normal.Y, normal.Z) for normal in normalcloud])
    normalclouds = [rg.PointCloud(normalcloud) for _ in range(number)]
    
    angles = [i * 0.5 * pi for i in range(number)]
    
    augmented_meshes = []
    augmented_detailed_meshes = []

    for angle, pointcloud, normalcloud in zip(angles, pointclouds, normalclouds):
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

        pointcloud.Rotate(angle, rot_axis, rot_point)
        normalcloud.Rotate(angle, rot_axis, rot_point)
        augmented_meshes.append(temp_meshes)
        augmented_detailed_meshes.append(temp_detailed_meshes)
    
    pointclouds = [cloud.GetPoints() for cloud in pointclouds]
    normalclouds = [cloud.GetPoints() for cloud in normalclouds]
    
    for cloud in normalclouds:
        cloud = System.Array[rg.Vector3f]([rg.Vector3f(point.X, point.Y, point.Z) for point in cloud])
    
    return augmented_meshes, pointclouds, normalclouds, augmented_detailed_meshes