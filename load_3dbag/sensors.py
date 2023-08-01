from parameters.params import LOGGER
import Rhino.Geometry as rg
import Rhino
import System
import time
import sys
import rhino3dm
import pprofile
import math
import numpy as np
import copy
from parameters.params import _MINIMUM_ANGLE, _MINIMUM_AREA, _WALL_RAY_TOLERANCE, QUAD_ONLY

from helpers.mesh import join_meshes, get_face_vertices

'''
Exception line 245 at index 11
'''

def delete_invalid_sensors(mesh, sensors, normals, minimum_angle = _MINIMUM_ANGLE, minimum_area=_MINIMUM_AREA):
    
    for i, (face, normal) in enumerate(zip(mesh.Faces, mesh.FaceNormals)):

        if face.IsTriangle:
            vertices = get_face_vertices(face, mesh)
            
            triangle = rg.Triangle3d(vertices[0], vertices[1], vertices[2])
            min_angle = min([triangle.AngleA, triangle.AngleB, triangle.AngleC])
            
            if min_angle < minimum_angle:
                sensors[i] = None
                normals[i] = None
            elif triangle.Area < minimum_area:
                sensors[i] = None
                normals[i] = None

#        elif face.IsQuad:
#            rectangle = rg.Rectangle3d(rg.Plane.WorldXY, vertices[0], vertices[2])
#            
#            triangle_1 = rg.Triangle3d(vertices[0], vertices[1], vertices[2])
#            triangle_2 = rg.Triangle3d(vertices[2], vertices[3], vertices[0])
#            
#            min_angle = min([triangle_1.AngleA, triangle_1.AngleB, triangle_1.AngleC, triangle_2.AngleA, triangle_2.AngleB, triangle_2.AngleC]) 
#            
#            if min_angle < minimum_angle:
#                sensors[i] = None
#                normals[i] = None
#            elif triangle_1.Area < minimum_area or triangle_2.Area < minimum_area:
#                sensors[i] = None
#                normals[i] = None

    
    return sensors, normals

# def compute_centroids(mesh, offset):
#     vectors = [rg.Vector3f.Multiply(mesh.FaceNormals[i], offset) for i in xrange(mesh.Faces.Count)]
#     centroids = [mesh.Faces.GetFaceCenter(i) for i in xrange(mesh.Faces.Count)]
    
#     for centroid, vec in zip(centroids, vectors):
#         centroid.Transform(rg.Transform.Translation(vec))
    
#     return centroids, vectors

# Make faster with Numpy, using arrays of vectors
def perpendicular_wall_vectors(vector):
    z_vector = rg.Vector3f(0,0,1)
    vec1 = rg.Vector3f.CrossProduct(vector, z_vector)
    vec2 = -vec1
    
    return (vec1, vec2)

def _to_numpy(points, vectors, offset):
    np_points = np.array([[point.X, point.Y, point.Z] for point in points])
    np_vectors = np.array([[vector.X, vector.Y, vector.Z] for vector in vectors]) * offset
    
    new_points = np_points + np_vectors
    rg_points = [rg.Point3d(*point) for point in new_points]
    
    return rg_points
    
#11.99%
def offset_points(points, vectors, offset):
    # offset_vectors = [rg.Vector3f.Multiply(vec, offset) for vec in vectors]
    # points = [point + offset_vec for point, offset_vec in zip(points, offset_vectors)]

    points = _to_numpy(points, vectors, offset)

    return points

def is_above_mesh(point, meshes, ray_vector=rg.Vector3d(0,0,-1)):
    intersects = False
    ray = rg.Ray3d(point, ray_vector)
    
    for mesh in meshes:
        #11.28%
        if rg.Intersect.Intersection.MeshRay(mesh, ray) > 0:
            intersects = True
            break

    return intersects

def roof_ray_intersection(point, roofs, face_height, grid_size, offset):
    upwards_vector = rg.Vector3d(0,0,1)
    ray = rg.Ray3d(point, upwards_vector)
    
    success = False
    for roof in roofs:
        # 9.80%
        distance = rg.Intersect.Intersection.MeshRay(roof, ray)
        
        if 0 < distance < 0.5 * face_height:
            point += upwards_vector * distance + rg.Vector3d(0,0,offset)
            success = True
            break

    return success, point

def wall_ray_intersection(point, normal, walls, grid_size, offset, tolerance=_WALL_RAY_TOLERANCE):
    upwards_vector = rg.Vector3f(0,0,1)
    
    left_vector = rg.Vector3f.CrossProduct(normal, upwards_vector)
    right_vector = -left_vector
    
    left_ray = rg.Ray3d(point, rg.Vector3d(left_vector))
    right_ray = rg.Ray3d(point, rg.Vector3d(right_vector))
    
    success = False
    for wall in walls:
        # 20%
        left_distance = rg.Intersect.Intersection.MeshRay(wall, left_ray)
        right_distance = rg.Intersect.Intersection.MeshRay(wall, right_ray)
        
        if 0 < left_distance < 0.5 * grid_size - tolerance:
            if left_distance < right_distance:
                left_vector = rg.Vector3d.Multiply(rg.Vector3d(left_vector), left_distance + offset)
                point += left_vector
                success = True
                break
        
        elif 0 < right_distance < 0.5 * grid_size - tolerance:
            if right_distance < left_distance:
                right_vector = rg.Vector3d.Multiply(rg.Vector3d(right_vector), right_distance + offset)
                point += right_vector
                success = True
                break

    return success, point    

def compute(ground, roofs, walls, building_heights, grid_size, offset, quad_only=QUAD_ONLY):            
    sensorpoints = []
    normals = []
    
    ground_normals = list(ground.FaceNormals)
    ground_centroids = [ground.Faces.GetFaceCenter(System.Int32(i)) for i in range(ground.Faces.Count)] 
    
    ground_centroids = offset_points(ground_centroids, ground_normals, offset)
    
    ground_centroids, ground_normals = delete_invalid_sensors(ground, ground_centroids, ground_normals)

    joined_roofs = join_meshes(roofs)
    roof_normals = joined_roofs.FaceNormals
    roof_centroids = [joined_roofs.Faces.GetFaceCenter(System.Int32(i)) for i in range(joined_roofs.Faces.Count)]
    
    roof_centroids = offset_points(roof_centroids, roof_normals, offset)
    
    sensorpoints.extend(ground_centroids)
    normals.extend(ground_normals)
    sensorpoints.extend(roof_centroids)
    normals.extend(roof_normals)
    
    meshes = [ground] + roofs
    joined_meshes = join_meshes(meshes)
    
    joined_walls = join_meshes(walls)
    
    for i, (wall, height) in enumerate(zip(walls, building_heights)):
        wall.FaceNormals.ComputeFaceNormals()
        
        wall_normals = wall.FaceNormals
        wall_sensors = [wall.Faces.GetFaceCenter(j) for j in range(wall.Faces.Count)]
        wall_sensors = offset_points(wall_sensors, wall_normals, offset)
        
        face_height = height / int(math.ceil(height / grid_size))
        
        # LOOSING POINTS HERE!!!!
        for point, normal in zip(wall_sensors, wall_normals):
            # 15.88%            
            if is_above_mesh(point, [joined_meshes]):
                sensorpoints.append(point)
                normals.append(normal)
            # Check if the sensorpoints are under the corresponding walls' roof (because of splitting error)
            elif is_above_mesh(point, [meshes[i+1]], ray_vector=rg.Vector3d(0,0,1)):
                sensorpoints.append(point)
                normals.append(normal)
            else:
                # Try to move the sensorpoint to close roof + offset
                success, point = roof_ray_intersection(point, [joined_roofs], face_height, grid_size, offset)
                
                if not success:
                    # Try to move the sensorpoint to close wall + offset
                    
                    # 30.56%
                    success, point = wall_ray_intersection(point, normal, [joined_walls], grid_size, offset)
                    
                    if not success:
                        point = None
                        normal = None
                
                sensorpoints.append(point)
                normals.append(normal)
    
    meshes = [ground] + roofs + walls
    joined_mesh = rg.Mesh()
    for mesh in meshes:
        joined_mesh.Append(mesh)
    
    if quad_only:
        horizontal_mesh = rg.Mesh()
        for mesh in meshes:
            horizontal_mesh.Append(mesh)
        
        new_sensorpoints = []
        new_normals = []
        for face, point, normal in zip(horizontal_mesh.Faces, sensorpoints, normals):
            if not face.IsTriangle:
                new_sensorpoints.append(point)
                new_normals.append(normal)
            else:
                new_sensorpoints.append(None)
                new_normals.append(None)
        
        sensorpoints = new_sensorpoints
        normals = new_normals
    
    LOGGER.debug(f"Computed {len(sensorpoints)} sensorpoints with grid_size {grid_size} and offset {offset}")
    return sensorpoints, normals

def filter_sensors(sensorpoints, normals):
    filtered_points = []
    filtered_normals = []
    pointmap = []

    for point, normal in zip(sensorpoints, normals):
        if point is not None:
            filtered_points.append(point)
            filtered_normals.append(normal)
            pointmap.append(True)
        else:
            pointmap.append(False)
    
    return filtered_points, filtered_normals, pointmap
    
    
    