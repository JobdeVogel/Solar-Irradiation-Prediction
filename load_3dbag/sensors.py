import Rhino.Geometry as rg
import System
import math
import helpers
import sys
import numpy as np
from parameters.params import _MINIMUM_ANGLE, _MINIMUM_AREA, _WALL_RAY_TOLERANCE, QUAD_ONLY

from helpers.mesh import join_meshes, get_face_vertices

def delete_invalid_sensors(mesh, sensors, normals, minimum_angle = _MINIMUM_ANGLE, minimum_area=_MINIMUM_AREA):
    """_summary_

    Args:
        mesh (rg.Mesh): mesh to iterate over the faces
        sensors (list[rg.Point3d]): list of sensors
        normals (list[rg.Vector3d]): list of normals for the sensors
        minimum_angle (float, optional): Minimum corner angle for the face. Defaults to _MINIMUM_ANGLE.
        minimum_area (_type_, optional): _description_. Defaults to _MINIMUM_AREA.
    """
    
    # Iterate over the mesh faces
    for i, face in enumerate(mesh.Faces):
        
        # Check if the face is a triangle
        if face.IsTriangle:
            # Get the vertices of the face
            vertices = get_face_vertices(face, mesh)
            
            # Generate a triangle from the vertices
            triangle = rg.Triangle3d(vertices[0], vertices[1], vertices[2])
            
            # Find the lowest angle of the triangle
            min_angle = min([triangle.AngleA, triangle.AngleB, triangle.AngleC])
            
            # If the minimum angle is lower than acceptable or the area is lower than acceptable,
            # set the sensor and normal to a None value
            if min_angle < minimum_angle:
                sensors[i] = None
                normals[i] = None
            elif triangle.Area < minimum_area:
                sensors[i] = None
                normals[i] = None

    return sensors, normals

def offset_points(points, vectors, offset):
    """Offset points using numpy in the direction of a vector

    Args:
        points (list[rg.Point3d]): list of points to offset
        vectors (list[rg.Vector3d]): list of vectors to use for the offset
        offset (float): distance to offset

    Returns:
        points (list[rg.Point3d]): list of offsetted points
    """
    
    # Convert the points and vectors to numpy arrays
    np_points = np.array([[point.X, point.Y, point.Z] for point in points])
    np_vectors = np.array([[vector.X, vector.Y, vector.Z] for vector in vectors]) * offset
    
    # Offset the points
    new_points = np_points + np_vectors
    points = [rg.Point3d(*point) for point in new_points]
    
    return points

def is_above_mesh(point, meshes, ray_vector=rg.Vector3d(0,0,-1)):
    """Check if a point is above any of a list of meshes

    Args:
        point (rg.Point3d): point
        meshes (list[rg.mesh]): list of meshes
        ray_vector (rg.Vector3d, optional): Direction to check intersection. Defaults to rg.Vector3d(0,0,-1).

    Returns:
        bool: True if above any of the meshes
    """
    ray = rg.Ray3d(point, ray_vector)
    
    for mesh in meshes:
        if rg.Intersect.Intersection.MeshRay(mesh, ray) > 0:
            return True

    return False

def roof_ray_intersection(point, roofs, face_height, offset):
    """Check if a point intersects with a list of roofs, based on a maximum distance. If this
    is the case, offset the point.

    Args:
        point (rg.Point3d): point
        roofs (list[rg.Mesh]): list of roof meshes to check
        face_height (float): height of the face for the corresponding sensorpoint mesh
        offset (float): distance to offset

    Returns:
        bool: True if point intersects
        point (rg.Point3d): offsetted point if offesetted else original point
    """
    
    # Generate an upward ray
    upwards_vector = rg.Vector3d(0,0,1)
    ray = rg.Ray3d(point, upwards_vector)
    
    # Iterate over the roofs
    for roof in roofs:
        # Compute the distance between roof and point
        distance = rg.Intersect.Intersection.MeshRay(roof, ray)
        
        # Check if the distance is smaller than half the face height
        if 0 < distance < 0.5 * face_height:
            
            # Offset the point
            point += upwards_vector * distance + rg.Vector3d(0,0,offset)
            return True, point 

    return False, point

def wall_ray_intersection(point, normal, walls, grid_size, offset, tolerance=_WALL_RAY_TOLERANCE):
    """Compute the intersection between a point on a wall and a list of wall meshes. If the
    intersection is close enough, move the point to the intersection + an offset

    Args:
        point (rg.Point3d): point on a wall
        normal (rg.Vector3d): normal for the point on the wall
        walls (list[rg.Mesh]): list of wall meshes
        grid_size (float): grid size
        offset (float): offset distance
        tolerance (f;pat, optional): Intersection tolerance. Defaults to _WALL_RAY_TOLERANCE.

    Returns:
        success (bool): True if intersection occured
        point (rg.Point): offsetted point if offesetted else original point
    """
    
    # Hrnrtate an upward vector
    upwards_vector = rg.Vector3f(0,0,1)
    
    # Compute the perpendicular vector in relation to the normals
    left_vector = rg.Vector3f.CrossProduct(normal, upwards_vector)
    right_vector = -left_vector
    
    # Convert the vectors to rays
    left_ray = rg.Ray3d(point, rg.Vector3d(left_vector))
    right_ray = rg.Ray3d(point, rg.Vector3d(right_vector))
    
    # Store the success value
    success = False
    
    # Iterate over the walls
    for wall in walls:
        
        # Compute the intersections to the left and right
        left_distance = rg.Intersect.Intersection.MeshRay(wall, left_ray)
        right_distance = rg.Intersect.Intersection.MeshRay(wall, right_ray)
        
        # Check if the intersection is close enough
        if 0 < left_distance < 0.5 * grid_size - tolerance:
            if left_distance < right_distance:
                # Move the point
                left_vector = rg.Vector3d.Multiply(rg.Vector3d(left_vector), left_distance + offset)
                point += left_vector
                success = True
                break
        
        # Check if the intersection is close enough
        elif 0 < right_distance < 0.5 * grid_size - tolerance:
            if right_distance < left_distance:
                # Move the point
                right_vector = rg.Vector3d.Multiply(rg.Vector3d(right_vector), right_distance + offset)
                point += right_vector
                success = True
                break

    return success, point

def compute(ground, roofs, walls, building_heights, grid_size, offset, quad_only=QUAD_ONLY, logger=False):
    """Compute the sensorpoints for building meshes

    Args:
        ground (rg.Mesh): ground mesh
        roofs (list[rg.Mesh]): list of roof meshes
        walls (list[rg.Mesh]): list of wall meshes
        building_heights (list[float]): list of corresponding building height
        grid_size (float): grid size
        offset (float): offset value for the sensorpoints
        quad_only (bool, optional): Indicate if quad faces should be computed only. Defaults to QUAD_ONLY.

    Returns:
        sensorpoints (list[rg.Point3d]): sensorpoints
        normals (list[rg.Vector3d])): normals for the sensorpoints
    """
    
    # Store the sensorpoints and normals of the sensorpoints            
    sensorpoints = []
    normals = []
    
    # Compute the centroids and normals for the ground mesh
    ground_normals = list(ground.FaceNormals)
    ground_centroids = [ground.Faces.GetFaceCenter(System.Int32(i)) for i in range(ground.Faces.Count)] 
    
    # Offset the ground centroids
    ground_centroids = offset_points(ground_centroids, ground_normals, offset)
    
    
    # Delete the invalid sensors for the ground
    ground_centroids, ground_normals = delete_invalid_sensors(ground, ground_centroids, ground_normals)
    

    # Compute the centroids and normals for the roof meshes
    joined_roofs = join_meshes(roofs)
    roof_normals = joined_roofs.FaceNormals
    roof_centroids = [joined_roofs.Faces.GetFaceCenter(System.Int32(i)) for i in range(joined_roofs.Faces.Count)]
    
    # Offset the roof centroids
    roof_centroids = offset_points(roof_centroids, roof_normals, offset)
    
    # Append the computed centroids to the sensors
    sensorpoints.extend(ground_centroids)
    sensorpoints.extend(roof_centroids)
    
    # Append the computed centroid normals to the normals
    normals.extend(ground_normals)
    normals.extend(roof_normals)
    
    # Join the ground and roof into one mesh
    meshes = [ground] + roofs
    joined_meshes = join_meshes(meshes)
    
    # Join the walls into one mesh
    joined_walls = join_meshes(walls)
    
    # Iterate over all the walls
    for i, (wall, height) in enumerate(zip(walls, building_heights)):
        # Compute the face normals for the wall
        wall.FaceNormals.ComputeFaceNormals()
        
        # Get the face centroids for the wall
        wall_sensors = [wall.Faces.GetFaceCenter(j) for j in range(wall.Faces.Count)]
        
        # Offset the face centroids
        wall_sensors = offset_points(wall_sensors, wall.FaceNormals, offset)
        
        # Compute the height of one face
        face_height = height / int(math.ceil(height / grid_size))    
        
        # Iterate over the points and normals in the wall mesh
        for point, normal in zip(wall_sensors, wall.FaceNormals):
            
            # Check if the point is above a horizontal surface
            if is_above_mesh(point, [joined_meshes]):
                
                # If this is the case, add the centroids to sensors and normals to normals
                sensorpoints.append(point)
                normals.append(normal)
              
            # Check if the sensorpoints are under the corresponding walls' roof (because of splitting error)
            # Sometimes the ground is not splitted properly so wall points seem to be floating in space
            elif is_above_mesh(point, [meshes[i+1]], ray_vector=rg.Vector3d(0,0,1)):
                
                # If this is the case, add the centroids to sensors and normals to normals
                sensorpoints.append(point)
                normals.append(normal)
            
            # The centroid is floating under a roof            
            
            else:
                # Try to move the sensorpoint to close roof + offset
                success, point = roof_ray_intersection(point, [joined_roofs], face_height, offset)
                
                if not success:
                    # Try to move the sensorpoint to close wall + offset
                    success, point = wall_ray_intersection(point, normal, [joined_walls], grid_size, offset)
                    
                    
                    if not success:
                        # Set this point and normal value to None, thus it is invalid
                        point = None
                        normal = None
                    
                # Append the invalid point and normal
                sensorpoints.append(point)
                normals.append(normal)
            
        
    # Only compute the sensors for quad faces, skip the triangle faces at the borders of buildings    
    if quad_only:
        # Join all the meshes together in one list
        meshes = [ground] + roofs + walls
        
        # Generate a single mesh
        joined_mesh = rg.Mesh()
        for mesh in meshes:
            joined_mesh.Append(mesh)
        
        # Generate a single mesh with all horizontal meshes
        horizontal_mesh = rg.Mesh()
        for mesh in meshes:
            horizontal_mesh.Append(mesh)
        
        # Store the new sensorpoints and normals
        new_sensorpoints = []
        new_normals = []
        
        # Iterate over the faces, points and normals
        for face, point, normal in zip(horizontal_mesh.Faces, sensorpoints, normals):
            
            # Check if the face is a triangle face
            if not face.IsTriangle:
                # The face is a quad, so add the point and normal to new sensors and normals
                new_sensorpoints.append(point)
                new_normals.append(normal)
            else:
                # Append a None value, it is a triangle
                new_sensorpoints.append(None)
                new_normals.append(None)
        
        # Overwrite the sensorpoints and normals
        sensorpoints = new_sensorpoints
        normals = new_normals
    
    if logger:
        logger.debug(f"Computed {len(sensorpoints)} sensorpoints with grid_size {grid_size} and offset {offset}")
    return sensorpoints, normals

def filter_sensors(sensorpoints, normals):
    """Filter points and normals based on None values

    Args:
        sensorpoints (list): list of rg.Point3d and None values
        normals (list): list of rg.Vector3d and None values

    Returns:
        filtered_points (list[rg.Point3d]): filtered points
        filtered_normals (list[rg.Vector3d]): filtered normals
        pointmap (list[bool]): list indicating if point was None or not for restoration in postprocessing
    """
    
    # Store the results
    filtered_points = []
    filtered_normals = []
    pointmap = []

    # Iterate over points and normals
    for point, normal in zip(sensorpoints, normals):
        # Check if the point is None
        if point is not None:
            filtered_points.append(point)
            filtered_normals.append(normal)
            pointmap.append(True)
        else:
            pointmap.append(False)
    
    return filtered_points, filtered_normals, pointmap
    
    
    