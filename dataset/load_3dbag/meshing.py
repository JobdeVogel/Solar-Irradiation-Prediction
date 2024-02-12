import Rhino.Geometry as rg
import System
import math
import random
import sys
import time

import Rhino
import json

from parameters.params import MAX_CONTAINMENT_ITERATIONS, _REDUCE_SEGMENTS_TOLERANCE, _MESH_SPLITTER_BBOX_HEIGHT, _ANGLE_TOLERANCE_POSTP_MESH, _DIAGONAL_LENGTH_RATIO_POSTP_MESH
#import cProfile
  
def temp_save(meshes, path):  
    options = Rhino.FileIO.SerializationOptions()
    data = []
    
    for mesh in meshes:
        data.append(mesh.ToJSON(options))
    
    with open(path, "w") as file:
        json.dump(data, file)  
  
  
def postprocess_mesh(mesh, check=False):
    """Postprocess a mesh such that there are no invalid faces or vertices

    Args:
        mesh (rg.Mesh): mesh
        check (bool, optional): Check if the mesh is valid. Defaults to False.

    Returns:
        mesh (rg.Mesh): postprocessd mesh
    """
    
    # Convert triangle faces to quad
    mesh.Faces.ConvertTrianglesToQuads(_ANGLE_TOLERANCE_POSTP_MESH, _DIAGONAL_LENGTH_RATIO_POSTP_MESH)
    
    # Get vertices and faces    
    vertices = mesh.Vertices
    faces = mesh.Faces
    
    # Initialize a rebuild mesh
    rebuild_mesh = rg.Mesh()

    # Add the mesh vertices and faces to the rebuild mesh
    rebuild_mesh.Vertices.AddVertices(vertices)
    rebuild_mesh.Faces.AddFaces(faces)
    
    # Compute the normals
    rebuild_mesh.Normals.ComputeNormals()
        
    # Transform the mesh to a compact shape
    rebuild_mesh.Compact()
        
    # Cull degenerate faces
    rebuild_mesh.Faces.CullDegenerateFaces()
    
    # Delete zero area mesh faces
    indices =  rebuild_mesh.Faces.GetZeroAreaFaces()[1]
    rebuild_mesh.Faces.DeleteFaces(indices, True)
        
    indices =  rebuild_mesh.Faces.GetZeroAreaFaces()[2]
    rebuild_mesh.Faces.DeleteFaces(indices, True)
        
    return rebuild_mesh

def get_random_face_center(mesh):
    """Extract a random face center from a mesh

    Args:
        mesh (rg.Mesh): mesh

    Returns:
        checkpoint (rg.Point3d): random face center
    """
    
    # Random index
    idx = random.randint(0, len(mesh.Faces))
    
    # Get the vertices from the face
    vertices = mesh.Faces.GetFaceVertices(idx)[1:]
    
    # Compute the center of the face
    center_x = sum(p.X for p in vertices) / len(vertices)
    center_y = sum(p.Y for p in vertices) / len(vertices)
    center_z = sum(p.Z for p in vertices) / len(vertices)

    # Generate a point
    checkpoint = rg.Point3d(center_x, center_y, center_z)   
    
    return checkpoint
    
def is_inside(mesh, curves, max_iterations=MAX_CONTAINMENT_ITERATIONS, logger=False, std=False):
    """Check if a planar mesh is inside any curve in a set of curves. This procedure works by taking a
    mesh face center and then checking if this center is inside the curve. In some special
    cases, this point intersects with the curve. In that case the multiple random faces centers
    are taken until a point does not intersect, or the max containment iterations is reached.
    If it still intersects, it is assumed that the mesh is inside the curve.

    Args:
        mesh (rg.Mesh): planar mesh
        curves (list(rg.NurbsCurve)): list of planar curves
        max_iterations (int, optional): Number of maximum containment iterations. Defaults to MAX_CONTAINMENT_ITERATIONS.

    Returns:
        inside (bool): indicates if inside one of the cuves
    """
      
    # Generate an intial checkpoint
    checkpoint = get_random_face_center(mesh)
        
    # Bool indicating if mesh is inside the curve
    inside = False
    
    # Find a quad mesh and get the face center
    checkpoint = None
    for i, face in enumerate(mesh.Faces):
        if face.IsQuad:
            checkpoint = mesh.Faces.GetFaceCenter(i)
            break
    
    # The mesh does not have quad faces (outlier situation)
    if not isinstance(checkpoint, rg.Point3d):
        # Generate an intial checkpoint
        checkpoint = get_random_face_center(mesh) 
    
    # Iterate over the curves
    for curve in curves:
        # Iterate over the max iterations
        for i in range(max_iterations):
            # Check if the curve contains the checkpoint
            
            if curve.Contains(checkpoint, rg.Plane.WorldXY, tolerance=1e-8) == rg.PointContainment.Inside:
                inside = True
                break
            elif curve.Contains(checkpoint, rg.Plane.WorldXY, tolerance=1e-8) == rg.PointContainment.Coincident:
                inside = True
            
                # Generate a new checkpoint
                checkpoint = get_random_face_center(mesh)
                    
                if i == max_iterations - 1:
                    if logger:
                        logger.warning('Point containment coincident')
                break
    if std:
        print((checkpoint.X, checkpoint.Y, checkpoint.Z))

    return inside

def project_outlines_to_world_xy(outlines):
    """Project polylines to the world XY plane, returns both a polyline and curves format

    Args:
        outlines (list[list[rg.Polyline]]): building outulines as polylines
        
    Returns:
        polylines (list[list[rg.Polyline]]): projected polylines
        curves (list[rg.NurbsCurve]): projected curves    
    """
    
    # Store the projected polylines and curves as lists
    polylines = []
    curves = []
    
    # Project the building outlines on the mesh plane
    for outline_set in outlines:
        # Store projections per building
        temp_polylines = []
        temp_curves = []
        
        # Iterate over the polylines used in the building outlines
        for polyline in outline_set:
            # Reduce the number of segments in the polyline
            polyline.ReduceSegments(_REDUCE_SEGMENTS_TOLERANCE)
            
            # Convert a duplicate polyline to a nurbscurve
            curve = polyline.Duplicate().ToNurbsCurve()
            
            # Project the curve on the mesh plane
            projected_curve = curve.ProjectToPlane(curve, rg.Plane.WorldXY)
            
            # Generate a polyline from the curve
            projected_polyline = projected_curve.TryGetPolyline()[1]
            
            # Append the building polylines and curves to all polylines and curves
            temp_polylines.append(projected_polyline)
            temp_curves.append(projected_curve)
        
        polylines.append(temp_polylines)
        curves.append(temp_curves)    
        
    return polylines, curves

def compute_area(ground, roofs):
    ground_area = rg.AreaMassProperties.Compute(ground).Area
    building_area = sum([rg.AreaMassProperties.Compute(building).Area for building in roofs])
    
    return ground_area, building_area

def remesh_rough(mesh):
    rough_mesh = mesh.Duplicate()
    
    rough_mesh.Reduce(100, False, 3, False, False)

    return rough_mesh

def polyline_isclockwise(polyline):
    """Check if the vertices of a polyline are clockwise

    Args:
        polyline (rg.Polyline): polyline

    Returns:
        bool: True if clockwise
    """
    
    # Connvert to duplicate nurbscurve to avoid in-place changes
    curve = polyline.Duplicate().ToNurbsCurve()
    return curve.ClosedCurveOrientation() == rg.CurveOrientation.Clockwise

def mesh_extrude_polyline(polyline, height, grid_size):
    """Extrude a polyline to a mesh

    Args:
        polyline (rg.Polyline): polyline
        height (float): height of the building
        grid_size (float): approximate size of the mesh faces

    Returns:
        meshh (rg.Mesh): wall mesh
    """
    vertices = []
    faces = []
    
    # Reduce the number of segments to a minimum
    polyline.ReduceSegments(_REDUCE_SEGMENTS_TOLERANCE)
    
    # Extract the segments to a list
    segments = polyline.GetSegments()
    
    # Get the lengths of each segment
    lengths = [segment.Length for segment in segments]
    
    # Compute how many faces each segment should have in horizonal direction
    num_segments = [int(math.ceil(length / grid_size)) for length in lengths]
    
    # Compute how many faces each segment should have in vertical directioin
    levels = int(math.ceil(height / grid_size))
    
    # Total number of faces in horizontal directtion
    num_p = sum(num_segments)
    
    # Vertices on the ground level
    base_vertices = []
    
    # Vertices all together
    vertices = []
    
    # Iterate over the segments
    for segment, num_segment in zip(segments, num_segments):
        
        # Iterate over the number of segments
        for i in range(num_segment):
            parameter = i * (1 / num_segment)
            
            # Add the vertex to the base vertices
            base_vertices.append(segment.PointAt(parameter))
    
    # Iterate over the vertical levels
    for j in range(levels + 1):
        parameter = j * (height / levels)
        
        # Iterate over the base vertices
        for vertex in base_vertices:
            # Add a base vertex at each level
            vertices.append(rg.Point3d(vertex.X, vertex.Y, vertex.Z + parameter))
    
    # Generate the mesh faces
    faces = System.Array[rg.MeshFace]([
        rg.MeshFace(i + j * num_p, i + (j + 1) * num_p, (i + 1) % num_p + (j + 1) * num_p, (i + 1) % num_p + j * num_p)
        for j in range(levels)
        for i in range(num_p)
    ])
    
    # Initialize a new mesh
    mesh = rg.Mesh()
    
    # Add the vertices and faces to the mesh
    mesh.Vertices.AddVertices(System.Array[rg.Point3d](vertices))
    mesh.Faces.AddFaces(faces)
    
    return mesh

def generate_vertical(building_outlines, courtyard_outlines, heights, grid_size, logger=False):
    """_summary_

    Args:
        building_outlines (list[list[rg.Polyline]]): List of building outlines
        courtyard_outlines (list[list[rg.Polyline]]): Inner courtyard polylines of the buildings
        heights (list[float]): heigths of the buildings
        grid_size (float): grid size

    Returns:
        meshes (list[rg.Mesh]): walls for the buildings based on the outlines
        outlines (list[rg.Polyline]): outlines of the buildings, possibly reversed direction
    """
    
    # Store the outputs in lists
    meshes = []

    # Iterate over the buildings
    for building, courtyard, height in zip(building_outlines, courtyard_outlines, heights):
        # Store courtyard and building walls in one temp mesh
        temp_mesh = rg.Mesh()
        
        # Reverse direction of outlines if necessary
        for outline in building:
            # Reverse direction if necessary
            if not polyline_isclockwise(outline):
                outline.Reverse()
            
            # Extrude the polyline in z direction based on height
            mesh = mesh_extrude_polyline(outline, height, grid_size)
            
            # Append the results to list variables
            temp_mesh.Append(mesh)
        
        # Iterate over courtyards
        for outline in courtyard:
            # Reverse directtion if necessary
            if polyline_isclockwise(outline):
                outline.Reverse()
            
            # Appeend the results to list variables
            mesh = mesh_extrude_polyline(outline, height, grid_size)
            temp_mesh.Append(mesh)
        
        # Append the building walls processed, as a single mesh        
        meshes.append(postprocess_mesh(temp_mesh))

    if logger:
        logger.debug(f'Generated {len(meshes)} meshes and outlines.')
    return meshes

def extrude(polyline):
    lower_polyline = rg.Polyline(
        [p + rg.Vector3d(0,0,-1) for p in polyline]
    )
    
    mesh = rg.Mesh()
    for pt in lower_polyline:
        mesh.Vertices.Add(pt)
        mesh.Vertices.Add(pt + rg.Vector3d(0,0,2))
    
    for i in range(len(polyline) - 1):
        mesh.Faces.AddFace(i * 2, i * 2 + 1, (i + 1) * 2 + 1, (i + 1) * 2)
    
    mesh.Vertices.CullUnused()
    mesh.Vertices.CombineIdentical(True, True)
    mesh.RebuildNormals()
    
    return mesh

def generate_horizontal(ground_outline, building_polylines, courtyard_polylines, building_curves, courtyard_curves, heights, grid_size, size, logger=False):
    """Generate ground and roofs by splitting a mesh plane

    Args:
        ground_outline (rg.Rectangle3d): ground patch outline
        building_outlines (list[list[rg.NurbsCurve]]): outlines for buildings
        courtyard_outlines (list[list[rg.NurbsCurve]]): outlines for courtyards
        heights (list[float]): building heights
        grid_size (float): grid size

    Returns:
        ground (rg.Mesh): 2D mesh for ground
        roofs (list[rg.Mesh]): roof meshes
        valid (list[bool]): indicates if a roof mesh is valid
    """
    
    # Generate plane parameters
    box_interval = rg.Box(ground_outline.BoundingBox)
    plane_width = rg.Interval(box_interval.X[0], box_interval.X[1])
    plane_height = rg.Interval(box_interval.Y[0], box_interval.Y[1])
    plane = rg.Plane.WorldXY

    # Compute the number of face divisions for the mesh plane    
    width_divisions = System.Int32(int(plane_width.Length / grid_size))
    length_divisions = System.Int32(int(plane_height.Length / grid_size))
    
    # Generate the mesh plane
    mesh_plane = rg.Mesh.CreateFromPlane(plane, plane_width, plane_height, width_divisions, length_divisions)   
    
    # Generate the splitters from the inner and outer polylines
    params = rg.MeshingParameters.QualityRenderMesh
    bbox = rg.BoundingBox(0,0,-_MESH_SPLITTER_BBOX_HEIGHT,size,size,_MESH_SPLITTER_BBOX_HEIGHT)
    
    # Store the splitters in a list
    splitters = []
    valid = []
    
    # Iterate over the buildings
    for outline_set in building_curves:
        # Store the splitters per building
        temp_splitters = []
        
        # Iterate over the outlines
        for curve in outline_set:
            # Create a duplicate curve
            temp_curve = curve.Duplicate()
            
            # Translate the curve to z = -1
            temp_curve.Translate(0,0,-1)
            
            # Generate a mesh splitter (extra unit on -z side)
            splitter = rg.Mesh.CreateFromCurveExtrusion(temp_curve, rg.Vector3d(0,0,1), params, bbox)
            
            # Add the splitter to all splitters for this specific building
            temp_splitters.append(splitter)
        
        # Add the building splitters to all building splitters
        splitters.append(temp_splitters)
    
    # Store the splitters in a list
    roofs = []

    # Iterate over the splitters and buildings
    for sample, (splitter_set, building_curve_set, courtyard_curve_set) in enumerate(zip(splitters, building_curves, courtyard_curves)):
        
        # Store the roof meshes for a single building in a temp mesh
        temp_roofs = rg.Mesh()
        
        # Iterate over the building outlines
        for splitter, building_curve in zip(splitter_set, building_curve_set):
            success = False
            
            # Split the mesh
            # ----- WARNING: VERY TIME CONSUMING! -----
            elements = mesh_plane.Split(splitter)
            
            # Postprocess the mesh elements
            elements = [postprocess_mesh(element) for element in elements]
            
            # If more than one splitting elements returned
            if len(elements) > 1:
                success = True
                
                # Check if the element is inside the building or outside (part of the ground)
                relations = [is_inside(element, [building_curve]) for element in elements]
                                
                # Generate a roof mesh
                roof = rg.Mesh()
                
                # Generate a ground elements mesh
                ground_elements = rg.Mesh()
                
                # Iterate over the splitted elements and the relations
                for element, relation in zip(elements, relations):
                    # If the element is inside the roof outline
                    if relation:
                        # Add the element to the roof mesh
                        roof.Append(element)
                    else:
                        # Add the element to the ground mesh
                        ground_elements.Append(element)
                    
                # Add the roof mesh to the temp roofs for this building
                temp_roofs.Append(roof)
                
                # Overwrite the mesh plane to the ground elements
                mesh_plane = ground_elements
            else:
                if logger:
                    logger.warning("Splitting did not result in multiple elements")
        
        # Check if this building has courtyards
        if len(courtyard_curve_set) > 0 and success:
            # Iterate over the polylines in the courtyards
            for i, courtyard_curve in enumerate(courtyard_curve_set):
                # Generate a splitter for the courtyard
                splitter = rg.Mesh.CreateFromCurveExtrusion(courtyard_curve, rg.Vector3d(0,0,2), params, bbox)
                
                # Split the roofs of this building by the courtyard splitter
                elements = temp_roofs.Split(splitter)

                # If the splitting resulted in more than one mesh
                if len(elements) > 1:
                    # Check if the element is inside the courtyard or outside (part of the roof)     
                    relations = [is_inside(element, [courtyard_curve]) for element in elements]
                    
                    # Generate a new roof mesh
                    roof = rg.Mesh()
                    
                    # Store the courtyard elements
                    courtyard_elements = rg.Mesh()
                    
                    # Iterate over the splitted elements from the roof
                    for element, relation in zip(elements, relations):
                        # If the roof is inside the courtyard
                        if relation:
                            # Add to the courtyard elements
                            courtyard_elements.Append(element)
                        else:
                            # Add to the roof elements
                            roof.Append(element)
                            temp_roofs = element
                    
                    # Add the courtyard elements to the mesh plane
                    mesh_plane.Append(courtyard_elements)
                else:
                    if logger:
                        logger.warning("Splitting did not result in multiple elements")
        
        if success:
            # Add the roof to the list of roofs
            roofs.append(roof)
            valid.append(True)
        else:
           valid.append(False)
   
    # Postprocess the ground mesh plane mesh
    ground = postprocess_mesh(mesh_plane)
    
    heights = [height for i, height in enumerate(heights) if valid[i]]

    # Iterate over the roof meshes to translate to the correct height
    for i, (mesh, height) in enumerate(zip(roofs, heights)):
        # Generate a duplicate translated mesh
        translated_mesh = mesh.Duplicate()
        
        # Create a transform based on the height
        transform = rg.Transform.Translation(System.Double(0.0),System.Double(0.0),System.Double(height))
        
        # Move the mesh
        translated_mesh.Transform(transform)
        
        # Set the translated mesh after postprocessing
        roofs[i] = postprocess_mesh(translated_mesh)
    
    for i, roof in enumerate(roofs):
        if len(roof.Vertices) < 3:
            valid[i] = False
        elif len(roof.Faces) == 0:
            valid[i] = False

    invalid_idxs = []
    for i, val in enumerate(valid):
        if not val:
            invalid_idxs.append(i)

    return ground, roofs, invalid_idxs

'''
REMESHING FUNCTIONS
'''

def remesh_horizontal(mesh):
    """Remesh horizontal mesh elements

    Args:
        mesh (rg.Mesh): horizontal mesh

    Returns:
        rough_mesh (rg.Mesh): reduced
    """
    
    rough_mesh = mesh.Duplicate()
    
    rough_mesh.Reduce(100, False, 3, False, False)

    return rough_mesh

def remesh_vertical(curve, height):
    """Mesh vertical elements by using outlines and height

    Args:
        outlines (list[rg.Polyline]): building outlines
        height (list[float]): heights of the buildings
    Returns:
        mesh (rg.Mesh): reduced mesh
    """
    
    extrusion = rg.Extrusion.Create(curve, height, False)
    
    params = rg.MeshingParameters.QualityRenderMesh
    
    return rg.Mesh.CreateFromSurface(extrusion, params)

def triangulate_quad(quad_mesh):
    tri_mesh = quad_mesh.Duplicate()
    tri_mesh.Faces.ConvertQuadsToTriangles()
    return tri_mesh

def generate_mesh(patch_outline, building_outlines, courtyard_outlines, building_heights, grid_size, size, rough=False, logger=False):
    """Generate a patch mesh based on a patch outline, building polylines and courtyard outlines

    Args:
        patch_outline (rg.Rectangle3d): ground patch outline
        building_outlines (list[list[rg.Polyline]]): building outline polylines
        courtyard_outlines (list[list[rg.Polyline]]): courtyard outline polylines
        building_heights (list[float]): heights per building
        grid_size (float): grid size
        size (float): size of patch
        rough (bool, optional): Indicate if function should also return rough meshes. Defaults to False.

    Returns:
        mesh_plane (rg.Mesh): ground mesh plane
        walls (list[rg.Mesh]): walls
        roofs (list[rg.Mesh]): roofs
        rough_ground (rg.Mesh, optional), rough ground mesh
        rough_walls (list[rg.Mesh], optional): rough walls
        rough_roofs (list(rg.Mesh), optional): rough roofs
    """
    
    building_polylines, building_curves = project_outlines_to_world_xy(building_outlines)
    courtyard_polylines, courtyard_curves = project_outlines_to_world_xy(courtyard_outlines)
    
    if logger:
        logger.info(f'Generating roofs and ground for mesh patch')
    
#    ground_mesh, building_meshes, building_polylines, courtyard_polylines, all_heights = _generate_horizontal(patch_outline, building_polylines, courtyard_polylines, building_heights, grid_size, size)
    mesh_plane, roofs, valid = generate_horizontal(patch_outline, building_polylines, courtyard_polylines, building_curves, courtyard_curves, building_heights, grid_size, size)
    
    # Overwite invalid outlines and heights
    building_polylines = [i for j, i in enumerate(building_polylines) if j not in valid]
    building_curves = [i for j, i in enumerate(building_curves) if j not in valid]
    courtyard_polylines = [i for j, i in enumerate(courtyard_polylines) if j not in valid]
    courtyard_curves = [i for j, i in enumerate(courtyard_curves) if j not in valid]
    building_heights = [i for j, i in enumerate(building_heights) if j not in valid]
    
    if logger:
        logger.info(f'Generating walls for mesh patch')
    
    # Generate the walls for the building outlines and compute corresponding heights
    # Requires outlines in format polylines
    walls = generate_vertical(building_polylines, courtyard_polylines, building_heights, grid_size)
    
    # Compute the mesh plane for the ground and roofs
    # Requires outlines in format curves    
    if rough:
        if logger:
            logger.info(f'Generating rough meshes')
            
        rough_ground = remesh_horizontal(mesh_plane)
        rough_roofs = [remesh_horizontal(roof) for roof in roofs]
        
        rough_walls = []
        for building, courtyard, height in zip(building_curves, courtyard_curves, building_heights):
            mesh = rg.Mesh()
            
            for curve in building:
                mesh.Append(remesh_vertical(curve, height))
            
            for curve in courtyard:
                mesh.Append(remesh_vertical(curve, height))
                
            rough_walls.append(mesh)
        
        return mesh_plane, walls, roofs, rough_ground, rough_walls, rough_roofs
    else:
        return mesh_plane, walls, roofs, None, None, None