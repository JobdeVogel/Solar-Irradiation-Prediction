from parameters.params import LOGGER
import Rhino.Geometry as rg
import System
import math
import sys
from parameters.params import _REDUCE_SEGMENTS_TOLERANCE, _MESH_SPLITTER_BBOX_HEIGHT, _ANGLE_TOLERANCE_POSTP_MESH, _DIAGONAL_LENGTH_RATIO_POSTP_MESH
#import cProfile
  
def postprocess_mesh(mesh, check=False):
    mesh.Faces.ConvertTrianglesToQuads(_ANGLE_TOLERANCE_POSTP_MESH, _DIAGONAL_LENGTH_RATIO_POSTP_MESH)
        
    vertices = mesh.Vertices
    faces = mesh.Faces
    
    rebuild_mesh = rg.Mesh()

    rebuild_mesh.Vertices.AddVertices(vertices)
    rebuild_mesh.Faces.AddFaces(faces)
    
    if check:
        print(f'Adding vertices and faces: {rebuild_mesh.IsValidWithLog()}')
    
    rebuild_mesh.Normals.ComputeNormals()
    
    if check:
        print(f'Computing normals: {rebuild_mesh.IsValidWithLog()}')
    
    rebuild_mesh.Compact()
    
    if check:
        print(f'Compact: {rebuild_mesh.IsValidWithLog()}')
        
    rebuild_mesh.Faces.CullDegenerateFaces()
    
    if check:
        print(f'Cull degenerate faces: {rebuild_mesh.IsValidWithLog()}')
        sys.exit()
    
    return rebuild_mesh

def is_inside(mesh, polylines):
    # TOLERANCE PROBLEMS!!
    checkpoint = None

    vertices = mesh.Faces.GetFaceVertices(0)[1:]

    center_x = sum(p.X for p in vertices) / len(vertices)
    center_y = sum(p.Y for p in vertices) / len(vertices)
    center_z = sum(p.Z for p in vertices) / len(vertices)

    checkpoint = rg.Point3d(center_x, center_y, center_z)    
    
    inside = False
    for polyline in polylines:
        if polyline.Contains(checkpoint, rg.Plane.WorldXY, tolerance=1e-8) == rg.PointContainment.Inside:
            inside = True
        elif polyline.Contains(checkpoint, rg.Plane.WorldXY, tolerance=1e-8) == rg.PointContainment.Coincident:
            inside = True
            LOGGER.warning('Point containment coincident')

    return inside

def triangulate_quad(quad_mesh):
    tri_mesh = quad_mesh.Duplicate()
    tri_mesh.Faces.ConvertQuadsToTriangles()
    return tri_mesh

def remesh_rough(mesh):
    rough_mesh = mesh.Duplicate()
    
    rough_mesh.Reduce(100, False, 3, False, False)

    return rough_mesh

def generate_horizontal(ground_outline, building_outlines, courtyard_outlines, heights, grid_size):
    # Generate plane parameters
    box_interval = rg.Box(ground_outline.BoundingBox)
    plane_width = rg.Interval(box_interval.X[0], box_interval.X[1])
    plane_height = rg.Interval(box_interval.Y[0], box_interval.Y[1])
    plane = rg.Plane.WorldXY
    
    copy_building_outlines = []
    copy_courtyard_outlines = []
    for buildings, courtyards in zip(building_outlines, courtyard_outlines):
        temp = []
        for polyline in buildings:
            temp.append(polyline.Duplicate())
        copy_building_outlines.append(temp)
        
        temp = []
        for polyline in courtyards:
            temp.append(polyline.Duplicate())
        copy_courtyard_outlines.append(temp)
            
    
    width_divisions = System.Int32(int(plane_width.Length / grid_size))
    length_divisions = System.Int32(int(plane_height.Length / grid_size))
    
    # Generate the mesh plane
    mesh_plane = rg.Mesh.CreateFromPlane(plane, plane_width, plane_height, width_divisions, length_divisions)   
    
    # Project the building outlines on the mesh plane
    for i, outlines in enumerate(building_outlines):
        for j, polyline in enumerate(outlines):
            curve = polyline.ToNurbsCurve()
            
            projected_curve = curve.ProjectToPlane(curve, rg.Plane.WorldXY)
            
            # tolerance = System.Double(0.001)
            # projected_curve = rg.Curve.ProjectToMesh(curve, mesh_plane, rg.Vector3d(0,0,-1), tolerance)            
            
            building_outlines[i][j] = projected_curve
    
    # Project the courtyard outlines on the mesh plane
    for i, outlines in enumerate(courtyard_outlines):
        for j, polyline in enumerate(outlines):
            curve = polyline.ToNurbsCurve()
            # projected_curve = rg.Curve.ProjectToMesh(curve, mesh_plane, rg.Vector3d(0,0,-1), 0.001)
            
            projected_curve = curve.ProjectToPlane(curve, rg.Plane.WorldXY)
            
            courtyard_outlines[i][j] = projected_curve
    
    # Generate the splitters from the inner and outer polylines
    params = rg.MeshingParameters.QualityRenderMesh
    bbox = rg.BoundingBox(0,0,-_MESH_SPLITTER_BBOX_HEIGHT,100,100,_MESH_SPLITTER_BBOX_HEIGHT)
    
    splitters = []
    for outlines in building_outlines:
        temp_splitters = []
        
        for curve in outlines:
            template = curve.Duplicate()
                   
            template.Translate(0,0,-1)
            polyline = template.TryGetPolyline()[1]          
            
            polyline.ReduceSegments(_REDUCE_SEGMENTS_TOLERANCE)
            splitter = rg.Mesh.CreateFromCurveExtrusion(polyline.ToNurbsCurve(), rg.Vector3d(0,0,2), params, bbox)
            temp_splitters.append(splitter)
        
        splitters.append(temp_splitters)
    
    roofs = []
    
    # Split the mesh plane
    for splitter_set, outlines, courtyards in zip(splitters, building_outlines, courtyard_outlines):
        temp_roofs = rg.Mesh()
        
        # For each building outline
        for splitter, outline in zip(splitter_set, outlines):
            elements = mesh_plane.Split(splitter)
            elements = [postprocess_mesh(element) for element in elements]
            
            if len(elements) > 1:        
                relations = [is_inside(element, [outline]) for element in elements]               
                
                # For each splitted element
                roof = rg.Mesh()
                ground_elements = rg.Mesh()
                for element, relation in zip(elements, relations):
                    if relation:
                        roof.Append(element)
                    else:
                        ground_elements.Append(element)
                
                temp_roofs.Append(roof)
                
                mesh_plane = ground_elements

        if len(courtyards) > 0:
            for polyline in courtyards:
                splitter = rg.Mesh.CreateFromCurveExtrusion(polyline.ToNurbsCurve(), rg.Vector3d(0,0,2), params, bbox)
                
                elements = temp_roofs.Split(splitter)

                if len(elements) > 1:        
                    relations = [is_inside(element, [polyline]) for element in elements]

                    # For each splitted element
                    roof = rg.Mesh()
                    courtyard_elements = rg.Mesh()
                    for element, relation in zip(elements, relations):
                        if relation:
                            courtyard_elements.Append(element)
                        else:
                            roof.Append(element)
                    
                    mesh_plane.Append(courtyard_elements)
                    temp_roofs = roof
                        
        roofs.append(temp_roofs)
    
        # translated_mesh = rg.Mesh()
        # for vertex in mesh_plane.Vertices:
        #     translated_mesh.Vertices.Add(vertex.X, vertex.Y, 0.0)
        
        # translated_mesh.Faces.AddFaces(mesh_plane.Faces)
        
    ground = postprocess_mesh(mesh_plane)
               
    for i, (mesh, height) in enumerate(zip(roofs, heights)):
        translated_mesh = mesh.Duplicate()
        
        transform = rg.Transform.Translation(System.Double(0.0),System.Double(0.0),System.Double(height))
        translated_mesh.Transform(transform)
        
        roofs[i] = postprocess_mesh(translated_mesh)
    
    return ground, roofs

def compute_area(ground, roofs):
    ground_area = rg.AreaMassProperties.Compute(ground).Area
    building_area = sum([rg.AreaMassProperties.Compute(building).Area for building in roofs])
    
    return ground_area, building_area




def polyline_isclockwise(polyline):
    curve = polyline.Duplicate().ToNurbsCurve()
    return curve.ClosedCurveOrientation() == rg.CurveOrientation.Clockwise

def flip_polyline(polyline):
    flipped_polyline = rg.Polyline()
    
    for point in polyline.ToArray()[::-1]:
        flipped_polyline.Add(point.X, point.Y, point.Z)
    
    return flipped_polyline

def mesh_extrude_polyline(polyline, height, grid_size):
    vertices = []
    faces = []
    
    polyline.ReduceSegments(_REDUCE_SEGMENTS_TOLERANCE)
    segments = polyline.GetSegments()
    lengths = [segment.Length for segment in segments]
    
    num_segments = [int(math.ceil(length / grid_size)) for length in lengths]
    
    levels = int(math.ceil(height / grid_size))
    num_p = sum(num_segments)
    
    base_vertices = []
    vertices = []
    faces = []
    
    rough_mesh = rg.Mesh()
    
    for segment, num_segment in zip(segments, num_segments):
        
        for i in range(num_segment):
            parameter = i * (1 / num_segment)
            
            base_vertices.append(segment.PointAt(parameter))
    
    for j in range(levels + 1):
        parameter = j * (height / levels)
        
        for vertex in base_vertices:
            vertices.append(rg.Point3d(vertex.X, vertex.Y, vertex.Z + parameter))
    
    faces = System.Array[rg.MeshFace]([
        rg.MeshFace(i + j * num_p, i + (j + 1) * num_p, (i + 1) % num_p + (j + 1) * num_p, (i + 1) % num_p + j * num_p)
        for j in range(levels)
        for i in range(num_p)
    ])
    
    mesh = rg.Mesh()
    for vertex in vertices:
        mesh.Vertices.Add(vertex)
    
    mesh.Faces.AddFaces(faces)
    
    return mesh

def generate_vertical(building_outlines, courtyard_outlines, heights, grid_size):
    meshes = []
    outlines = []
    wall_heights = []

    for building, courtyard, height in zip(building_outlines, courtyard_outlines, heights):
        temp_mesh = rg.Mesh()
        
        for outline in building:
            if not polyline_isclockwise(outline):
                outline.Reverse()
            
            mesh = mesh_extrude_polyline(outline, height, grid_size)
            temp_mesh.Append(mesh)
            outlines.append(outline)
            wall_heights.append(height)
        
        for outline in courtyard:
            if polyline_isclockwise(outline):
                outline.Reverse()
            
            mesh = mesh_extrude_polyline(outline, height, grid_size)
            temp_mesh.Append(mesh)
            outlines.append(outline)
            wall_heights.append(height)
        
        meshes.append(postprocess_mesh(temp_mesh))

    LOGGER.debug(f'Generated {len(meshes)} meshes and outlines.')
    return meshes, outlines, wall_heights

def triangulate_quad(quad_mesh):
    tri_mesh = quad_mesh.Duplicate()
    tri_mesh.Faces.ConvertQuadsToTriangles()
    return tri_mesh

def remesh_horizontal(mesh):
    rough_mesh = mesh.Duplicate()
    
    rough_mesh.Reduce(100, False, 3, False, False)

    return rough_mesh

def remesh_vertical(outlines, height):
    params = rg.MeshingParameters.QualityRenderMesh
    bbox = rg.BoundingBox(0,0,1,100,100,height-1)
    
    template = outlines.Duplicate()
    template.ReduceSegments(_REDUCE_SEGMENTS_TOLERANCE)
    mesh = rg.Mesh.CreateFromCurveExtrusion(template.ToNurbsCurve(), rg.Vector3d(0,0,1), params, bbox)
        
    return mesh

