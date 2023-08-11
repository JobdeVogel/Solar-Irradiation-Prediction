from parameters.params import LOGGER
import Rhino.Geometry as rg
import System

from parameters.params import _FACE_MERGE_TOLERANCE

def load_obj(path):
    """Load a 3D BAG LoD 1.2 file

    Args:
        path (str): path to the .obj file

    Returns:
        mesh (rg.Mesh): mesh from the .obj file
    """
    
    # Open the file
    with open(path) as file:
        lines = file.readlines()

    # Store the vertices and faces in lists
    vertices=[]
    faces=[]
    
    # Iterate over file lines
    for line in lines:
        # Check if line is a vertex
        if line.find("v")==0 and line.find("n")==-1 and line.find("t")==-1:
            vertex = rg.Point3d(float((line.split(' '))[1]),float((line.split(' '))[2]),float((line.split(' '))[3]))
            vertices.append(vertex)
        
        # Check if line is a face
        if line.find("f")==0:
            if len(line.split(' '))==4:
                faces.append(rg.MeshFace(int(line.split(' ')[1].split('/')[0])-1,int(line.split(' ')[2].split('/')[0])-1,int(line.split(' ')[3].split('/')[0])-1))
            if len(line.split(' '))==5:
                faces.append(rg.MeshFace(int(line.split(' ')[1].split('/')[0])-1,int(line.split(' ')[2].split('/')[0])-1,int(line.split(' ')[3].split('/')[0])-1,int(line.split(' ')[4].split('/')[0])-1))
    
    # Initialize a mesh
    mesh = rg.Mesh()

    # Append the vertices and faces to the mesh
    mesh.Vertices.AddVertices(System.Array[rg.Point3d](vertices))
    mesh.Faces.AddFaces(System.Array[rg.MeshFace](faces))
    
    # Conpute the normals, make compact, merge faces
    mesh.Normals.ComputeNormals()
    mesh.Compact()
    mesh.MergeAllCoplanarFaces(_FACE_MERGE_TOLERANCE)
    
    return mesh

def translate_mesh(mesh, bbox):
    """Translate a mesh to the world origin 0,0,0

    Args:
        mesh (rg.Mesh): mesh
        bbox (rg.BoundingBox): boundingbox around the mesh

    Returns:
        mesh (rg.Mesh): translated mesh
        bbox (rg.BoundingBox): translateed boundingbox
    """
    # Compute the x and y coordinates of the original mesh center
    center = bbox.Center
    x, y = center[0], center[1]

    # Translate the mesh to the origin
    mesh.Translate(-x, -y, 0)
    
    # Translate the boundingbox to the origin
    translation = rg.Transform.Translation(-x, -y, 0)
    bbox.Transform(translation)
    
    return mesh, bbox

def get_bbox(mesh):
    """Compute the boundingbox around a mesh

    Args:
        mesh (rg.Mesh): mesh

    Returns:
        boundingbox (rg.BoundingBox): the boundingbox around the mesh
    """
    
    xy_plane = rg.Plane(rg.Point3d(0.0, 0.0, 0.0), rg.Vector3d(0.0, 0.0, 1.0))
    boundingbox = mesh.GetBoundingBox(xy_plane)
    
    return boundingbox

def partition_mesh(mesh):
    """Partition a building mesh in floors, walls and roofs

    Args:
        mesh (rg.Mesh): building mesh

    Returns:
        valid (bool): indicates if partition succesfull
        roof_mesh (rg.Mesh): roof mesh of the building
        wall_mesh (rg.Mesh): wall mesh of the building
        floor_mesh (rg.Mesh): floor mesh of the building
    """
    
    # Transform the mesh to compact shape
    mesh.Compact()
    
    # Duplicate mesh to new roof, wall and floor mesh
    roof_mesh = mesh.Duplicate()
    roof_faces_idxs = []
    
    wall_mesh = mesh.Duplicate()
    wall_faces_idxs = []
    
    floor_mesh = mesh.Duplicate()
    floor_faces_idxs = []
    
    # Check the normal direction of the mesh faces
    for i, normal in enumerate(mesh.FaceNormals):
        # Append the face to the corresponding idx list based on normal direction
        if normal.Z > 0:
            roof_faces_idxs.append(i)
        elif normal.Z == 0:
            wall_faces_idxs.append(i)
        else:
            floor_faces_idxs.append(i)
    
    # Check if the floor, wall and roof has enough faces to be valid
    valid = True
    if len(floor_faces_idxs) == 0 or len(wall_faces_idxs) == 0 or len(roof_faces_idxs) == 0:
        valid = False
        LOGGER.warning("File contains meshes without roofs, walls or floors.")
        
        return valid, roof_mesh, wall_mesh, floor_mesh
    
    # Transform indices to Sytem arrays
    roof_idxs = System.Array[System.Int32](wall_faces_idxs + floor_faces_idxs)
    wall_idxs = System.Array[System.Int32](roof_faces_idxs + floor_faces_idxs)
    floor_idxs = System.Array[System.Int32](roof_faces_idxs + wall_faces_idxs)
    
    # Delete the faces in roof, wall and floor based on index
    roof_mesh.Faces.DeleteFaces(roof_idxs)
    wall_mesh.Faces.DeleteFaces(wall_idxs)
    floor_mesh.Faces.DeleteFaces(floor_idxs)
    
    return valid, roof_mesh, wall_mesh, floor_mesh

def level_mesh(roof_mesh, wall_mesh, floor_mesh):
    """Level the mesh to height z = 0

    Args:
        roof_mesh (rg.Mesh): roof mesh of single building
        wall_mesh (rg.Mesh): wall mesh of single building
        floor_mesh (rg.Mesh): floor mesh of single building

    Returns:
        roof_mesh (rg.Mesh): leveled roof mesh of single building
        wall_mesh (rg.Mesh): leveled wall mesh of single building
        floor_mesh (rg.Mesh): leveled floor mesh of single building
    """
    
    # Get the height of the building based on the first face in the floor mesh
    height = floor_mesh.Faces.GetFaceCenter(0).Z
    
    # Move the meshes to the correct height
    roof_mesh.Translate(0, 0, -height)
    wall_mesh.Translate(0, 0, -height)
    floor_mesh.Translate(0, 0, -height)
    
    return roof_mesh, wall_mesh, floor_mesh

def load(path):
    """Load a 3D BAG .obj file LoD 1.2 to Rhino roof and wall meshes

    Args:
        path (str): Path to the .obj file

    Returns:
        roof_meshes (list[rg.Mesh]): roof meshes
        wall_meshes (list[rg.Mesh]): wall meshes
        bbox (rg.BoundingBox): Boundingbox around building geometry
    """
    
    # Load the mesh from the .obj path
    mesh = load_obj(path)
    
    # Translate the mesh to the origin using a bounding box
    mesh, bbox = translate_mesh(mesh, get_bbox(mesh))

    # Store the roof an wall meshes in lists
    roof_meshes = []
    wall_meshes = []
    
    for mesh in mesh.SplitDisjointPieces():
        # Partition the mesh in roofs, walls and floors, check if splitting is valid
        valid, roof_mesh, wall_mesh, floor_mesh = partition_mesh(mesh)
        
        # Go to next iteration, skip this mesh
        if not valid:
            continue
        
        # Move the meshes to ground level z = 0
        roof_mesh, wall_mesh, floor_mesh = level_mesh(roof_mesh, wall_mesh, floor_mesh)
        
        roof_meshes.append(roof_mesh)
        wall_meshes.append(wall_mesh)

    LOGGER.info(f"Loaded 3D BAG dataset with {len(roof_meshes)} roofs and {len(wall_meshes)} walls from {path}.")

    return roof_meshes, wall_meshes, bbox