from parameters.params import LOGGER
import Rhino.Geometry as rg
import System

from parameters.params import _FACE_MERGE_TOLERANCE

def load_obj(path):
    with open(path) as file:
        lines = file.readlines()

    vertices=[]
    faces=[]
    
    for line in lines:
        if line.find("v")==0 and line.find("n")==-1 and line.find("t")==-1:
            vertex = rg.Point3d(float((line.split(' '))[1]),float((line.split(' '))[2]),float((line.split(' '))[3]))
            vertices.append(vertex)
            
        if line.find("f")==0:
            if len(line.split(' '))==4:
                faces.append(rg.MeshFace(int(line.split(' ')[1].split('/')[0])-1,int(line.split(' ')[2].split('/')[0])-1,int(line.split(' ')[3].split('/')[0])-1))
            if len(line.split(' '))==5:
                faces.append(rg.MeshFace(int(line.split(' ')[1].split('/')[0])-1,int(line.split(' ')[2].split('/')[0])-1,int(line.split(' ')[3].split('/')[0])-1,int(line.split(' ')[4].split('/')[0])-1))
    
    mesh = rg.Mesh()
    for vertex in vertices:
        mesh.Vertices.Add(vertex)
    
    for face in faces:
        mesh.Faces.AddFace(face)
    
    mesh.Normals.ComputeNormals()
    mesh.Compact()
    mesh.MergeAllCoplanarFaces(_FACE_MERGE_TOLERANCE)
    
    return mesh

def translate_mesh(mesh, bbox):
    center = bbox.Center
    x, y = center[0], center[1]

    mesh.Translate(-x, -y, 0)
    
    translation = rg.Transform.Translation(-x, -y, 0)
    bbox.Transform(translation)
    
    return mesh, bbox

def get_bbox(mesh):
    xyPlane = rg.Plane(rg.Point3d(0.0, 0.0, 0.0), rg.Vector3d(0.0, 0.0, 1.0))
    boundingBox = mesh.GetBoundingBox(xyPlane)
    
    return boundingBox

def partition_mesh(mesh):
    roofMesh = rg.Mesh()
    roofMesh.CopyFrom(mesh)
    roofFacesIdxs = []
    
    wallMesh = rg.Mesh()
    wallMesh.CopyFrom(mesh)
    wallFacesIdxs = []
    
    floorMesh = rg.Mesh()
    floorMesh.CopyFrom(mesh)
    floorFacesIdxs = []
    
    for i, normal in enumerate(mesh.FaceNormals):
        if normal.Z > 0:
            roofFacesIdxs.append(i)
        elif normal.Z == 0:
            wallFacesIdxs.append(i)
        else:
            floorFacesIdxs.append(i)
    
    valid = True
    if len(floorFacesIdxs) == 0 or len(wallFacesIdxs) == 0 or len(roofFacesIdxs) == 0:
        valid = False
        LOGGER.warning("File contains meshes without roofs, walls or floors.")
        
        return valid, roofMesh, wallMesh, floorMesh
    
    
    roof_idxs = System.Array[System.Int32](wallFacesIdxs + floorFacesIdxs)
    wall_idxs = System.Array[System.Int32](roofFacesIdxs + floorFacesIdxs)
    floor_idxs = System.Array[System.Int32](roofFacesIdxs + wallFacesIdxs)
    
    roofMesh.Faces.DeleteFaces(roof_idxs)
    wallMesh.Faces.DeleteFaces(wall_idxs)
    floorMesh.Faces.DeleteFaces(floor_idxs)
    
    roofMesh.Compact()
    wallMesh.Compact()
    floorMesh.Compact()
    
    return valid, roofMesh, wallMesh, floorMesh

def level_mesh(mesh, roof_mesh, wall_mesh, floor_mesh):
    height = floor_mesh.Faces.GetFaceCenter(0).Z
    
    mesh.Translate(0, 0, -height)
    roof_mesh.Translate(0, 0, -height)
    wall_mesh.Translate(0, 0, -height)
    floor_mesh.Translate(0, 0, -height)
    
    return mesh, roof_mesh, wall_mesh, floor_mesh

def load(path):
    mesh = load_obj(path)
    
    mesh, bbox = translate_mesh(mesh, get_bbox(mesh))
    
    global_mesh = rg.Mesh()

    local_roof_meshes = []
    local_wall_meshes = []
    
    for i, local_mesh in enumerate(mesh.SplitDisjointPieces()):
        valid, roof_mesh, wall_mesh, floor_mesh = partition_mesh(local_mesh)
        
        if not valid:
            continue
        
        local_mesh, roof_mesh, wall_mesh, floor_mesh = level_mesh(local_mesh, roof_mesh, wall_mesh, floor_mesh)
        global_mesh.Append(local_mesh)
        
        local_roof_meshes.append(roof_mesh)
        local_wall_meshes.append(wall_mesh)

    LOGGER.info(f"Loaded 3D BAG dataset with {len(local_roof_meshes)} roofs and {len(local_wall_meshes)} walls from {path}.")

    return local_roof_meshes, local_wall_meshes, bbox
