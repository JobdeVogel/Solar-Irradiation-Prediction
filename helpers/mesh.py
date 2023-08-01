import Rhino.Geometry as rg
from parameters.params import LOGGER

def join_meshes(meshes):
    joined_meshes = rg.Mesh()
    for mesh in meshes:
        joined_meshes.Append(mesh)
    
    return joined_meshes

def get_face_vertices(face, mesh):
    # vertex_0 = rg.Point3d(mesh.Vertices[face.A].X, mesh.Vertices[face.A].Y, mesh.Vertices[face.A].Z)
    # vertex_1 = rg.Point3d(mesh.Vertices[face.B].X, mesh.Vertices[face.B].Y, mesh.Vertices[face.B].Z)
    # vertex_2 = rg.Point3d(mesh.Vertices[face.C].X, mesh.Vertices[face.C].Y, mesh.Vertices[face.C].Z)
    # vertex_3 = rg.Point3d(mesh.Vertices[face.D].X, mesh.Vertices[face.D].Y, mesh.Vertices[face.D].Z)
    
    vertices = [rg.Point3d(mesh.Vertices[vertex].X, mesh.Vertices[vertex].Y, mesh.Vertices[vertex].Z)
                for vertex in (face.A, face.B, face.C, face.D)]
    
    if face.IsQuad:
        return vertices[:4]
    else:
        return vertices[:3]