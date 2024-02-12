import Rhino.Geometry as rg

def generate(roof_meshes):
    heights = []

    for mesh in roof_meshes:
        heights.append(mesh.Faces.GetFaceCenter(0).Z)
    
    return heights