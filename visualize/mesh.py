import Rhino.Geometry as rg
import numpy as np
# import open3d

from ladybug.graphic import GraphicContainer
from ladybug_rhino.togeometry import to_mesh3d
from ladybug_rhino.fromgeometry import from_mesh3d
from ladybug_rhino.fromobjects import legend_objects
from ladybug_rhino.text import text_objects
from ladybug_rhino.color import color_to_color

def generate(legend_par):
    # generate Ladybug objects
    lb_mesh = to_mesh3d(_mesh)
    if offset_dom_:
        dom_st, dom_end = offset_dom_
        lb_mesh = lb_mesh.height_field_mesh(_values, (dom_st, dom_end))
    graphic = GraphicContainer(_values, lb_mesh.min, lb_mesh.max, legend_par_)

    # generate titles
    if legend_title_ is not None:
        graphic.legend_parameters.title = legend_title_
    if global_title_ is not None:
        title = text_objects(global_title_, graphic.lower_title_location,
                             graphic.legend_parameters.text_height * 1.5,
                             graphic.legend_parameters.font)

    # draw rhino objects
    lb_mesh.colors = graphic.value_colors
    mesh = from_mesh3d(lb_mesh)
    legend = legend_objects(graphic.legend)
    colors = [color_to_color(col) for col in lb_mesh.colors]
    legend_par = graphic.legend_parameters



# def show(meshes):
    
#     meshes_to_visualize = []
#     for mesh in meshes:   
#         vertices = np.array([[vertex.X, vertex.Y, vertex.Z] for vertex in mesh.Vertices])
#         faces = np.array([[face.A, face.B, face.C] for face in mesh.Faces]).astype(np.int32)
        
#         mesh = open3d.geometry.TriangleMesh()
#         mesh.vertices = open3d.utility.Vector3dVector(vertices)
#         mesh.triangles = open3d.utility.Vector3iVector(faces)
        
#         # mesh.triangle.colors = ''
#         mesh.compute_vertex_normals()
        
#         meshes_to_visualize.append(mesh)
    
#     open3d.visualization.draw_geometries(meshes_to_visualize)
    
    
    
    