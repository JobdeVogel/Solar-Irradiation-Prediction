import Rhino.Geometry as rg
import System
import time

from ladybug.graphic import GraphicContainer
from ladybug_rhino.togeometry import to_mesh3d
from ladybug_rhino.fromgeometry import from_mesh3d
from ladybug_rhino.color import color_to_color

from ladybug.legend import LegendParameters
from ladybug_rhino.togeometry import to_plane

def legend(colors=[(134, 144, 252), (247, 228, 10), (189, 88, 51), (212, 49, 0)], seg_count=20):
    colors = [System.Drawing.Color.FromArgb(System.Int32(color[0]), System.Int32(color[1]), System.Int32(color[2])) for color in colors]

    base_plane_ = to_plane(rg.Plane.WorldXY)

    leg_par = LegendParameters(segment_count=seg_count, colors=colors, base_plane=base_plane_)

    leg_par.continuous_legend = True
    leg_par.decimal_count = 0
    
    return leg_par
    
def generate_colored_mesh(mesh, values, legend_par, legend_title='(kWh/m2)', title='Solar Irradiance Simulation'):
    # generate Ladybug objects
    lb_mesh = to_mesh3d(mesh)
    
    graphic = GraphicContainer(values, lb_mesh.min, lb_mesh.max, legend_par)

    # generate titles
    graphic.legend_parameters.title = legend_title

    # draw rhino objects # ! Most time consuming!
    lb_mesh.colors = graphic.value_colors
    mesh = from_mesh3d(lb_mesh)
    
    return mesh
    
    
    
    