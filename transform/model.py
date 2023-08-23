from parameters.params import LOGGER
import Rhino.Geometry as rg

import uuid

try:
    from honeybee.face import Face
    from honeybee.facetype import face_types
    from honeybee.boundarycondition import boundary_conditions
    from honeybee.typing import clean_string, clean_and_id_string, clean_and_id_rad_string, clean_rad_string

    # Ladybug-Rhino dependencies
    from ladybug_rhino.togeometry import to_mesh3d, to_face3d
    # from ladybug_rhino.grasshopper import longest_list
    from ladybug_rhino.config import units_system, tolerance, angle_tolerance

    # Honeybee Core dependencies
    from honeybee.model import Model

    # Honeybee-Radiance dependencies
    from honeybee_radiance.lib.modifiers import modifier_by_identifier
    from honeybee_radiance.modifier.material import Plastic
    from honeybee_radiance.sensorgrid import SensorGrid
except:
    logging.error('Not able to import honeybee or ladybug packages.')


import time

from parameters.params import REFLECTANCE, SPECULAR, ROUGHNESS, MODIFIER_NAME

def modifier(reflectance=REFLECTANCE, specular=SPECULAR, roughness=ROUGHNESS, name=MODIFIER_NAME):
    # Set the default modifier properties
    specular = 0.0 if specular is None else specular
    roughness = 0.0 if roughness is None else roughness
    name = clean_and_id_rad_string('OpaqueMaterial') if name is None else \
        clean_rad_string(name)

    # Create the modifier
    modifier = Plastic.from_single_reflectance(name, reflectance, specular, roughness)
    modifier.display_name = name
    
    return modifier

def generate_HB_faces(mesh, modifier, name="faces", boundary_condition="outdoors"):
    faces = []
    for j, geo in enumerate(mesh):

        # Assign Name
        # display_name = '{}_{}'.format(longest_list(name, j), j + 1)
        display_name = 'temp'
        name = clean_and_id_string(display_name)
        
        # Type is not used in this context
        typ = None
        
        # Assign boundary condition
        bc = boundary_conditions.by_name(boundary_condition)
        
        # Generate LB faces
        
        lb_faces = to_face3d(geo)   # OPTIMIZE SPEED
        for i, lb_face in enumerate(lb_faces):
            face_name = '{}_{}'.format(name, i)
            hb_face = Face(face_name, lb_face, typ, bc)
            hb_face.display_name = display_name

            # Assign radiance modifier
            hb_face.properties.radiance.modifier = modifier
            
            # Append faces to collection
            faces.append(hb_face)

    return faces

def HB_model(faces, grid, name="model_" + str(time.time())):
    # Set a default name and get the Rhino Model units
    name = clean_string(name)
    units = units_system()

    # Generate model with only name and faces
    model = Model(name, [], faces, [], [], [],
                  units=units, tolerance=tolerance, angle_tolerance=angle_tolerance)
    model.display_name = name
    
    if isinstance(grid, list):
        model.properties.radiance.add_sensor_grids(grid)
    else:
        model.properties.radiance.add_sensor_grids([grid])
    
    return model

#model.to_hbjson(name='xjdsd', folder='C://Users//Job de Vogel//Desktop')

def grid(points, normals, name="custom_SensorGrid"):
    # Set the default name and process the points to tuples
    name = name
    pts = [(pt.X, pt.Y, pt.Z) if pt is not None else None for pt in points]

    # create the sensor grid object
    id  = clean_rad_string(name) if '/' not in name else clean_rad_string(name.split('/')[0])
    if len(normals) == 0:
        grid = SensorGrid.from_planar_positions(id, pts, (0, 0, 1))
    else:
        vecs = [(vec.X, vec.Y, vec.Z) if vec is not None else None for vec in normals]
        grid = SensorGrid.from_position_and_direction(id, pts, vecs)

    # set the display name
    grid.display_name = name
    if '/' in name:
        grid.group_identifier = \
            '/'.join(clean_rad_string(key) for key in name.split('/')[1:])
    
    return grid

def generate(ground_mesh, roof_mesh, wall_mesh, sensors, normals):
    mod = modifier()
    faces = []

    faces.extend(generate_HB_faces(ground_mesh, mod))
    faces.extend(generate_HB_faces(roof_mesh, mod))
    faces.extend(generate_HB_faces(wall_mesh, mod))

    sensor_grid = grid(sensors, normals)
    
    name = str(uuid.uuid4())
    
    model = HB_model(faces, sensor_grid, name=name)
    
    return model