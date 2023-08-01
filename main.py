"""
Generate a dataset of 3D BAG LoD 1.2 samples with optional irradiance values.
Parameters can be changed in parameters.params.

Developed by Job de Vogel
Faculty of Architecture and the Built Environment
TU Delft
"""

from parameters.params import LOGGER

import rhinoinside
rhinoinside.load()

# Following packages can be loaded after loading rhinoinside (ignore reportMissingImports)
import Rhino.Geometry as rg
import System
import Rhino
LOGGER.info('Finished loading rhinoinside...')

import time
import os
import numpy as np
import random
import sys

from helpers.mesh import join_meshes
from helpers.array import set_array_values
from load_3dbag import file, outlines, meshing, sensors
from transform import points, model
from input_output import save, serialize
from augment import augmentation
from simulate import run

from parameters.params import BAG_FILE_PATH, IRRADIANCE_PATH, GEOMETRY_PATH, OUTLINES_PATH, SIZE, GRID_SIZE, MIN_COVERAGE, OFFSET, NUM_AUGMENTS, MIN_AREA, WEA, SIMULATION_ARGUMENTS, MIN_FSI, RUN_SIMULATION, ALL_AUGMENTS

def delete_dataset(folder_paths, secure=True):
    """Delete all files in specified directories

    Args:
        folder_paths (list): list of directories (str) to empty
        secure (bool, optional): If secure set to true, ask the user to verify deletion. Defaults to True.
    """

    if secure:
        input(f"Are you sure you want to delete the following datasets? {folder_paths}")
    
    LOGGER.warning(f'Deleting all files in {folder_paths}')
    
    for path in folder_paths:
        for f in os.listdir(path):
            os.remove(os.path.join(path, f))

    LOGGER.warning(f'Deletion successfull!')
    return

def task(patch_outlines, all_building_outlines, all_heights, idx, run_irradiance_simulation=False, all_augments=ALL_AUGMENTS):
    """Generate a dataset sample

    Args:
        patch_outlines (list[rg.Polyline()]): All patch outlines for this 3D BAG file (type: polyline)
        all_building_outlines (list[list[str]]): All building outline polylines in json format
        all_heights (list[float]): Heights of all buildings
        idx (int): Patch index to compute
        run_irradiance_simulation (bool, optional): Run a solar irradiance simulation. Defaults to False.
    """
    
    # Extract the patch outline based on given index
    patch_outline = patch_outlines[idx]
    
    # Deserialize the building outlines
    all_building_outlines = serialize.deserialize(all_building_outlines)

    # Extract the building outlines that correspond with patch outline
    building_outlines, courtyard_outlines, building_heights, FSI_score, envelope_area, building_area = outlines.generate_building_outlines(patch_outline, all_building_outlines, all_heights, MIN_AREA)
    
    # Check if FSI is above minimum FSI
    if FSI_score > MIN_FSI:
        t_preprocessing = time.perf_counter()
        LOGGER.info(f'Started preprocessing mesh for patch[{idx}] with FSI value of {round(FSI_score, 2)}')
        LOGGER.info(f'Generating walls for mesh patch[{idx}]')
        
        # Generate the walls for the building outlines and compute corresponding heights
        walls, wall_outlines, wall_heights = meshing.generate_vertical(building_outlines, courtyard_outlines, building_heights, GRID_SIZE)
        
        LOGGER.debug(f'Generating ground and roofs for mesh patch[{idx}]')
        
        # Compute the mesh plane for the ground and roofs
        mesh_plane, roofs = meshing.generate_horizontal(patch_outline, building_outlines, courtyard_outlines, building_heights, GRID_SIZE)

        LOGGER.debug(f'Computing sensors for mesh patch[{idx}]')
        
        # Comput the sensorpoints for this sample
        sensorpoints, normals = sensors.compute(mesh_plane, roofs, walls, building_heights, GRID_SIZE, OFFSET)
        
        # Filter out the None values for invalid sensors
        filtered_points, filtered_normals, pointmap = sensors.filter_sensors(sensorpoints, normals)
        
        # Join the roofs and walls in a single mesh
        roof_mesh = join_meshes(roofs)
        wall_mesh = join_meshes(walls)
        
        # Compute rough versions of the ground, wall and roof mesh
        rough_ground_mesh = meshing.remesh_horizontal(mesh_plane)
        rough_roof_mesh = meshing.remesh_horizontal(roof_mesh)
        rough_wall_mesh = join_meshes([meshing.remesh_vertical(outline, height) for outline, height in zip(wall_outlines, wall_heights)])
    
        # Save the polylines to a json file
        save.save_outlines_to_json(building_outlines, f'outlines_{idx}', OUTLINES_PATH)

        LOGGER.info(f'Finished preprocessing mesh for patch[{idx}] in {time.perf_counter() - t_preprocessing}s')

        if all_augments:
            # Augment the joined mesh into different orientations
            augments = augmentation.augment([rough_ground_mesh, rough_roof_mesh, rough_wall_mesh], filtered_points, filtered_normals, NUM_AUGMENTS, detailed_meshes=[mesh_plane, roof_mesh, wall_mesh])
            
            meshes, pointclouds, normalclouds, detailed_meshes = augments
            
            for i, (mesh, pointcloud, normalcloud, detailed_mesh) in enumerate(zip(meshes, pointclouds, normalclouds, detailed_meshes)):
                LOGGER.info(f'Started computing augmentation {i} for patch index {idx}.')
                
                array = points.data_to_array(sensorpoints, normals)
                
                rough_ground_mesh, rough_roof_mesh, rough_wall_mesh = mesh
                mesh_plane, roof_mesh, wall_mesh = detailed_mesh
                
                filtered_points = pointcloud
                filtered_normals = normalcloud
                
                if run_irradiance_simulation:
                    # Generate a Honeybee model to simulate
                    HB_model = model.generate([rough_ground_mesh], [rough_roof_mesh], [rough_wall_mesh], filtered_points, filtered_normals)

                    # Simulate irradiance values
                    irradiance = run.main(HB_model, WEA, SIMULATION_ARGUMENTS, pointmap, add_none_values=True)
                
                    array = set_array_values(array, points=filtered_points, normals=filtered_normals, irradiance=irradiance, pointmap=pointmap)
                
                # Save the meshes to a json file
                mesh_types = ['ground', 'roofs', 'walls']
                meshes = [mesh_plane, roof_mesh, wall_mesh]
                
                if i == 0:
                    augment_idx = 'base'
                else:
                    augment_idx = 'rot' + str(i)
                
                save.save_mesh_to_json(meshes, mesh_types, f'mesh_{idx}_{augment_idx}', GEOMETRY_PATH)
                
                # Save the sensorpoints to a json file
                save.save_array_as_list(array, f'sensors_{idx}_{augment_idx}', IRRADIANCE_PATH)
        else:
            # Create an array for the sensorpoints
            array = points.data_to_array(sensorpoints, normals)
            
            if run_irradiance_simulation:
                # Generate a Honeybee model to simulate
                HB_model = model.generate([rough_ground_mesh], [rough_roof_mesh], [rough_wall_mesh], filtered_points, filtered_normals)
            
                # Simulate irradiance values
                irradiance = run.main(HB_model, WEA, SIMULATION_ARGUMENTS, pointmap, add_none_values=True)
            
                # Add the irradiance values to the sensorpoint array
                array = set_array_values(array, irradiance=irradiance)
            
            # Save the meshes to a json file
            mesh_types = ['ground', 'roofs', 'walls']
            meshes = [mesh_plane, roof_mesh, wall_mesh]
            save.save_mesh_to_json(meshes, mesh_types, f'mesh_{idx}_base', GEOMETRY_PATH)
            
            # Save the sensorpoints to a json file
            save.save_array_as_list(array, f'sensors_{idx}_base', IRRADIANCE_PATH)

    else:
        LOGGER.info(f'FSI_score {round(FSI_score, 2)} of sample {idx} not high enough to continue generating sample.')
    
def main(filename, start_idx, run_irradiance_simulation=RUN_SIMULATION):
    """Generate a sample, and optionally simulate solar irradiance.

    Args:
        filename (str): 3D BAG dataset patch
        start_idx (int): First patch sample index to compute
    """
    
    # Extract roof and wall meshes from 3D BAG dataset
    roof_meshes, wall_meshes, bbox = file.load(filename)

    # Generate ground outlines for dataset patches
    patch_outlines = outlines.generate_outlines_from_bbox(bbox, SIZE, MIN_COVERAGE)

    # Generate the building outlines for ALL building meshes
    all_building_outlines, all_heights = outlines.extract_building_outlines(wall_meshes, roof_meshes)
    
    # Sort the building_outlines to format [outer, inner_1, inner_2, etc.]
    for i, building_outline in enumerate(all_building_outlines):
        # If there are inner courtyards
        if len(building_outline) > 1:
            sorted_outlines = outlines.find_outer_polyline(building_outline)
            all_building_outlines[i] = sorted_outlines
    
    # Serialize the building outlines to avoid in-place changes
    all_building_outlines = serialize.serialize(all_building_outlines)

    # Iterate over all patch_outlines
    for idx in range(len(patch_outlines))[start_idx:]:
        start = time.perf_counter()
        LOGGER.info(f'Started computing patch[{idx}].')
        
        # Run the generation and simulation for one ground patch sample
        task(patch_outlines, all_building_outlines, all_heights, idx, run_irradiance_simulation=run_irradiance_simulation)
        
        LOGGER.info(f'Finished computing patch[{idx}] in {round(time.perf_counter() - start, 2)}s.')
        print('\n')

if __name__ == '__main__':
    random.seed(0)
    filename = BAG_FILE_PATH
    start_idx = 0
    
    # Delete the database
    folder_paths = [GEOMETRY_PATH, IRRADIANCE_PATH, OUTLINES_PATH]
    delete_dataset(folder_paths, secure=True)
    
    # Run the sample generation
    main(filename, start_idx)