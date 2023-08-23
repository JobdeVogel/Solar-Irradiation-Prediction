"""
Generate a dataset of 3D BAG LoD 1.2 samples with optional irradiance values.
Parameters can be changed in parameters.params.

Developed by Job de Vogel
Faculty of Architecture and the Built Environment
TU Delft
"""

from parameters.params import LOGGER
# LOGGER.info('Started loading packages and modules...')

import rhinoinside
rhinoinside.load()

# Following packages can be loaded after loading rhinoinside (ignore reportMissingImports)
import Rhino.Geometry as rg
import System
import Rhino

import time
import os
import numpy as np
import random
import string
import sys
import argparse

from helpers.mesh import join_meshes
from helpers.array import set_array_values
from load_3dbag import file, outlines, meshing, sensors
from transform import points, model
from input_output import save, serialize
from augment import augmentation
from simulate import run
from visualize.mesh import generate_colored_mesh, legend

from parameters.params import BAG_FILE_PATH, IRRADIANCE_PATH, GEOMETRY_PATH, RAW_PATH, OUTLINES_PATH, SIZE, GRID_SIZE, MIN_COVERAGE, OFFSET, NUM_AUGMENTS, MIN_AREA, WEA, SIMULATION_ARGUMENTS, MIN_FSI, VISUALIZE_MESH

LOGGER.info('Finished loading packages and modules...')

parser = argparse.ArgumentParser(prog='name', description='random info', epilog='random bottom info')
parser.add_argument('-b', '--BAG_FILE_PATH', type=str, nargs='?', default=BAG_FILE_PATH, help='')
parser.add_argument('-i', '--IRRADIANCE_PATH', type=str, nargs='?', default=IRRADIANCE_PATH, help='')
parser.add_argument('-g', '--GEOMETRY_PATH', type=str, nargs='?', default=GEOMETRY_PATH, help='')
parser.add_argument('-raw', '--RAW_PATH', type=str, nargs='?', default=RAW_PATH, help='')
parser.add_argument('-o', '--OUTLINES_PATH', type=str, nargs='?', default=OUTLINES_PATH, help='')
parser.add_argument('-s', '--SIZE', nargs='?', type=float, default=SIZE, help='')
parser.add_argument('-gs', '--GRID_SIZE', type=float, nargs='?', default=GRID_SIZE, help='')
parser.add_argument('-mc', '--MIN_COVERAGE', type=float, nargs='?', default=MIN_COVERAGE, help='')
parser.add_argument('-of', '--OFFSET', type=float, nargs='?', default=OFFSET, help='')
parser.add_argument('-na', '--NUM_AUGMENTS', type=int, nargs='?', default=NUM_AUGMENTS, help='')
parser.add_argument('-ma', '--MIN_AREA', type=float, nargs='?', default=MIN_AREA, help='')
parser.add_argument('-w', '--WEA', type=str, nargs='?', default=WEA, help='')
parser.add_argument('-sa', '--SIMULATION_ARGUMENTS', type=str, nargs='?', default=SIMULATION_ARGUMENTS, help='')
parser.add_argument('-f', '--MIN_FSI', type=float, nargs='?', default=MIN_FSI, help='')
parser.add_argument('-v', '--VISUALIZE_MESH', default=VISUALIZE_MESH, action='store_true', help='')

args= parser.parse_args()
BAG_FILE_PATH, IRRADIANCE_PATH, GEOMETRY_PATH, RAW_PATH, OUTLINES_PATH, SIZE, GRID_SIZE, MIN_COVERAGE, OFFSET, NUM_AUGMENTS, MIN_AREA, WEA, SIMULATION_ARGUMENTS, MIN_FSI, VISUALIZE_MESH = vars(args).values()

class Sample:
    def __init__(self, idx):
        self.idx = idx
        self.ground = rg.Mesh()
        self.patch_outline = None
        self.building_outlines = []
        self.courtyard_outlines = []
        self.heights = []        
        self.walls = []
        self.roofs = []
        
        self.rough_ground = None
        self.rough_walls = None
        self.rough_roofs = None
        
        self.sensorpoints = []
        self.sensornormals = []
        
        self.filtered_points = []
        self.filtered_normals = []
        self.pointmap = []
        
        self.augments = []
        
        self.irradiance_results = []
        self.arrays = []
        
        self.models = []
        self.FSI_score = 0
        self.envelope_area = 0
        self.building_area = 0
    
    def compute_outlines(self, patch_outline, all_building_outlines, all_heights):
        # Store the patch outline as rectangle3d
        self.patch_outline = patch_outline
        
        try:
            # Extract the building outlines that correspond with patch outline
            self.building_outlines, self.courtyard_outlines, self.heights, self.FSI_score, self.envelope_area, self.building_area = outlines.generate_building_outlines(
                patch_outline, 
                all_building_outlines, 
                all_heights)
        except Exception as e:
            LOGGER.critical(f'Outlines computation for sample {self.idx} failed')
            LOGGER.critical(e)
    
    def compute_mesh(self, rough=True):
        try:
            # Generate meshes
            self.ground, self.walls, self.roofs, self.rough_ground, self.rough_walls, self.rough_roofs = meshing.generate_mesh(
                self.patch_outline, 
                self.building_outlines, 
                self.courtyard_outlines, 
                self.heights, 
                GRID_SIZE, 
                SIZE, 
                rough=rough)
        except Exception as e:
            LOGGER.critical(f'Mesh computation for sample {self.idx} failed')
            LOGGER.critical(e)
    
    def compute_sensors(self, compute_filtered=True):
        try:
            # Compute the sensorpoints for this sample
            self.sensorpoints, self.sensornormals = sensors.compute(self.ground, self.roofs, self.walls, self.heights, GRID_SIZE, OFFSET)
            
            if compute_filtered:
                # # Filter out the None values for invalid sensors
                self.filtered_points, self.filtered_normals, self.pointmap = sensors.filter_sensors(self.sensorpoints, self.sensornormals)
        except Exception as e:
            LOGGER.critical(f'Sensors computation for sample {self.idx} failed')
            LOGGER.critical(e)

    def augment(self):
        rough_roof = join_meshes(self.rough_roofs)
        rough_walls = join_meshes(self.rough_walls)
        
        roof_mesh = join_meshes(self.roofs)
        wall_mesh = join_meshes(self.walls)   
        
        # Augment the joined mesh into different orientations
        self.augments = augmentation.augment(
            [self.rough_ground, rough_roof, rough_walls], 
            self.filtered_points, 
            self.filtered_normals, 
            NUM_AUGMENTS, 
            detailed_meshes=[self.ground, roof_mesh, wall_mesh])

        self.models = [None for i in range(len(self.augments[0]))]
        self.arrays = [None for i in range(len(self.augments[0]))]
        self.irradiance_results = [None for i in range(len(self.augments[0]))]
        
        return len(self.augments[0])
  
    def add_model(self, augment_idx):
        meshes, pointclouds, normalclouds, _ = self.augments
        
        mesh = meshes[augment_idx]
        pointcloud = pointclouds[augment_idx]
        normalcloud = normalclouds[augment_idx]
        
        rough_ground_mesh, rough_roof_mesh, rough_wall_mesh = mesh
        
        HB_model = model.generate([rough_ground_mesh], [rough_roof_mesh], [rough_wall_mesh], pointcloud, normalcloud)
        
        self.models[augment_idx] = HB_model
        
    def simulate(self, augment_idx):
        model = self.models[augment_idx]
        
        name = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        
        # Simulate irradiance values
        self.irradiance_results[augment_idx] = run.main(model, WEA, SIMULATION_ARGUMENTS, self.pointmap, add_none_values=True)    
    
    def store_sensors_as_arrays(self):
        base_array = points.data_to_array(self.sensorpoints, self.sensornormals)
        
        meshes, pointclouds, normalclouds, detailed_meshes = self.augments

        for i, (mesh, pointcloud, normalcloud, detailed_mesh) in enumerate(zip(meshes, pointclouds, normalclouds, detailed_meshes)):
            irradiance = self.irradiance_results[i]
            
            if irradiance != None:
                array = set_array_values(base_array, points=pointcloud, normals=normalcloud, irradiance=irradiance, pointmap=self.pointmap)
            else:
                array = set_array_values(base_array, points=pointcloud, normals=normalcloud, pointmap=self.pointmap)
            
            self.arrays[i] = array
    
    def save_mesh(self, visualization=False):
        irradiance = self.irradiance_results[0]
        mesh_legend = legend()
        
        roof_mesh = join_meshes(self.roofs)
        wall_mesh = join_meshes(self.walls)
        
        # Generate a colored mesh that can be visualized from Ladybug Grasshopper
        if visualization:
            mesh = join_meshes([self.ground_mesh, roof_mesh, wall_mesh])
            colored_mesh = generate_colored_mesh(mesh, irradiance, mesh_legend)
    
            mesh_types = ['ground', 'roofs', 'walls', 'colored_mesh']
            meshes = [self.ground_mesh, roof_mesh, wall_mesh, colored_mesh]
        else:
            mesh_types = ['ground', 'roofs', 'walls']
            meshes = [self.ground_mesh, roof_mesh, wall_mesh]

        # Save the meshes to a json file
        save.save_mesh_to_json(meshes, mesh_types, f'mesh_{self.idx}_base', GEOMETRY_PATH)
    
    def save_augment_meshes(self, visualization=False):
        _, _, _, detailed_meshes = self.augments
        mesh_legend = legend()

        for i, mesh in enumerate(detailed_meshes):
            ground_mesh, roof_mesh, wall_mesh = mesh
        
            # Generate a colored mesh that can be visualized from Ladybug Grasshopper
            if visualization:
                irradiance = self.irradiance_results[i]

                mesh = join_meshes([ground_mesh, roof_mesh, wall_mesh])
                colored_mesh = generate_colored_mesh(mesh, irradiance, mesh_legend)
                
                mesh_types = ['ground', 'roofs', 'walls', 'colored_mesh']
                meshes = [ground_mesh, roof_mesh, wall_mesh, colored_mesh]
            else:
                mesh_types = ['ground', 'roofs', 'walls']
                meshes = [ground_mesh, roof_mesh, wall_mesh]

            if i == 0:
                name = 'base'
            else:
                name = 'rot' + str(i)

            # Save the meshes to a json file
            save.save_mesh_to_json(meshes, mesh_types, f'mesh_{self.idx}_{name}', GEOMETRY_PATH)

    def save_raw(self):
        for i, array in enumerate(self.arrays):
            name = f'irradiance_sample_{self.idx}_augmentation_{i}'
            
            save.save_array(array, name, RAW_PATH)

    def save_sensors(self):
        for i, array in enumerate(self.arrays):        
            if i == 0:
                name = 'base'
            else:
                name = 'rot' + str(i)
    
            # Save the sensorpoints to a json file
            save.save_array_as_list(array, f'sensors_{self.idx}_{name}', IRRADIANCE_PATH)

    def save_outlines(self):    
        # # Save the polylines to a json file
        save.save_outlines_to_json(self.building_outlines, f'outlines_{self.idx}', OUTLINES_PATH)
    
    @property
    def count(self):
        return len(self.augments[0])
    
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

def task(patch_outline, all_building_outlines, all_heights, idx, visualize_mesh=VISUALIZE_MESH):
    """Generate a dataset sample

    Args:
        patch_outlines (list[rg.Polyline()]): All patch outlines for this 3D BAG file (type: polyline)
        all_building_outlines (list[list[str]]): All building outline polylines in json format
        all_heights (list[float]): Heights of all buildings
        idx (int): Patch index to compute
        run_irradiance_simulation (bool, optional): Run a solar irradiance simulation. Defaults to False.
    """

    # Initializa a sample
    sample = Sample(idx)
    
    # Compute the outlines
    sample.compute_outlines(patch_outline, all_building_outlines, all_heights)
    
    # save the outlines
    # sample.save_outlines()
    
    # Check if FSI is above minimum FSI
    if sample.FSI_score > MIN_FSI:
        t_preprocessing = time.perf_counter()
        
        # Generate meshes
        LOGGER.info(f'Started preprocessing mesh for patch[{sample.idx}] with FSI value of {round(sample.FSI_score, 2)}')
        sample.compute_mesh()    

        # Compute the sensors
        LOGGER.info(f'Computing sensors for mesh patch[{sample.idx}]')
        sample.compute_sensors()

        # Augment the sample to different orientations
        sample.augment()
        
        # Iterate over the augmentations
        for idx in range(sample.count):
                                                
            LOGGER.info(f'Generating model for mesh patch[{sample.idx}] augmentation {idx}')
            sample.add_model(idx)
            LOGGER.info(f'Simulating irradiance model for mesh patch[{sample.idx}] augmentation {idx}')
            sample.simulate(idx)

        # Store the sensors, including irradiance values, as arrays in the sample object
        sample.store_sensors_as_arrays()
        
        # Save the detailed geometry
        LOGGER.info(f'Saving mesh patch[{sample.idx}] and generating visualization')
        sample.save_augment_meshes(visualization=visualize_mesh)
        
        # Save the irradiance values as lists
        sample.save_sensors()
        
        # Save the results as npy file
        sample.save_raw()
        
        LOGGER.info(f'Finished preprocessing mesh for patch[{sample.idx}] in {round(time.perf_counter() - t_preprocessing, 2)}s')
        del sample
    else:
        LOGGER.info(f'FSI_score {round(sample.FSI_score, 2)} of sample {sample.idx} not high enough to continue generating sample.')

def main(filename, start_idx):
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

    # Store the samples
    samples = []

    # Iterate over all patch_outlines
    for idx in range(len(patch_outlines))[start_idx:]:
        start = time.perf_counter()
        LOGGER.info(f'Started computing patch[{idx}].')
        
        sample = None

        # Deserialize the building outlines
        deserializeed_building_outlines = serialize.deserialize(all_building_outlines)
        
        # Extract the patch outline based on given index
        patch_outline = patch_outlines[idx]
        
        # # Run the generation and simulation for one ground patch sample
        sample = task(patch_outline, deserializeed_building_outlines, all_heights, idx)
            
        
        # Append the sample
        samples.append(sample)
        
        LOGGER.info(f'Finished computing patch[{idx}] in {round(time.perf_counter() - start, 2)}s.')

    return samples


if __name__ == '__main__':
    random.seed(0)
    filename = BAG_FILE_PATH
    
    # ! problems: 181, 448
    # ! nice looking 447
    start_idx = 0
    
    # Delete the database
    folder_paths = [GEOMETRY_PATH, IRRADIANCE_PATH, OUTLINES_PATH, RAW_PATH]
    delete_dataset(folder_paths, secure=True)
    
    # Run the sample generation
    main(filename, start_idx)