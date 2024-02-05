import math
import os

abs_path = os.path.dirname(os.path.dirname(__file__))

# General
GRID_SIZE = 1.0
OFFSET = 0.01
NUM_AUGMENTS = 1
MIN_FSI = 0.15
MIN_AREA = 10
VISUALIZE_MESH = False

# Load 3DBAG using load_3d_bag.file.py
_FACE_MERGE_TOLERANCE = 0.01


# Patch outline using load_3d_bag.outlines.py
SIZE = 100
MIN_COVERAGE = 100 #less coverage is more samples
TRANSLATE_TO_ORIGIN = True
FSI = True
_SPLIT_TOLERANCE = 0.001

# Meshing using load_3d_bag.meshing.py
MAX_CONTAINMENT_ITERATIONS = 50
_REDUCE_SEGMENTS_TOLERANCE = 0.001
_MESH_SPLITTER_BBOX_HEIGHT = 2
_ANGLE_TOLERANCE_POSTP_MESH = math.pi/90
_DIAGONAL_LENGTH_RATIO_POSTP_MESH = 0.01

# Sensors using load_3d_bag.sensors.py
_MINIMUM_ANGLE = 0.017
_MINIMUM_AREA = 0.0001
_WALL_RAY_TOLERANCE = 0.1
QUAD_ONLY = False

# Validation
MAX_AREA_ERROR = 0.01 # Quite strict, but sure enough to get accurate models

# Save settings
PICKLE_PROTOCOL = 2

# Model settings
REFLECTANCE = 0.2
SPECULAR = 0.0
ROUGHNESS = 0.0
MODIFIER_NAME = 'default_urban_modifier'

_DATA_PATH = os.path.join(abs_path, 'data')

# Directory folders
STATS_PATH = os.path.join(abs_path, 'stats')
BAG_PATH = os.path.join(_DATA_PATH, 'bag')
IRRADIANCE_PATH = os.path.join(_DATA_PATH, 'irradiance')
RAW_PATH = os.path.join(_DATA_PATH, 'raw')
GEOMETRY_PATH = os.path.join(_DATA_PATH, 'geometry')
OUTLINES_PATH = os.path.join(_DATA_PATH, 'outlines')
HBJSON_PATH = os.path.join(_DATA_PATH, 'models')
BAG_FILE_PATH = os.path.join(_DATA_PATH, 'bag\\8-304-528-LoD12-3D.obj') # for single simulations

# Simulation
WEA = os.path.join(_DATA_PATH, 'wea\\NLD_Amsterdam.062400_IWEC.epw')

SIMULATION_ARGUMENTS = '-ab 7 -aa 0 -ar 256 -ad 1024 -as 512 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w'
SIM_OUT_FOLDER = os.path.join(_DATA_PATH, 'simulation')
USE_GPU = True

SKY_DENSITY = 1

WORKERS = 1