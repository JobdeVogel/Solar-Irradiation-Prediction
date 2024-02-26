import math
import os

abs_path = os.path.dirname(os.path.dirname(__file__))

# General
GRID_SIZE = 1.0
OFFSET = 0.01
NUM_AUGMENTS = 1
RANDOM_SENSORS = True

'''
When the size of the dataset sample is increased, it is recommended
to also reduce the minimum GSI. Bigger patches of land typically have
a lower GSI.
'''
MIN_GSI = 0.15
MIN_AREA = 10
VISUALIZE_MESH = False

# Load 3DBAG using load_3d_bag.file.py
_FACE_MERGE_TOLERANCE = 0.01


# Patch outline using load_3d_bag.outlines.py
'''
For size it is important to note that the 3D BAG has patches of different
sizes. Size 10 has an outline of slightly more than 600x600m. If the given
size is greater than the 3D BAG sample size, it will be skipped. This means
that a dense area will be skipped.

Size indication:
* BAG size 11: 300x300m
* BAG size 10: 600x600m
* BAG size 9: 1200x1200m
* BAG size 8: 2400x2400m

If the size of a patch increases it is possible that the preprocessing time on
CPU will outweigh the simulation time on GPU.
'''

SIZE = 100

'''
MIN_COVERAGE is a percentage of the area that will NOT be reused in the next sample.
A min coverage of 80 means that 20% of the sample will be reused in the next sample.
'''
MIN_COVERAGE = 100 #less coverage is more samples
TRANSLATE_TO_ORIGIN = True
GSI = True
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