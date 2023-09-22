import math

# General
BAG_FILE_PATH = "./data/bag/9-324-520-LoD12-3D.obj"
WEA = "./data/wea/NLD_Amsterdam.062400_IWEC.epw"
GRID_SIZE = 1.0
OFFSET = 0.01
NUM_AUGMENTS = 1
MIN_FSI = 0.2
MIN_AREA = 10
VISUALIZE_MESH = False

# Load 3DBAG using load_3d_bag.file.py
_FACE_MERGE_TOLERANCE = 0.01

# Patch outline using load_3d_bag.outlines.py
SIZE = 100
MIN_COVERAGE = 25 #less coverage is more samples
TRANSLATE_TO_ORIGIN = True
FSI = True
_SPLIT_TOLERANCE = 0.01

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

# Save settings
PICKLE_PROTOCOL = 2

# Model settings
REFLECTANCE = 0.0
SPECULAR = 0.0
ROUGHNESS = 0.0
MODIFIER_NAME = 'default_urban_modifier'

# Directory folders
STATS_PATH = './stats/'
BAG_PATH = './data/bag/'
IRRADIANCE_PATH = './data/irradiance/'
RAW_PATH = './data/raw/'
GEOMETRY_PATH = './data/geometry/'
OUTLINES_PATH = './data/outlines/'
HBJSON_PATH = './data/models/'

# Simulation
WEA_PATH = "C:\\Users\\Job de Vogel\\AppData\\Roaming\\ladybug_tools\\weather\\NLD_Amsterdam.062400_IWEC\\NLD_Amsterdam.062400_IWEC.epw"

SIMULATION_ARGUMENTS = '-ab 9 -ad 5000 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15'

USE_GPU = True

SKY_DENSITY = 1

WORKERS = 1