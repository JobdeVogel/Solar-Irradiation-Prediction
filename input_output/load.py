import Rhino.Geometry as rg

import Rhino
import json
import numpy as np
import pickle

# Honeybee Core dependencies
from honeybee.model import Model

# Load a mesh from a json file
def load_mesh_from_json(name, folder, mesh_types):
    path = folder + "/" + name + ".json"

    with open(path, "r") as file:
        data = json.load(file)

    if isinstance(mesh_types, str):
        if mesh_types == 'meta':
            return data['meta']
        else:
            return [rg.Mesh.FromJSON(data[mesh_types])], data['meta']
    else:
        meta = data['meta']
        data.pop('meta')
        meshes = []
        
        for key in data.keys():
            meshes.append(rg.Mesh.FromJSON(data[key]))
        
        return meshes, meta

# Load a mesh from a json file
def load_outlines_from_json(name, folder):
    path = folder + "/" + name + ".json"

    with open(path, "r") as file:
        data = json.load(file)

    curves = []
    for key in data.keys():
        curves.append(rg.NurbsCurve.FromJSON(data[key]))
        
    return data

def load_array(file):
     return np.load(file)

def load_array_as_list(name, folder):
    path = folder + "/" + name + ".json"
    
    with open(path, 'r') as file:
        data = json.load(file)
    
    data = json.loads(data)
        
    # with open(file, 'rb') as file:
    #     data = pickle.load(file)
    
    return data

# Load an HB model from a file
def load_hbjson(name, folder):
    path = folder + "/" + name + ".hbjson"
    model = Model.from_file(path)