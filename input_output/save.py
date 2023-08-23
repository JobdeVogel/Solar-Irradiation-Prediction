from parameters.params import LOGGER
import Rhino.Geometry as rg
import Rhino
import json
import time

import numpy as np
from parameters.params import PICKLE_PROTOCOL
import datetime

# Generate a meta data dict based on keys and values
def generate_meta(keys, values):
    meta = {}
    
    for key, value in zip(keys, values):
        if key.lower() == 'time':
            meta[key] = str(datetime.datetime.now())
        else:
            meta[key] = value
    
    return meta

# Save a mesh to a json file in base64 format
# The json data is a base64 encoded string of the byte array representing the geometry.
def save_mesh_to_json(meshes, mesh_types, name, folder, meta_data=None):
    start = time.time()
    path = folder + "/" + name + ".json"
    
    options = Rhino.FileIO.SerializationOptions()
    data = {}
    
    for i, mesh in enumerate(meshes):
        if isinstance(mesh, list):
            temp_mesh = rg.Mesh()
            
            for m in mesh:
                temp_mesh.Append(m)
            
            data[mesh_types[i]] = temp_mesh.ToJSON(options)
        else:
            data[mesh_types[i]] = mesh.ToJSON(options)
    
    data['meta'] = meta_data

    with open(path, "w") as file:
        json.dump(data, file)
    
    LOGGER.info(f"Mesh {name} saved in {round(time.time() - start)}s")
    
def save_outlines_to_json(outlines, name, folder):
    start = time.time()
    path = folder + "/" + name + ".json"
    
    options = Rhino.FileIO.SerializationOptions()
    data = {}
    
    for i, outline in enumerate(outlines):
        for j, polyline in enumerate(outline):
            curve = polyline.ToNurbsCurve()
            curve_name = f"curve_{str(i)}_{str(j)}"
            data[curve_name] = curve.ToJSON(options)

    with open(path, "w") as file:
        json.dump(data, file)
    
    LOGGER.info(f"Polyines {name} saved in {round(time.time() - start)}s")

def save_array(array, name, folder):
    start = time.perf_counter()
    path = folder + name + '.npy'
    
    np.save(path, array)
    LOGGER.info(f"Array saved in {round(time.perf_counter() - start)}s")

def save_array_as_list(array, name, folder):
    start = time.time()
    list_data = array.tolist()
    
    path = folder + "/" + name + ".json"
    
    data = json.dumps(list_data)

    with open(path, 'w') as file:
        json.dump(data, file)
    
    LOGGER.info(f"Array_list {name} saved in {round(time.time() - start)}s")
    
    # with open('./data/list_data.pkl', 'wb') as file:
    #     pickle.dump(list_data, file, protocol=PICKLE_PROTOCOL)

# Save an HB model to a file as json file
def save_hbjson(model, name, folder):
    start = time.time()
    model.to_hbjson(name=name, folder=folder)
    LOGGER.info(f"HB_model {name} saved in {round(time.time() - start)}s")