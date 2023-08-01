"""

"""

from honeybee.model import Model
from honeybee.room import Room
from honeybee.config import folders as hb_folders
from lbt_recipes.recipe import Recipe

import honeybee.dictutil as hb_dict_util
import honeybee_radiance.dictutil as radiance_dict_util
import honeybee_energy.dictutil as energy_dict_util

import json
import time
import warnings

def annual_irradiance(model, wea, sim_arguments, identifier='custom_name', distplay_name='custom_name'):   
    # Assign an identifier and display_name
    model.identifier = identifier
    model.display_name = distplay_name

    # Pass the model to the recipe
    recipe = Recipe('cumulative-radiation')
#    recipe = Recipe('annual-irradiance')
    
    # Assign the correct parameters to the simulation
    recipe.input_value_by_name('model', model)
    recipe.input_value_by_name('wea', wea)
    
    if recipe.name == 'cumulative_radiation':
        recipe.input_value_by_name('sky-density', 1)
    
    recipe.input_value_by_name('radiance-parameters', sim_arguments)
    
    # Run the simulation
    warnings.warn(f'Started solar irradiance simulation for model with identiefier {identifier}')
    
    start = time.perf_counter()
    project_folder = recipe.run(settings='--workers 1', radiance_check=True)
    end = time.perf_counter()
    
    warnings.warn(f'Finished solar irradiance simulation for model with identiefier {identifier} in {end-start}s')

    # Retrieve the results
    irradiance = recipe.output_value_by_name('cumulative-radiation', project_folder)[0]
    
    return irradiance

def main(model, wea, sim_arguments, pointmap):
    # Compute the annual irradiance for the given model
    values = annual_irradiance(model, wea, sim_arguments)

    # Store the irradiance values such that invalid sensorpoints get irriance value of 0
    irradiance = iter(values)
    
    result_mesh = []
    for point in pointmap:
        if point:
            result_mesh.append(next(irradiance))
        else:
            result_mesh.append(0)
    
if __name__ == '__main__':
    model = ''
    wea = ''
    pointmap = ''
    
    sim_arguments = '-ab 6 -ad 25000 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15'
    
    main(model, wea, sim_arguments, pointmap)