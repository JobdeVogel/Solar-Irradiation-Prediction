# Honeybee Core dependencies
from honeybee.model import Model
from simulate import run

from lbt_recipes.recipe import Recipe
from lbt_recipes.settings import RecipeSettings

from parameters.params import USE_GPU, SKY_DENSITY, WORKERS, WEA, SIMULATION_ARGUMENTS, SIM_OUT_FOLDER

import csv
import time

# Load an HB model from a file
def load_hbjson(name, folder):
    path = folder + "/" + name + ".hbjson"
    model = Model.from_file(path)
    
    return model

def annual_irradiance(model, wea, sim_arguments, workers):   
    # Pass the model to the recipe
    recipe = Recipe('cumulative-radiation')
#    recipe = Recipe('annual-irradiance')
    
    # Assign the correct parameters to the simulation
    recipe.input_value_by_name('model', model)
    recipe.input_value_by_name('wea', wea)
    
    if recipe.name == 'cumulative_radiation':
        recipe.input_value_by_name('sky-density', SKY_DENSITY)
    
    # if not USE_GPU:
    #     sim_arguments += ' -g 0'
    
    recipe.input_value_by_name('radiance-parameters', sim_arguments)
    print(recipe._inputs)

    settings = RecipeSettings()
    settings.workers = workers
    settings.report_out = False
    settings.reload_old = False
    settings.folder = SIM_OUT_FOLDER + '/' + model.display_name
    
    print('Started simulating...')
    # Run the simulation 
    project_folder = recipe.run(settings=settings, radiance_check=True, silent=False)

    # Retrieve the results
    irradiance = recipe.output_value_by_name('cumulative-radiation', project_folder)[0]
    
    return irradiance

def par_convergence(model, parameters):
    for j, params in enumerate(parameters):
        print(f'Started simulation {j} with params {params}')
        
        if str(params[-1]) == 'g':
            continue
            
            print(f'Simulating with 11 CPU workers')
            start = time.time()
            values = annual_irradiance(model, WEA, params, 11)
            end = time.time()
            values += [end-start]
        else:
            start = time.time()        
            values = annual_irradiance(model, WEA, params, WORKERS)
            end = time.time()
            values += [end-start]
            
        with open('./data/convergence/results.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(values)
            
        with open('./data/convergence/params.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            
            data = params.split(" ")
            par_values = []
            par_types = []

            for i, d in enumerate(data):
                if not i%2 == 0:
                    par_values.append(d)
                else:
                    par_types.append(d[1:])
            
            if j == 0:
                writer.writerow(par_types)
                writer.writerow(par_values)
            else:
                writer.writerow(par_values)
            
if __name__=='__main__':
    name, folder = 'renderfarm_test_model', './data/renderfarm'
    
    print('Loading hbjson')
    model = load_hbjson(name, folder)
    print('Finished loading')
    
    # values = annual_irradiance(model, WEA, SIMULATION_ARGUMENTS, 1)
    # print(values)
    
    # arguments = ['-ab 3 -aa 0.4 -ar 8 -ad 32 -as 16 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 3 -aa 0.2 -ar 16 -ad 64 -as 32 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 3 -aa 0.1 -ar 32 -ad 128 -as 64 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 3 -aa 0.05 -ar 64 -ad 256 -as 128 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 3 -aa 0 -ar 128 -ad 512 -as 256 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 3 -aa 0 -ar 256 -ad 1024 -as 512 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 3 -aa 0 -ar 512 -ad 2048 -as 1024 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 3 -aa 0 -ar 1024 -ad 4096 -as 2048 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 3 -aa 0 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 3 -aa 0.05 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -g',
    #              '-ab 4 -aa 0.4 -ar 8 -ad 32 -as 16 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 4 -aa 0.2 -ar 16 -ad 64 -as 32 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 4 -aa 0.1 -ar 32 -ad 128 -as 64 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 4 -aa 0.05 -ar 64 -ad 256 -as 128 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 4 -aa 0 -ar 128 -ad 512 -as 256 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 4 -aa 0 -ar 256 -ad 1024 -as 512 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 4 -aa 0 -ar 512 -ad 2048 -as 1024 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 4 -aa 0 -ar 1024 -ad 4096 -as 2048 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 4 -aa 0 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 4 -aa 0.05 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -g',
    #              '-ab 5 -aa 0.4 -ar 8 -ad 32 -as 16 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 5 -aa 0.2 -ar 16 -ad 64 -as 32 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 5 -aa 0.1 -ar 32 -ad 128 -as 64 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 5 -aa 0.05 -ar 64 -ad 256 -as 128 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 5 -aa 0 -ar 128 -ad 512 -as 256 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 5 -aa 0 -ar 256 -ad 1024 -as 512 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 5 -aa 0 -ar 512 -ad 2048 -as 1024 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 5 -aa 0 -ar 1024 -ad 4096 -as 2048 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 5 -aa 0 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 5 -aa 0.05 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -g',
    #              '-ab 7 -aa 0.4 -ar 8 -ad 32 -as 16 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 7 -aa 0.2 -ar 16 -ad 64 -as 32 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 7 -aa 0.1 -ar 32 -ad 128 -as 64 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 7 -aa 0.05 -ar 64 -ad 256 -as 128 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 7 -aa 0 -ar 128 -ad 512 -as 256 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 7 -aa 0 -ar 256 -ad 1024 -as 512 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 7 -aa 0 -ar 512 -ad 2048 -as 1024 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 7 -aa 0 -ar 1024 -ad 4096 -as 2048 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 7 -aa 0 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 7 -aa 0.05 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -g',
    #              '-ab 9 -aa 0.4 -ar 8 -ad 32 -as 16 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 9 -aa 0.2 -ar 16 -ad 64 -as 32 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 9 -aa 0.1 -ar 32 -ad 128 -as 64 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 9 -aa 0.05 -ar 64 -ad 256 -as 128 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 9 -aa 0 -ar 128 -ad 512 -as 256 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 9 -aa 0 -ar 256 -ad 1024 -as 512 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 9 -aa 0 -ar 512 -ad 2048 -as 1024 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 9 -aa 0 -ar 1024 -ad 4096 -as 2048 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 9 -aa 0 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 9 -aa 0.05 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -g',
    #              '-ab 11 -aa 0.4 -ar 8 -ad 32 -as 16 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 11 -aa 0.2 -ar 16 -ad 64 -as 32 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 11 -aa 0.1 -ar 32 -ad 128 -as 64 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 11 -aa 0.05 -ar 64 -ad 256 -as 128 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 11 -aa 0 -ar 128 -ad 512 -as 256 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 11 -aa 0 -ar 256 -ad 1024 -as 512 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 11 -aa 0 -ar 512 -ad 2048 -as 1024 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 11 -aa 0 -ar 1024 -ad 4096 -as 2048 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 11 -aa 0 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 11 -aa 0.05 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -g',
    #              '-ab 12 -aa 0.4 -ar 8 -ad 32 -as 16 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 12 -aa 0.2 -ar 16 -ad 64 -as 32 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 12 -aa 0.1 -ar 32 -ad 128 -as 64 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 12 -aa 0.05 -ar 64 -ad 256 -as 128 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 12 -aa 0 -ar 128 -ad 512 -as 256 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 12 -aa 0 -ar 256 -ad 1024 -as 512 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 12 -aa 0 -ar 512 -ad 2048 -as 1024 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 12 -aa 0 -ar 1024 -ad 4096 -as 2048 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 12 -aa 0 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
    #              '-ab 12 -aa 0.05 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -g',
    #              ]
    
    arguments = ['-ab 13 -aa 0 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
                 '-ab 14 -aa 0 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
                 '-ab 15 -aa 0 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
                 '-ab 16 -aa 0 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15',
                 '-ab 17 -aa 0 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15']
    
    par_convergence(model, arguments[49:])