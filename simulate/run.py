"""
Run a solar irradiance simulation for a given model

Developed by Job de Vogel : 2023-07-17
"""
from lbt_recipes.recipe import Recipe
from lbt_recipes.settings import RecipeSettings

from parameters.params import USE_GPU, SKY_DENSITY, WORKERS

def annual_irradiance(model, wea, sim_arguments):   
    # Pass the model to the recipe
    recipe = Recipe('cumulative-radiation')
#    recipe = Recipe('annual-irradiance')
    
    # Assign the correct parameters to the simulation
    recipe.input_value_by_name('model', model)
    recipe.input_value_by_name('wea', wea)
    
    if recipe.name == 'cumulative_radiation':
        recipe.input_value_by_name('sky-density', SKY_DENSITY)
    
    if not USE_GPU:
        sim_arguments += ' -g 0'
    
    recipe.input_value_by_name('radiance-parameters', sim_arguments)

    settings = RecipeSettings()
    settings.workers = WORKERS
    settings.report_out = False
    
    # Run the simulation 
    project_folder = recipe.run(settings=settings, radiance_check=True, silent=False, queenbee_path='queenbee')

    # Retrieve the results
    irradiance = recipe.output_value_by_name('cumulative-radiation', project_folder)[0]
    
    return irradiance

def main(model, wea, sim_arguments, pointmap, add_none_values=True, logger=False):
    if logger:
        logger.info(f'Started solar irradiance simulation with arguments {sim_arguments}')
    # Compute the annual irradiance for the given model
    values = annual_irradiance(model, wea, sim_arguments)
    
    if logger:
        logger.info(f'Computed {len(values)} solar irradiance values for model {model.display_name}')
    
    # Store the irradiance values such that invalid sensorpoints get irriance value of 0
    irradiance = iter(values)
    
    result_mesh = []
    for point in pointmap:
        if point:
            result_mesh.append(next(irradiance))
        else:
            if add_none_values:
                result_mesh.append(0)
    
    return result_mesh