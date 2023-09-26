# Honeybee Core dependencies
from honeybee.model import Model
from simulate import run

from lbt_recipes.recipe import Recipe
from lbt_recipes.settings import RecipeSettings

from parameters.params import USE_GPU, SKY_DENSITY, WORKERS, WEA, SIMULATION_ARGUMENTS, SIM_OUT_FOLDER

# Load an HB model from a file
def load_hbjson(name, folder):
    path = folder + "/" + name + ".hbjson"
    model = Model.from_file(path)
    
    return model

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
    settings.folder = SIM_OUT_FOLDER + '/' + model.display_name
    
    print('Started simulating...')
    # Run the simulation 
    project_folder = recipe.run(settings=settings, radiance_check=True, silent=False, queenbee_path='queenbee')

    # Retrieve the results
    irradiance = recipe.output_value_by_name('cumulative-radiation', project_folder)[0]
    
    return irradiance

def par_convergence():
    pass

if __name__=='__main__':
    name, folder = 'renderfarm_test_model', './data/renderfarm'
    
    print('Loading hbjson')
    model = load_hbjson(name, folder)
    print('Finished loading')
    
    values = annual_irradiance(model, WEA, SIMULATION_ARGUMENTS)
    print(values)
    