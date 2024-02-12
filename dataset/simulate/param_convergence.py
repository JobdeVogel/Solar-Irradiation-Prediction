# Honeybee Core dependencies

import sys
import os
 
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
 
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
 
# adding the parent directory to 
# the sys.path.
sys.path.append(parent)

from honeybee.model import Model

from lbt_recipes.recipe import Recipe
from lbt_recipes.settings import RecipeSettings

import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt

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
    # settings.folder = SIM_OUT_FOLDER + '/' + model.display_name
    
    print('Started simulating...')
    # Run the simulation 
    project_folder = recipe.run(settings=settings, radiance_check=True, silent=False)

    # # Retrieve the results
    irradiance = recipe.output_value_by_name('cumulative-radiation', project_folder)[0]
    
    return irradiance

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def par_convergence(model, parameters, output_file):
    for j, params in enumerate(parameters):
        print(f'Started simulation {j} with params {params}')
        
        if '-g' in params:            
            print(f'Simulating with 39 CPU workers')
            start = time.time()
            values = annual_irradiance(model, WEA, params, 39)
            end = time.time()
            values += [end-start]
        else:
            # continue
            start = time.time()        
            values = annual_irradiance(model, WEA, params, WORKERS)
            end = time.time()
            values += [end-start]
            
        with open(output_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(values)
            
def errors(df):    
    rmses = []
    max_errors = []
    stds = []
    for i in range(df.shape[1]):
        column_index = i

        selected_column = df.iloc[:-1, column_index]
        comparison_column = df.iloc[:-1, 0]
        
        squared_errors = (selected_column - comparison_column) ** 2
        mean = squared_errors.mean()
        rmse = math.sqrt(mean)  
            
        error = np.max(np.abs(selected_column - comparison_column))
        std = np.std(np.abs(selected_column - comparison_column))
        
        max_index = np.argmax(np.abs(selected_column - comparison_column))
        
        rmses.append(rmse)
        max_errors.append(error)
        stds.append(std)
        
    return rmses, max_errors, stds
    
def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def split_string_floats(params):
    set_values = []
    set_types = []
    
    for param in params:
        values = []
        types = []
        
        data = param.split(" ")
        
        for d in data:
            if is_float(d):
                values.append(d)
            else:
                types.append(d)
        
        set_values.append(values)
        set_types.append(types)
    
    return set_values, set_types
        
def colors(set_types):
    color = []
    
    for i, types in enumerate(set_types):
        if i == 0:  #base
            color.append([0,0,1])
        elif '-g' in types:     #cpu
            color.append([1,0,0])
        else:   #gpu
            color.append([0,1,0])
    
    return color

def plot(input_file, output_file, params):
    pd.read_csv(input_file, header=None).T.to_csv(output_file, header=False, index=False)

    irradiance = pd.read_csv(output_file, header=None)

    if irradiance.shape[1] > len(params):
        print(irradiance.shape[1])
        print(len(params))
        
        irradiance = irradiance.iloc[:, :len(params)]

        print('WARNING: number of results higher than number of params')
    elif len(params) > irradiance.shape[1]:
        print(irradiance.shape[1])
        print(len(params))
        params = params[:irradiance.shape[1]]
        print('WARNING: number of params higher than number of results')
    
    times = irradiance.iloc[-1, :]
    
    # Remove the times from the data
    irradiance = irradiance.iloc[:-1, :]
    
    rmses, max_errors, stds = errors(irradiance)

    rmses = np.array(rmses)
    times = np.array(times)
    
    set_values, set_types = split_string_floats(params)
    
    num_of_vis_params = 5
    
    annotations = []
    for types, values in zip(set_types, set_values):
        annotation = ""
        for t, v in zip(types[:num_of_vis_params], values[:num_of_vis_params]):
            text = "{} {}".format(t, str(v))
            
            annotation += text
            annotation += '\n'
        
        annotations.append(annotation)
    
    fig,ax = plt.subplots()
    plt.xlabel('RMSE')
    plt.ylabel('Computation Time (s)')
    plt.title('RMSE\'s for parameter convergence test')
    
    c = colors(set_types)
    norm = plt.Normalize(1,4)
    
    sc = plt.scatter(rmses,times, c=c, norm=norm)
    
    plt.grid(True)
    
    annot = ax.annotate("", xy=(0,0), xytext=(20,-20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
    
    annot.set_visible(False)
    
    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "Index: {}\nRMSE: {}\nParams:\n{}".format(
                                "".join(list(map(str,ind["ind"]))[0]), 
                                "".join([str(round(rmses[n], 2)) for n in ind["ind"]][0]),
                                "".join([annotations[n] for n in ind["ind"]][0])
                                )
        annot.set_text(text)
        
        # annot.get_bbox_patch().set_facecolor(cmap(norm(indices[ind["ind"][0]])))
        # annot.get_bbox_patch().set_alpha(0.4)
    
    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
        
    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.show()
    
    ######
    fig,ax = plt.subplots()
    
    plt.xlabel('Max_error')
    plt.ylabel('Computation Time (s)')
    plt.title('Max error\'s for parameter convergence test')
    
    sc = plt.scatter(max_errors,times, c=c, norm=norm)
    
    plt.grid(True)
    
    annot.set_visible(False)
    
    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
    
    annot.set_visible(False)
    
    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "Index: {}\nRMSE: {}\nParams:\n{}".format(
                                "".join(list(map(str,ind["ind"]))[0]), 
                                "".join([str(round(rmses[n], 2)) for n in ind["ind"]][0]),
                                "".join([annotations[n] for n in ind["ind"]][0])
                                )
        annot.set_text(text)
        
        # annot.get_bbox_patch().set_facecolor(cmap(norm(indices[ind["ind"][0]])))
        # annot.get_bbox_patch().set_alpha(0.4)
    
    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            # else:
            #     if vis:
            #         annot.set_visible(False)
            #         fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect("motion_notify_event", hover)
    
    plt.show()
    
    
    
    
    ######
    fig,ax = plt.subplots()
    
    plt.xlabel('Standard_Deviation')
    plt.ylabel('Computation Time (s)')
    plt.title('Standard Deviation\'s for parameter convergence test')
    
    sc = plt.scatter(stds,times, c=c, norm=norm)
    
    plt.grid(True)
    
    annot.set_visible(False)
    
    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
    
    annot.set_visible(False)
    
    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "Index: {}\nRMSE: {}\nParams:\n{}".format(
                                "".join(list(map(str,ind["ind"]))[0]), 
                                "".join([str(round(rmses[n], 2)) for n in ind["ind"]][0]),
                                "".join([annotations[n] for n in ind["ind"]][0])
                                )
        annot.set_text(text)
        
        # annot.get_bbox_patch().set_facecolor(cmap(norm(indices[ind["ind"][0]])))
        # annot.get_bbox_patch().set_alpha(0.4)
    
    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            # else:
            #     if vis:
            #         annot.set_visible(False)
            #         fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect("motion_notify_event", hover)
    
    plt.show()
    

if __name__=='__main__':
    name, folder = 'renderfarm_test_model', 'C://Users//Job de Vogel//Desktop'

    print('Loading hbjson')
    model = load_hbjson(name, folder)
    print('Finished loading')

    # arguments = ['-ab 14 -aa 0 -ar 4096 -ad 16384 -as 8192 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
    #              '-ab 14 -aa 0 -ar 4096 -ad 16384 -as 8192 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g'
    #              '-ab 14 -aa 0 -ar 4096 -ad 16384 -as 8192 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g'
    #              '-ab 14 -aa 0 -ar 4096 -ad 16384 -as 8192 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g'
    #         ]

    arguments = ['-ab 14 -aa 0 -ar 4096 -ad 16384 -as 8192 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 3 -aa 0.4 -ar 8 -ad 32 -as 16 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 3 -aa 0.4 -ar 8 -ad 32 -as 16 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 3 -aa 0.2 -ar 16 -ad 64 -as 32 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 3 -aa 0.1 -ar 32 -ad 128 -as 64 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 3 -aa 0.05 -ar 64 -ad 256 -as 128 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 3 -aa 0 -ar 128 -ad 512 -as 256 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 3 -aa 0 -ar 256 -ad 1024 -as 512 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 3 -aa 0 -ar 512 -ad 2048 -as 1024 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 3 -aa 0 -ar 1024 -ad 4096 -as 2048 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 3 -aa 0 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 4 -aa 0.4 -ar 8 -ad 32 -as 16 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 4 -aa 0.2 -ar 16 -ad 64 -as 32 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 4 -aa 0.1 -ar 32 -ad 128 -as 64 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 4 -aa 0.05 -ar 64 -ad 256 -as 128 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 4 -aa 0 -ar 128 -ad 512 -as 256 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 4 -aa 0 -ar 256 -ad 1024 -as 512 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 4 -aa 0 -ar 512 -ad 2048 -as 1024 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 4 -aa 0 -ar 1024 -ad 4096 -as 2048 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 4 -aa 0 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 5 -aa 0.4 -ar 8 -ad 32 -as 16 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 5 -aa 0.2 -ar 16 -ad 64 -as 32 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 5 -aa 0.1 -ar 32 -ad 128 -as 64 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 5 -aa 0.05 -ar 64 -ad 256 -as 128 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 5 -aa 0 -ar 128 -ad 512 -as 256 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 5 -aa 0 -ar 256 -ad 1024 -as 512 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 5 -aa 0 -ar 512 -ad 2048 -as 1024 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 5 -aa 0 -ar 1024 -ad 4096 -as 2048 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 5 -aa 0 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 7 -aa 0.4 -ar 8 -ad 32 -as 16 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 7 -aa 0.2 -ar 16 -ad 64 -as 32 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 7 -aa 0.1 -ar 32 -ad 128 -as 64 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 7 -aa 0.05 -ar 64 -ad 256 -as 128 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 7 -aa 0 -ar 128 -ad 512 -as 256 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 7 -aa 0 -ar 256 -ad 1024 -as 512 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 7 -aa 0 -ar 512 -ad 2048 -as 1024 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 7 -aa 0 -ar 1024 -ad 4096 -as 2048 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 7 -aa 0 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 9 -aa 0.4 -ar 8 -ad 32 -as 16 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 9 -aa 0.2 -ar 16 -ad 64 -as 32 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 9 -aa 0.1 -ar 32 -ad 128 -as 64 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 9 -aa 0.05 -ar 64 -ad 256 -as 128 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 9 -aa 0 -ar 128 -ad 512 -as 256 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 9 -aa 0 -ar 256 -ad 1024 -as 512 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 9 -aa 0 -ar 512 -ad 2048 -as 1024 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 9 -aa 0 -ar 1024 -ad 4096 -as 2048 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 9 -aa 0 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 11 -aa 0.4 -ar 8 -ad 32 -as 16 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 11 -aa 0.2 -ar 16 -ad 64 -as 32 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 11 -aa 0.1 -ar 32 -ad 128 -as 64 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 11 -aa 0.05 -ar 64 -ad 256 -as 128 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 11 -aa 0 -ar 128 -ad 512 -as 256 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 11 -aa 0 -ar 256 -ad 1024 -as 512 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 11 -aa 0 -ar 512 -ad 2048 -as 1024 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 11 -aa 0 -ar 1024 -ad 4096 -as 2048 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 11 -aa 0 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 12 -aa 0.4 -ar 8 -ad 32 -as 16 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 12 -aa 0.2 -ar 16 -ad 64 -as 32 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 12 -aa 0.1 -ar 32 -ad 128 -as 64 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 12 -aa 0.05 -ar 64 -ad 256 -as 128 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 12 -aa 0 -ar 128 -ad 512 -as 256 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 12 -aa 0 -ar 256 -ad 1024 -as 512 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 12 -aa 0 -ar 512 -ad 2048 -as 1024 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 12 -aa 0 -ar 1024 -ad 4096 -as 2048 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
            '-ab 12 -aa 0 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.1 -w'
            '-ab 3 -aa 0.4 -ar 8 -ad 32 -as 16 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 3 -aa 0.4 -ar 8 -ad 32 -as 16 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 3 -aa 0.2 -ar 16 -ad 64 -as 32 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 3 -aa 0.1 -ar 32 -ad 128 -as 64 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 3 -aa 0.05 -ar 64 -ad 256 -as 128 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 3 -aa 0 -ar 128 -ad 512 -as 256 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 3 -aa 0 -ar 256 -ad 1024 -as 512 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 3 -aa 0 -ar 512 -ad 2048 -as 1024 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 3 -aa 0 -ar 1024 -ad 4096 -as 2048 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 3 -aa 0 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 4 -aa 0.4 -ar 8 -ad 32 -as 16 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 4 -aa 0.2 -ar 16 -ad 64 -as 32 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 4 -aa 0.1 -ar 32 -ad 128 -as 64 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 4 -aa 0.05 -ar 64 -ad 256 -as 128 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 4 -aa 0 -ar 128 -ad 512 -as 256 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 4 -aa 0 -ar 256 -ad 1024 -as 512 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 4 -aa 0 -ar 512 -ad 2048 -as 1024 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 4 -aa 0 -ar 1024 -ad 4096 -as 2048 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 4 -aa 0 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 5 -aa 0.4 -ar 8 -ad 32 -as 16 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 5 -aa 0.2 -ar 16 -ad 64 -as 32 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 5 -aa 0.1 -ar 32 -ad 128 -as 64 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 5 -aa 0.05 -ar 64 -ad 256 -as 128 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 5 -aa 0 -ar 128 -ad 512 -as 256 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 5 -aa 0 -ar 256 -ad 1024 -as 512 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 5 -aa 0 -ar 512 -ad 2048 -as 1024 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 5 -aa 0 -ar 1024 -ad 4096 -as 2048 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 5 -aa 0 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 7 -aa 0.4 -ar 8 -ad 32 -as 16 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 7 -aa 0.2 -ar 16 -ad 64 -as 32 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 7 -aa 0.1 -ar 32 -ad 128 -as 64 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 7 -aa 0.05 -ar 64 -ad 256 -as 128 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 7 -aa 0 -ar 128 -ad 512 -as 256 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 7 -aa 0 -ar 256 -ad 1024 -as 512 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 7 -aa 0 -ar 512 -ad 2048 -as 1024 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 7 -aa 0 -ar 1024 -ad 4096 -as 2048 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 7 -aa 0 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 9 -aa 0.4 -ar 8 -ad 32 -as 16 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 9 -aa 0.2 -ar 16 -ad 64 -as 32 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 9 -aa 0.1 -ar 32 -ad 128 -as 64 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 9 -aa 0.05 -ar 64 -ad 256 -as 128 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 9 -aa 0 -ar 128 -ad 512 -as 256 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 9 -aa 0 -ar 256 -ad 1024 -as 512 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 9 -aa 0 -ar 512 -ad 2048 -as 1024 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 9 -aa 0 -ar 1024 -ad 4096 -as 2048 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 9 -aa 0 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 11 -aa 0.4 -ar 8 -ad 32 -as 16 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 11 -aa 0.2 -ar 16 -ad 64 -as 32 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 11 -aa 0.1 -ar 32 -ad 128 -as 64 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 11 -aa 0.05 -ar 64 -ad 256 -as 128 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 11 -aa 0 -ar 128 -ad 512 -as 256 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 11 -aa 0 -ar 256 -ad 1024 -as 512 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 11 -aa 0 -ar 512 -ad 2048 -as 1024 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 11 -aa 0 -ar 1024 -ad 4096 -as 2048 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 11 -aa 0 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 12 -aa 0.4 -ar 8 -ad 32 -as 16 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 12 -aa 0.2 -ar 16 -ad 64 -as 32 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 12 -aa 0.1 -ar 32 -ad 128 -as 64 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 12 -aa 0.05 -ar 64 -ad 256 -as 128 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 12 -aa 0 -ar 128 -ad 512 -as 256 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 12 -aa 0 -ar 256 -ad 1024 -as 512 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 12 -aa 0 -ar 512 -ad 2048 -as 1024 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 12 -aa 0 -ar 1024 -ad 4096 -as 2048 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
            '-ab 12 -aa 0 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g'
            ]

    input_file = r"C:\\Users\\Job de Vogel\\Desktop\\Other\\results.csv"
    output_file = r"C:\\Users\\Job de Vogel\\Desktop\\Other\\results_flipped.csv"
    
    # input_file = r"C:\\Users\\Job de Vogel\\Desktop\\results.csv"
    # output_file = r"C:\\Users\\Job de Vogel\\Desktop\\results_flipped.csv"
    # par_convergence(model, arguments, input_file)
    plot(input_file, output_file, arguments)