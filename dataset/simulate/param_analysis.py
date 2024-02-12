import pandas as pd
import numpy as np
import sys
import math

import matplotlib.pyplot as plt

input_file = r"C:\\Users\\Job de Vogel\Desktop\\results.csv"
params_file = r"C:\\Users\\Job de Vogel\Desktop\\params.csv"
output_file = r"C:\\Users\\Job de Vogel\Desktop\\results_flipped.csv"

def errors(df):
    rmses = []
    max_errors = []
    for i in range(df.shape[1]):
        column_index = i

        selected_column = df.iloc[:-1, column_index]
        comparison_column = df.iloc[:-1, 0]
        
        squared_errors = (selected_column - comparison_column) ** 2
        mean = squared_errors.mean()
        rmse = math.sqrt(mean)  
        
        max_error = np.max(np.abs(selected_column - comparison_column))
        max_index = np.argmax(np.abs(selected_column - comparison_column))
        
        rmses.append(rmse)
        max_errors.append(max_error)
        
    return rmses, max_errors
    
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
        irradiance = irradiance.iloc[:, :len(params)]
        print('WARNING: number of results higher than number of params')
    elif len(params) > irradiance.shape[1]:
        params = params[:irradiance.shape[1]]
        print('WARNING: number of params higher than number of results')
    
    times = irradiance.iloc[-1, :]
    
    # Remove the times from the data
    irradiance = irradiance.iloc[:-1, :]
    
    rmses, max_errors = errors(irradiance)

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
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
        
    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.show()
    
arguments = ['-ab 3 -aa 0.4 -ar 8 -ad 32 -as 16 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
             '-ab 3 -aa 0.2 -ar 16 -ad 64 -as 32 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w',
             '-ab 3 -aa 0.1 -ar 32 -ad 128 -as 64 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
             '-ab 3 -aa 0.05 -ar 64 -ad 256 -as 128 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
             '-ab 3 -aa 0 -ar 128 -ad 512 -as 256 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
             '-ab 3 -aa 0 -ar 256 -ad 1024 -as 512 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
             '-ab 3 -aa 0 -ar 512 -ad 2048 -as 1024 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
             '-ab 3 -aa 0 -ar 1024 -ad 4096 -as 2048 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g',
             '-ab 3 -aa 0 -ar 2048 -ad 8192 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15 -w -g']

plot(input_file, output_file, arguments)