import pandas as pd
import numpy as np
import sys

import matplotlib.pyplot as plt

input_file = "C:\\Users\\Job de Vogel\Desktop\\results.csv"
params_file = "C:\\Users\\Job de Vogel\Desktop\\params.csv"
output_file = "C:\\Users\\Job de Vogel\Desktop\\results_flipped.csv"

pd.read_csv(input_file, header=None).T.to_csv(output_file, header=False, index=False)

df = pd.read_csv(output_file, header=None)

msres = []
max_errors = []

times = df.iloc[-1, :]

for i in range(df.shape[1]):
    column_index = i

    selected_column = df.iloc[:-1, column_index]
    last_column = df.iloc[:-1, -1]

    squared_errors = (selected_column - last_column) ** 2
    sqrt_errors = np.sqrt(squared_errors)

    msre = sqrt_errors.mean()
    
    max_error = np.max(np.abs(selected_column - last_column))
    max_index = np.argmax(np.abs(selected_column - last_column))
    
    msres.append(msre)
    max_errors.append(max_error)

# ! Add -g to the params file in accelerad test
params = pd.read_csv(params_file).iloc[:,:]

# Plotting
errors = np.array(msres)
times = np.array(times)

# annotation = np.array([str(error) for error in errors])
annotations = [params.iloc[i, 1:6].to_string(name=False, dtype=False) for i in range(len(params))]

indices = np.random.randint(1,5,size=len(df.index))

norm = plt.Normalize(1,4)

g = params.loc[:, 'g']

colors = np.zeros((len(params.index), 3))

for i, t in enumerate(g):
    if str(t) == '0':
        colors[i] = [0,1,0]
    else:
        colors[i] = [1,0,0]

fig,ax = plt.subplots()
plt.xlabel('MSRE')
plt.ylabel('Computation Time (s)')
plt.title('MSRE errors for parameter convergence test')

# sc = plt.scatter(errors,times,c=values, s=100, cmap=cmap, norm=norm)
sc = plt.scatter(errors,times, c=colors, norm=norm)

plt.grid(True)

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))

annot.set_visible(False)

def update_annot(ind):
    
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "Index: {}\nMSRE: {}\nParams:\n{}".format(
                            " ".join(list(map(str,ind["ind"]))), 
                            " ".join([str(round(errors[n], 2)) for n in ind["ind"]]),
                            " ".join([annotations[n] for n in ind["ind"]])
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

