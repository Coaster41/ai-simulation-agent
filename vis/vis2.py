import matplotlib.pyplot as plt
import matplotlib.patches as mplpatches
from typing import Optional
import numpy as np
 # argument parser
plt.style.use('BME163') # use bme163 style


data = {
    "cnn" : 1.52e-3,
    "fnn" : 3.13e-4,
    
    "linear reg" : 1.06e-3,
    "probabalistic": 7.3e-3,
    "simulator" : 1.27e4
}


figureWidth, figureHeight = 5, 3
panelWidth,  panelHeight  = 4, 2

normalizedArray = np.array([figureWidth, figureHeight, figureWidth, figureHeight])
panelData = np.divide(np.array([0.5, 0.5, panelWidth, panelHeight]), normalizedArray)


y_ticks = np.arange(0, 0.11, 0.01)
panel = plt.axes(panelData)
panel.set_yscale("log")
panel.set_xlim(0, 6)
panel.set_ylim(1e-5, 5e4)
panel.set_title("Prediction Time taken")
# panel.set_yticks(np.arange(0,0.11,0.01), 
#                  [(y_ticks[i] if i % 2 == 0 else "") for i in range(len(y_ticks))])
panel.set_xticks(np.arange(1,6,1), data.keys())

panel.set_xlabel("Model")
panel.set_ylabel("Time (s), using logscale")

def plot_rectangle(panel, width, height, x, y, color):
    panel.add_patch(
        mplpatches.Rectangle([x, y], width, height,
                             facecolor = color,
                             edgecolor = "black", 
                             linewidth = 2)
    )


# create the gradient
viridis_data = [(253/255, 231/255, 37/255),
                (94/255, 201/255, 98/255),
                (33/255, 145/255, 140/255),
                (59/255, 82/255, 139/255),
                (68/255, 1/255, 84/255)]

viridis_data.reverse()

# copied from assignment 1, but modified for overlappting
def generateGradient(gap, overlap, colors):
    # generate a gradient for a list of colors from using a gap
    # this gap is the range between gradients
    ncolors = len(colors) - 1 # subtrace one because for n colors, there are n - 1 ranges
    
    R = np.array([])
    G = np.array([])
    B = np.array([])


    for i in range(ncolors):
        R_add = np.linspace(colors[i][0], colors[i+1][0], gap)
        G_add = np.linspace(colors[i][1], colors[i+1][1], gap) 
        B_add = np.linspace(colors[i][2], colors[i+1][2], gap)   
        # take all but the last overlap elements  
        R = np.append(R[:-overlap], R_add)  
        G = np.append(G[:-overlap], G_add)
        B = np.append(B[:-overlap], B_add)


    return (R,G,B)   



viridis_gradient = generateGradient(26, 1, viridis_data)
# make a dictionary

viridis_values = np.linspace(0,100,4)

viridis_values = np.concatenate([viridis_values, np.array([20.0,])])
print(viridis_values)
print(np.array([50, ]))
data_keys = data.keys()
data_val = list(data.values())

viridis_dict = {}
for i in range(len(viridis_gradient[0])):
    viridis_dict[i] = (viridis_gradient[0][i],
                       viridis_gradient[1][i],
                       viridis_gradient[2][i])
    
for i in range(1,6,1):
    
    value = data_val[i -1]
    # hardcode the last bar
    color = viridis_dict[int(viridis_values[i - 1])] if i < 5 else (212/255, 52/255, 13/255)
    plot_rectangle(panel, 0.5, value, i - 0.25, 0, color)


plt.savefig("results_2.png",dpi=600) 

