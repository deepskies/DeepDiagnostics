import numpy as np 
import matplotlib as mpl

def get_hex_colors(n_colors:int, colorway:str): 
    cmap = mpl.pyplot.get_cmap(colorway)
    hex_colors = []
    arr=np.linspace(0, 1, n_colors)
    for hit in arr: 
        hex_colors.append(mpl.colors.rgb2hex(cmap(hit)))

    return hex_colors