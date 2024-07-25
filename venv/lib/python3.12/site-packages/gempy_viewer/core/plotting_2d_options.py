from dataclasses import dataclass
from matplotlib import colors as mcolors, cm


@dataclass
class Plotting2DOptions:
    cmap: mcolors.ListedColormap = cm.viridis
    norm: mcolors.Normalize = mcolors.Normalize(vmin=0, vmax=1)
    
    def __post_init__(self):
        self.cmap.set_under(color="white")
        self.cmap.set_over(color="black")
        
        self.norm.clip = True
        self.norm.autoscale_None = False
        

    # ? Do I need _custom_colormap field