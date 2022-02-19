
#df = pd.read_csv("/mnt/c/Users/draco/Documents/schapiro_lab/exemplar-001/quantificatcv2n/unmicst-exemplar-001_cell.csv") #TODO dynamic path


#import stuff

import webbrowser as browser

from pathlib import Path
from collections import defaultdict
from PIL import Image

import pandas as pd
import numpy as np
import imageio
import plotly.graph_objects as go

from jinja2 import Environment, FileSystemLoader


#jinja config
env = Environment(loader=FileSystemLoader('./'))
print("templates loaded")

#parameters
file_extensions = [".tif", ".tiff"]
mainpath = Path("./")
print(mainpath)

# TODO SEE IF THERE IS A DIFFERENCE WITH MULTIPLE IMAGES, MAY NEED TO REDO THIS

# assemble a dictionary of paths to the images
images = [str(p) for p in mainpath.rglob('*') if p.suffix in file_extensions] # retrieve the paths for the images
categories = [str(p.parent).replace("data/","") for p in mainpath.rglob('*') if p.suffix in file_extensions] # retrieve the paths for the "categories"

combined_list = list(zip(categories, images)) # order them into pairs
img_path_dict = defaultdict(list) #arrange and populate a dictionary  

for key, value in combined_list:
    img_path_dict[key].append(value)

#loading images
segmentation_cells = imageio.imread(img_path_dict["segmentation/unmicst-exemplar-001"][0])
segmentation_nuclei = imageio.imread(img_path_dict["segmentation/unmicst-exemplar-001"][1])
whole_image = imageio.imread(("data/registration/exemplar-001.ome.tif")) # doesn't work with imageio)
whole_image_pil = Image.fromarray(whole_image)
quantification = pd.read_csv("data/quantification/unmicst-exemplar-001_cell.csv")


num_cells = np.amax(segmentation_cells)
num_nuclei = np.amax(segmentation_nuclei) 

# Create figure
fig = go.Figure()

# Constants
img_width = 2509
img_height = 3138
scale_factor = 1

# Add invisible scatter trace.
# This trace is added to help the autoresize logic work.
fig.add_trace(
    go.Scatter(
        x=[0, img_width * scale_factor],
        y=[0, img_height * scale_factor],
        mode="markers",
        marker_opacity=0
    )
)

# Configure axes
fig.update_xaxes(
    visible=True,
    range=[0, img_width * scale_factor]
)

fig.update_yaxes(
    visible=True,
    range=[0, img_width * scale_factor],
    # the scaleanchor attribute ensures that the aspect ratio stays constant
    scaleanchor="x"
)

# Add image
fig.add_layout_image(
    dict(
        x=0,
        sizex=img_width * scale_factor,
        y=img_height * scale_factor,
        sizey=img_height * scale_factor,
        xref="x",
        yref="y",
        opacity=1.0,
        layer="below",
        sizing="stretch",
        source=whole_image_pil)
)

# Configure other layout
fig.update_layout(
    width=400,
    height=400,
    margin={"l": 0, "r": 0, "t": 0, "b": 0},
)

# Disable the autosize on double click because it adds unwanted margins around the image
# More detail: https://plotly.com/python/configuration-options/
fig.write_html("html/figures/zoomable_image.html", config={'doubleClick': 'reset', 'scrollZoom': True, 'displayModeBar': True})

html_parameters = {
    "num_cells": num_cells,
    "num_nuclei": num_nuclei
    
}

#generate html
TEMPLATE_FILE = "html/template.j2"
template = env.get_template(TEMPLATE_FILE)
report_html_code = template.render(html_parameters = html_parameters)

print(report_html_code)

with open("report.html", "w") as fh:
    fh.write(report_html_code)

browser.open("report.html")