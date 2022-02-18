
#df = pd.read_csv("/mnt/c/Users/draco/Documents/schapiro_lab/exemplar-001/quantificatimageion/unmicst-exemplar-001_cell.csv") #TODO dynamic path


#import stuff

import webbrowser as browser
import plotly.express as px


import pandas as pd
import numpy as np
import imageio

from jinja2 import Environment, FileSystemLoader
from pathlib import Path


#jinja config
env = Environment(loader=FileSystemLoader('templates'))
print("templates loaded")

# reading data
quantification = pd.read_csv(Path("./data/quantification/unmicst-exemplar-001_cell.csv")) #TODO don't hardcode the filepaths

segmentation_cells = imageio.imread(Path("./data/segmentation/unmicst-exemplar-001/cell.ome.tif"))
print(segmentation_cells)
segmentation_nuclei = imageio.imread(Path("./data/segmentation/unmicst-exemplar-001/nuclei.ome.tif"))
whole_ = imageio.imread(Path("./data/registration/exemplar-001.ome.tif"))

num_cells = np.amax(segmentation_cells)
num_nuclei = np.amax(segmentation_nuclei) 

print(quantification.head(10))
print(num_cells)
print(num_nuclei)

html_parameters = {
    "num_cells": num_cells,
    "num_nuclei": num_nuclei

}

#generate html
TEMPLATE_FILE = "template.j2"
template = env.get_template(TEMPLATE_FILE)
report_html_code = template.render(html_parameters = html_parameters)

print(report_html_code)

with open("report.html", "w") as fh:
    fh.write(report_html_code)