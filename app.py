#import stuff


import webbrowser as browser
import os
import re
from pathlib import Path
from collections import defaultdict

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import cv2


from jinja2 import Environment, FileSystemLoader



#jinja config
env = Environment(loader=FileSystemLoader('./'))
print("templates loaded")

#parameters
file_extensions = [".tif", ".tiff", ".csv"]
datapath = Path("./data")

max_img_size = 20971520 #20MB # not precis
scale_factor = 1

# TODO SEE IF THERE IS A DIFFERENCE WITH MULTIPLE IMAGES, MAY NEED TO REDO THIS

# assemble a dictionary of paths to the images

all_images = [str(p) for p in datapath.rglob('*') if p.suffix in file_extensions] # retrieve the paths for the images
images_no_work_folder = [i for i in all_images if not "/work" in i]
#print(images_no_work_folder)
categories = [str(Path(p).parent).replace("data/", "") for p in images_no_work_folder] # retrieve the paths for the "categories"
categories = [re.sub(r'^.*?/', '', c) for c in categories] #cleaning the path


combined_list = list(zip(categories, all_images)) # order them into pairs
image_path_dict = defaultdict(list) #arrange and populate a dictionary 
for key, value in combined_list:
    image_path_dict[key].append(value)
#scale down huge images by calculating a scalefactor based on their filesize
path_to_whole_img= image_path_dict["registration"]

for file in path_to_whole_img: #TODO maybe add check for filetype if nessecary?
    
    new_size = os.path.getsize(file)
    print("Size: " + str(new_size))
    while new_size > max_img_size:  
        scale_factor = scale_factor*0.98 # reduce by two percent each loop, needs testing?
        print("New scale factor: " + str(scale_factor))
        new_size = new_size*scale_factor
        print("Size: " + str(new_size))
    

#loading images
#segmentation_cells = imageio.imread(img_path_dict["segmentation/unmicst-exemplar-001"][0])
#segmentation_nuclei = imageio.imread(img_path_dict["segmentation/unmicst-exemplar-001"][1])
#os.chdir("data/exemplar-002/registration/")
whole_image = cv2.imread("data/exemplar-002/registration/exemplar-002.ome.tif")
#print[whole_image[:10]]
print(image_path_dict["quantification"])
quantification = pd.read_csv(image_path_dict["quantification"][0]) #

fig = go.Figure(go.Image(z=whole_image))
fig.show()



fig.write_html("html/figures/zoomable_image.html", config={'doubleClick': 'reset', 'scrollZoom': True, 'displayModeBar': True})


num_cells = 0 #np.amax(segmentation_cells)
num_nuclei = 0 #np.amax(segmentation_nuclei) 

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