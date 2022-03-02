import webbrowser as browser
import os
import re
from pathlib import Path
from collections import defaultdict


import imageio
import yaml
from pandas import read_csv
import plotly.express as px
import numpy as np
import cv2
from jinja2 import Environment, FileSystemLoader


# jinja config6
env = Environment(loader=FileSystemLoader("./"))

# read parameters
data_path = "data/"
params_paths = [
    path for path in Path("data").rglob("*.yml")
]  # just in case there are mulltiple parameter files
for path in params_paths:
    with open(path, "r") as f:
        mcmicro_params = yaml.safe_load(f)

file_extensions = list(mcmicro_params["singleFormats"].keys()) + list(
    mcmicro_params["multiFormats"].keys()
)

sample_name = mcmicro_params["sampleName"]

regex = re.compile(r"^.*?:|\,.*$|'")
modules = {
    key: re.sub(regex, "", str(value[0]))
    for key, value in mcmicro_params.items()
    if "module" in key
}
segmentation_path = Path(
    data_path
    + sample_name
    + "/segmentation/"
    + modules["modulesPM"]
    + "-"
    + sample_name  # Can you run multiple segmetations algorithms?
)

registration_path = Path(data_path + sample_name + "/registration")
quantification_path = Path(data_path + sample_name + "/quantification")

max_img_size = (
    104857600  # 100MB # the actual html ends up being much smaller, how? Plotly magic??
)

scale_factor = 1

# load quantification csv

quantification = [
    read_csv(csv)
    for csv in quantification_path.glob(modules["modulesPM"] + "-" + sample_name + "*")
    if csv.suffix == ".csv"
]


# loading images
whole_image = [
    imageio.imread(image)
    for image in registration_path.rglob(sample_name + "*")
    if image.suffix in file_extensions
][0]

segmentation_cells = [
    imageio.imread(image)
    for image in segmentation_path.rglob("cell*")
    if image.suffix in file_extensions
]

segmentation_nuclei = [
    imageio.imread(image)
    for image in segmentation_path.rglob("nuclei*")
    if image.suffix in file_extensions
]


# assuming only a single registration image can exist

num_cells = np.amax(segmentation_cells)
num_nuclei = np.amax(segmentation_nuclei)

# getting parameters for downscaling
image_file_size = [
    os.path.getsize(image) for image in Path(registration_path).rglob(sample_name + "*")
][
    0
]  # assuming only a single registration image can exist

image_width = whole_image.shape[0]
image_height = whole_image.shape[1]
image_size = whole_image.size
image_factor = image_file_size / image_size
scaled_file_size = image_file_size
scale_factor = 1

# calculate scaling factor
while scaled_file_size > max_img_size:
    scale_factor = scale_factor * 0.98
    print("New scale factor: " + str(scale_factor))
    scaled_width = image_width * scale_factor
    scaled_height = image_height * scale_factor
    scaled_file_size = scaled_width * scaled_height * image_factor
    print("New file size: " + str(scaled_file_size))
    print("New image size " + str(scaled_width) + "||" + str(scaled_height))

# scale image

scaled_image_size = (int(scaled_width), int(scaled_height))
whole_image_resized = cv2.resize(whole_image, scaled_image_size)
# read the quantification files

# retrieve the number of cells from the segementation -> maximum pixel intensitiy


# crop the brightest spot as a high res sample
side_length = 500
blurry_image = cv2.GaussianBlur(whole_image, (5, 5), 0)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blurry_image)
p1 = (int(maxLoc[0] - (side_length / 2)), int(maxLoc[1] - (side_length / 2)))
p2 = (int(maxLoc[0] + (side_length / 2)), int(maxLoc[1] + (side_length / 2)))
high_res_crop = whole_image[p1[1] : p2[1], p1[0] : p2[0]]
# cv2.namedWindow("image", cv2.WINDOW_KEEPRATIO)

# cv2.imshow("image", high_res_crop)
# cv2.resizeWindow("image", 400, 400)
# cv2.waitKey(0)
# Determine where to split up the image


# Create plotly plot for the resized image

fig1 = px.imshow(whole_image_resized)
fig2 = px.imshow(high_res_crop)
fig1.write_html(
    "html/figures/zoomable_image.html",
    config={"doubleClick": "reset", "scrollZoom": True, "displayModeBar": True},
)

fig2.write_html(
    "html/figures/high_res.html",
    config={"doubleClick": "reset", "scrollZoom": False, "displayModeBar": False},
)

# pass parameters to html
html_parameters = {"num_cells": num_cells, "num_nuclei": num_nuclei}

# generate html
TEMPLATE_FILE = "html/template.j2"
template = env.get_template(TEMPLATE_FILE)
report_html_code = template.render(html_parameters=html_parameters)

# write the report file
with open("report.html", "w") as fh:
    fh.write(report_html_code)

# open it in the default browser
browser.open("report.html")
