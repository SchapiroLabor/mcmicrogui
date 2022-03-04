from turtle import color
import webbrowser as browser
import os
import re
from pathlib import Path


import imageio
import yaml
from pandas import read_csv
import plotly
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from skimage import (
    filters,
    measure,
    color,
    segmentation,
)  # TODO do you need both skimage and cv2??
from cv2 import GaussianBlur, resize, cvtColor, COLOR_BGR2GRAY

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

regex = re.compile(r"^.*?:|\,.*$|'|graph")  # regex for cleaning up the modules
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
cores_path = Path(data_path + sample_name + "/qc/" + modules["moduleDearray"])

max_image_size = (
    104857600  # 100MB # the actual html ends up being much smaller, how? Plotly magic??
)

scale_factor = 1

# helper function for loading data


def read_csv_or_image_data(path: Path, pattern: str, file_ext):
    """Reads data from the given paths

    Args:
        path (Path): Path from where files should be read
        pattern (str): File names to look for
        file_types (_type_): Files types to look for, currently accepts images and csv
    """
    if ".csv" in file_ext:
        data = [read_csv(csv) for csv in path.rglob(pattern) if csv.suffix == ".csv"]

    else:
        data = [
            imageio.imread(image)
            for image in path.rglob(pattern)
            if image.suffix in file_ext
        ]
    return data


# load quantification csv
quantification = read_csv_or_image_data(
    quantification_path, modules["modulesPM"] + "-" + sample_name + "*", ".csv"
)


# loading images TODO account for multiple files
whole_image = read_csv_or_image_data(
    registration_path, sample_name + "*", file_extensions
)[0]

segmentation_cells = read_csv_or_image_data(segmentation_path, "cell*", file_extensions)

segmentation_nuclei = read_csv_or_image_data(
    segmentation_path, "nuclei*", file_extensions
)
cores = read_csv_or_image_data(cores_path, "*", ".tif")[0]


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
while scaled_file_size > max_image_size:
    scale_factor = scale_factor * 0.98
    print("New scale factor: " + str(scale_factor))
    scaled_width = image_width * scale_factor
    scaled_height = image_height * scale_factor
    scaled_file_size = scaled_width * scaled_height * image_factor
    print("New file size: " + str(scaled_file_size))
    print("New image size " + str(scaled_width) + "||" + str(scaled_height))

# scale image

scaled_image_size = (int(scaled_width), int(scaled_height))
whole_image_resized = resize(whole_image, scaled_image_size)
cores_resized = resize(cores, scaled_image_size)

side_length = 500
blurry_image = GaussianBlur(
    whole_image, (5, 5), 0
)  # blurring makes it less susceptible to outlier bright pixels

maxLoc = np.unravel_index(np.argmax(blurry_image, axis=None), blurry_image.shape)
p1 = (int(maxLoc[1] - (side_length / 2)), int(maxLoc[0] - (side_length / 2)))
p2 = (int(maxLoc[1] + (side_length / 2)), int(maxLoc[0] + (side_length / 2)))
high_res_crop = whole_image[p1[1] : p2[1], p1[0] : p2[0]]
high_res_segementation_mask = segmentation_cells[0][p1[1] : p2[1], p1[0] : p2[0]]

high_res_segementation_mask_cleared = segmentation.clear_border(
    high_res_segementation_mask
)
high_res_segementation_mask_label = measure.label(high_res_segementation_mask_cleared)

segmentation_label_overlay = color.label2rgb(
    high_res_segementation_mask_label, image=high_res_crop, bg_label=0
)

fig3 = px.imshow(segmentation_label_overlay)

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

fig3.write_html(
    "html/figures/segmentation.html",
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
