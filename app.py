# IMPORTS
"""This script reads output data from MCMICRO and generates a report in the form of an html file."""


import webbrowser as browser
import os
import re
from pathlib import Path
import timeit
import imageio

import yaml
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from skimage import (
    measure,
    segmentation,
)  # TODO do you need both skimage and cv2??
from cv2 import GaussianBlur, resize
from jinja2 import Environment, FileSystemLoader


# SETUP


# starting a timer to measure how the performance of the script
start = timeit.default_timer()
print("[report] Generating report")
# jinja config
env = Environment(loader=FileSystemLoader("./"))

# set default parameter for plotly plots
default_plot_layout = {
    "height": 500,
    "width": 500,
    "autosize": False,
    "paper_bgcolor": "rgba(0,0,0,0)",
}


# FUNCTIONS


def read_csv_or_image_data(path: Path, pattern: str, file_ext, recursive=True):
    """Read matching data from the given path

    Args:
        path (Path): Path from where files should be read
        pattern (str): File names to look for
        file_types (_type_): Files types to look for, currently accepts images and csv
    """
    try:
        if ".csv" in file_ext and recursive == True:
            data = [
                pd.read_csv(csv) for csv in path.rglob(pattern) if csv.suffix == ".csv"
            ]

        if ".csv" in file_ext and recursive == False:
            data = [
                pd.read_csv(csv) for csv in path.glob(pattern) if csv.suffix == ".csv"
            ]

        if ".csv" not in file_ext and recursive == True:
            data = [
                imageio.imread(image)
                for image in path.rglob(pattern)
                if image.suffix in file_ext
            ]

        if ".csv" not in file_ext and recursive == False:
            data = [
                imageio.imread(image)
                for image in path.glob(pattern)
                if image.suffix in file_ext
            ]
        if data == []:
            raise FileNotFoundError
        else:
            return data
    except FileNotFoundError:
        print(
            f"[report] ERROR: No valid files found in {path}. Please check that the path exists and contains files of the types {file_ext+ ['.csv']}."
        )
        quit()


def plot_core_overlay(image: np.array, mask: np.array) -> go.Figure:
    """Plot an overlay of TMA cores.

    Args:
        image (Array): the image that should be overlayed, as a numpy array
        mask (Array): the image that should be used as a mask, as a numpy array

    Returns:
        Figure: a plotly figure containing the core overlay
    """
    # clean the mask
    mask_cleared = segmentation.clear_border(mask)
    # prepare the plotly figure
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=image, showscale=False, colorscale="Viridis"))
    # apply default layout config
    fig.update_layout(default_plot_layout)
    # axis need to be reversed for the image not to be upside down
    fig.update_yaxes(autorange="reversed")

    # measure the properties of each core
    props = measure.regionprops(mask_cleared.astype(int), image)
    properties = ["area", "perimeter", "intensity_mean"]
    # determine the amount of cores
    min = int(mask[np.nonzero(mask_cleared)].min())
    max = int(mask[np.nonzero(mask_cleared)].max())
    # For each label, add a filled scatter trace for its contour,
    # and display the properties of the label in the hover of this trace.
    for i in range(min, max + 1):
        try:
            cells_per_core = quantification[i - 1]["CellID"].max()
        except IndexError:
            if i <= max:
                cells_per_core = "NOT FOUND"
                print(
                    f"[report] Can not find quantification data for core #{i}. Please check if /quantification contains a valid .csv file for each core."
                )
            else:
                pass
        try:

            # Find contours
            y, x = measure.find_contours(mask_cleared == i, 0.5)[0].T
            # retrieve corresponding properties
            hoverinfo = f"<b>Number of cells: {cells_per_core}</b><br>"
            for prop_name in properties:
                hoverinfo += f"<b>{prop_name} (downscaled): {getattr(props[index], prop_name):.2f}</b><br>"
            # plot the core
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name=f"Core #{i}",
                    mode="lines",
                    fill="toself",
                    showlegend=True,
                    hovertemplate=hoverinfo,
                    hoveron="points+fills",
                )
            )
        except IndexError:
            print(
                f"Can not find contour for core {i}. Please check if /dearray/masks contains a valid mask for each core."
            )

    # return plot
    return fig


def plot_cell_contours_plotly(
    im: np.array,
    mask: np.array,
    data: pd.DataFrame,
    marker: str,
    cutoff: int,
    show_bg: bool = True,
    color_above_cutoff: str = "orangered",
    color_below_cutoff: str = "green",
) -> go.Figure:
    """Plot a segmentation overlay of cells

    Args:
        im (np.array): an image to plot a cell overlay on
        mask (np.array): a valid segmenation mask
        data (pd.DataFrame): quantification data provided by MCMICRO
        marker (str): a marker, must be quantified in the quantification data
        cutoff (int): marker intensity cutoff for plotting cells
        show_bg (bool, optional): Wheter to show the image as a background for the plot. Defaults to True.
        color_above_cutoff (str, optional): The color of cell outlines with the marker intensity above the cutoff. Defaults to "lightblue".
        color_below_cutoff (str, optional): The color of cell outlines with the marker intensity below the cutoff. Defaults to "red".

    Returns:
        Figure: a plotly figure containing the cells outlines
    """
    # Reset index so the CellID is the index
    data = data.set_index("CellID")

    # Cell
    flag = data[marker].apply(lambda x: marker if x >= cutoff else "")
    color = data[marker].apply(
        lambda x: color_above_cutoff if x >= cutoff else color_below_cutoff
    )

    # Get range of cells
    min = mask[np.nonzero(mask)].min()
    max = mask[np.nonzero(mask)].max()

    # Plot masked marker
    if show_bg == False:
        # remove the rest of the image
        fig = px.imshow(im * mask.astype(bool), color_continuous_scale="Viridis")
    else:
        # show whole image
        fig = px.imshow(im, color_continuous_scale="Viridis")
    error_counter = 0
    fig.update_layout(default_plot_layout)
    # Plot contours each cell at a time
    for i in range(min, max + 1):
        try:
            # Find contours
            y, x = measure.find_contours(mask == i, 0.8)[0].T

            #
            fig.add_trace(
                go.Scattergl(
                    x=x,
                    y=y,
                    mode="lines",
                    showlegend=False,
                    opacity=0.5,
                    text=flag[i],
                    line=dict(color=color[i]),
                )
            )
        except IndexError:
            # count invalid mask indices
            error_counter += 1

    # return plot
    print(
        f"[report] Can not find contours for {error_counter} cells. Please check if /segmentation contains a valid cell segmentation mask."
    )
    return fig


def overlay_images_at_centroid(bg: np.array, fg: np.array, cen_y: float, cen_x: float):
    """Overlay two image arrays at a center point

    Args:
        bg (np.array): Background image as array
        fg (np.array): Foreground overlay as array
        cen_y (float): y postion of the central point
        cen_x (float): x position of the central point

    Returns:
        np.array: Overlayed image as array
    """

    h1, w1 = fg.shape[:2]
    y_offset = max(
        0, cen_y - (h1 // 2)
    )  # calculate the distance from the center point to the outer edge of the image overlay, if it's < 0 set it to 0 using max to compare it to 0
    x_offset = max(0, cen_x - (w1 // 2))

    overlay_bounds_y = bg[y_offset : y_offset + h1, x_offset : x_offset + w1].shape[
        0
    ]  # subset the background array with the offset distance, take it's shape
    overlay_bounds_x = bg[y_offset : y_offset + h1, x_offset : x_offset + w1].shape[
        1
    ]  # we will use this ensure that the offset distance does not "leave" the bounds of the array, causing errors

    bg[
        y_offset : y_offset + h1, x_offset : x_offset + w1
    ] = fg[  # subset the overlay with the bounds to make sure we do not try overlaying outside of the background array
        0:overlay_bounds_y, 0:overlay_bounds_x
    ]
    return bg


# READING/SETTING PARAMETERS AND FILES

print("[report] Reading MCMICRO output")
# set parameter for max image size, used for dynamic downscaling later on
max_image_size = (
    104857600  # 100MB # the actual html ends up being much smaller, how? Plotly magic??
)
# set inital scale factor for dynamic downscaling
scale_factor = 1

# y/x of the high res sample
side_length = 500

# find MCMICRO parameter yaml file
data_path = "data/"
params_paths = [
    path for path in Path("data").rglob("*.yml")
]  # globbing just in case there are mulltiple parameter files

# open the parameter yaml

if params_paths == []:
    print(
        "[report] ERROR: MCMICRO parameter file not found or could not be read. Please check /data/qc for a valid params.yml and rerun the script."
    )
    quit()

for path in params_paths:
    with open(path, "r") as f:
        mcmicro_params = yaml.safe_load(f)


# read valid file extensions from parameter file
file_extensions = list(mcmicro_params["singleFormats"].keys()) + list(
    mcmicro_params["multiFormats"].keys()
)

# read the name of the image sample
sample_name = mcmicro_params["sampleName"]

# read the MCMICRO modules that were run from the parameter yaml
regex = re.compile(r"^.*?:|\,.*$|'|graph")  # regex for cleaning up the modules
# remove unnessecary parts from the module string
modules = {
    key: re.sub(regex, "", str(value[0]))
    for key, value in mcmicro_params.items()
    if "module" in key
}

# assemble path to segmentation files
segmentation_path = Path(
    data_path
    + sample_name
    + "/segmentation/"
    + modules["modulesPM"]
    + "-"
    + sample_name  # Can you run multiple segmetations algorithms?
)
# assemble path to registration files
registration_path = Path(data_path + sample_name + "/registration")

# assemble path to quantification files
quantification_path = Path(data_path + sample_name + "/quantification")

# assemble path to dearrayed cores + corresponding files
cores_path = Path(data_path + sample_name + "/dearray/")
core_masks_path = Path(data_path + sample_name + "/dearray/masks")
core_centroid_path = Path(data_path + sample_name + "/qc/coreo/centroidsY-X.txt")

if cores_path.is_dir == True:
    print("[report] dearray folder detected. Assuming TMA data.")
    tma_mode = True
else:
    tma_mode = False

# load quantification csv(s)
quantification = read_csv_or_image_data(
    quantification_path, modules["modulesPM"] + "-" + sample_name + "*", ".csv"
)

# loading images TODO Can there be multiple registration images?
try:
    whole_image = read_csv_or_image_data(
        registration_path, sample_name + "*", file_extensions
    )[0]
except IndexError:
    print(
        "[report] ERROR: Registration image not found or could not be read. Please check /data/registration for a valid image and rerun the script."
    )
    quit()
# load segementation masks
segmentation_cells = read_csv_or_image_data(segmentation_path, "cell*", file_extensions)

segmentation_nuclei = read_csv_or_image_data(
    segmentation_path, "nuclei*", file_extensions
)

# read coordinates of the TMA core centroids and format them into an array
with open(core_centroid_path) as f:
    line = f.readlines()
    core_centroids = [string.split() for string in line]

core_centroids = [
    [int(float(coordinate)) for coordinate in coordinate_pair]
    for coordinate_pair in core_centroids
]

core_centroids = np.array(core_centroids)


# PROCESSING QUANTIFICATION DATA


# determine number of cores
num_cores = len(core_centroids)

# determine number of cells from segmentation masks
num_cells = np.amax(segmentation_cells)
num_nuclei = np.amax(segmentation_nuclei)


# REASSEMBLING CORE MASK


# read the single core files
cores = read_csv_or_image_data(cores_path, "*", file_extensions, recursive=False)
# read the single core masks
single_core_masks = read_csv_or_image_data(
    core_masks_path, "*", file_extensions, recursive=False
)
# scale the (smaller) masks to the size of the actual cores
for index, mask in enumerate(single_core_masks):
    single_core_masks[index] = resize(mask, cores[index].shape)

# prepare an empty image to fill with core masks in the next steps
whole_core_mask = np.zeros(whole_image.shape)

# make each core mask distinct by assigning a specific pixel intensity to it, similar to the cell segmentation masks
for index, mask in enumerate(single_core_masks):
    mask[mask > 0] = index + 1

# stich the mulitple single core masks together to one large core mask
for index, core_centroid in enumerate(core_centroids):
    whole_core_mask = overlay_images_at_centroid(
        whole_core_mask,
        single_core_masks[index],
        core_centroids[index][0],
        core_centroids[index][1],
    )


# SCALING IMAGE


# getting parameters for downscaling
image_file_size = [
    os.path.getsize(image) for image in Path(registration_path).rglob(sample_name + "*")
][
    0
]  # assuming only a single registration image can exist TODO account for multiple ones?

# getting image dimension + size
image_width = whole_image.shape[0]
image_height = whole_image.shape[1]
image_size = whole_image.size

# file size per pixel
image_factor = image_file_size / image_size

# initialize variables for downscaling
scaled_file_size = image_file_size
scale_factor = 1

# calculate scaling factor to match maximum file size
while scaled_file_size > max_image_size:
    scale_factor = scale_factor * 0.98
    scaled_width = image_width * scale_factor
    scaled_height = image_height * scale_factor
    scaled_file_size = scaled_width * scaled_height * image_factor


# scale images and masks
print(f"[report] Scaling image to {int(scaled_width)}x{int(scaled_height)} pixels")
scaled_image_size = (int(scaled_width), int(scaled_height))
whole_image_resized = resize(whole_image, scaled_image_size)
whole_core_mask_resized = resize(whole_core_mask, scaled_image_size)


# HIGH RES SAMPLE


blurry_image = GaussianBlur(
    whole_image, (5, 5), 0
)  # blurring makes it less susceptible to outlier bright pixels

# find the maximum intensity values in the image, get its coordinates
maxLoc = np.unravel_index(np.argmax(blurry_image, axis=None), blurry_image.shape)

# calculate the corners of a square around maxLoc
p1 = (int(maxLoc[1] - (side_length / 2)), int(maxLoc[0] - (side_length / 2)))
p2 = (int(maxLoc[1] + (side_length / 2)), int(maxLoc[0] + (side_length / 2)))
# TODO account for multiple images
# subset the images + masks with the corner points
high_res_crop = whole_image[p1[1] : p2[1], p1[0] : p2[0]]
high_res_segmentation_mask_cells = segmentation_cells[0][p1[1] : p2[1], p1[0] : p2[0]]
high_res_segmentation_mask_nuclei = segmentation_nuclei[0][p1[1] : p2[1], p1[0] : p2[0]]


# PLOTTING


# prepare overview figure
Overview_plot = go.Figure()
Overview_plot.add_trace(go.Heatmap(z=whole_image_resized))

# reverse the y axis, so the image is not upside dowsn
Overview_plot.update_yaxes(autorange="reversed")

# add dropdown
Overview_plot.update_layout(
    default_plot_layout,
    updatemenus=[
        dict(
            buttons=list(
                [
                    # Button 1: "Overview"
                    dict(
                        label="Overview",
                        method="update",
                        # new data to display when the button is chosen
                        args=[{"z": [whole_image_resized]}],
                    ),
                    # Button 2: "High res sample"
                    dict(
                        label="High res sample",
                        method="update",
                        # new data to display when the button is chosen
                        args=[
                            {
                                "z": [high_res_crop],
                            }
                        ],
                    ),
                ]
            ),
            # Button layout
            direction="down",
            # large padding to the left so it looks better on the page
            pad={
                "l": 250,
            },
            showactive=True,
            x=0,
            xanchor="left",
            y=1.2,
            yanchor="top",
            bgcolor="white",
        )
    ],
)


# TMA
if tma_mode:
    TMA_plot = plot_core_overlay(whole_image_resized, whole_core_mask_resized)
    TMA_text = "Image"
else:
    TMA_text = (
        "TMA mode off or no TMA provided. Run MCMICRO with --tma to process TMA data."
    )

# Segmentation
segmentation_plot = plot_cell_contours_plotly(
    high_res_crop, high_res_segmentation_mask_cells, quantification[0], "DNA_1", 0
)


# ASSEMBLE/GENERATE HTML


# Overview
Overview_html = Overview_plot.to_html(
    config={"doubleClick": "reset", "scrollZoom": True, "displayModeBar": False},
    full_html=False,
)

# TMA
if tma_mode:
    TMA_html = TMA_plot.to_html(
        config={"doubleClick": "reset", "scrollZoom": True, "displayModeBar": False},
        full_html=False,
    )
else:
    TMA_html = ""

# Segmentation
segmentation_html = segmentation_plot.to_html(
    config={
        "doubleClick": "reset",
        "scrollZoom": True,
        "displayModeBar": False,
    },
    full_html=False,
)

# parameters for the jinja template
html_parameters = {
    "num_cores": num_cores,
    "num_cells": num_cells,
    "num_nuclei": num_nuclei,
    "Overview": Overview_html,
    "TMA": TMA_html,
    "Segmentation": segmentation_html,
    "TMA_text": TMA_text,
}

# read the jinja template
TEMPLATE_FILE = "html/template.html"
template = env.get_template(TEMPLATE_FILE)

# pass the parameters and generate the html
report_html_code = template.render(html_parameters=html_parameters)

# write the report file
with open("report.html", "w") as fh:
    fh.write(report_html_code)

# open it in the default browser
browser.open("report.html")

# stop the timer
stop = timeit.default_timer()
elapsed = "{:.2f}".format(stop - start)
print(f"[report] Report generated in {elapsed} seconds")
