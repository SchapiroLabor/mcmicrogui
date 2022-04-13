# IMPORTS
"""This script reads output data from MCMICRO and generates a report in the form of an html file."""

import webbrowser as browser
import os
import re
from pathlib import Path
from glob import glob
import timeit
import imageio
import argparse
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

# parse command line arguments for input and output
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i:",
    "--input",
    type=str,
    help="Location of the input data folder to read",
)
parser.add_argument(
    "-o:",
    "--output",
    type=str,
    help="Location where to output the report html file",
)
args = parser.parse_args()
# set default parameter for plotly plots
default_plot_layout = {
    "height": 500,
    "width": 500,
    "autosize": False,
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
}


# FUNCTIONS


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
                    f"[report] Can not find quantification data for core #{i}. Please check if {str(quantification_path.resolve())} contains a valid .csv file for each core."
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
                f"Can not find contour for core {i}. Please check if /dearray/masks in {data_path_abs} contains a valid mask for each core."
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
    min = int(mask[np.nonzero(mask)].min())
    max = int(mask[np.nonzero(mask)].max())

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
        f"[report] Can not find contours for {error_counter} cells. Please check if /segmentation contains a valid cell segmentation mask."  # TODO read path variable in error message
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


# READING/SETTING PARAMETERS AND PATHS
if not args.input == None:
    data_path = args.input
    data_path_abs = str(Path(data_path).resolve())
else:
    data_path = "./data/"
    data_path_abs = str(Path(data_path).resolve())
    print(
        f"No path to data source was provided. Assuming {data_path_abs} as data source."
    )

print(f"[report] Reading MCMICRO output from {data_path_abs}")
if not Path(data_path).is_dir():
    print(
        f"[report] ERROR: Could not find input folder. Please check if {data_path_abs} exists and contains the full valid output of MCMICRO"
    )
    quit()
# set parameter for max image size, used for dynamic downscaling later on
max_image_size = (
    104857600  # 100MB # the actual html ends up being much smaller, how? Plotly magic??
)
# set inital scale factor for dynamic downscaling
scale_factor = 1

# y/x of the high res sample
side_length = 500

# find MCMICRO parameter yaml file


params_paths = [
    path for path in Path(data_path).rglob("*.yml")
]  # globbing just in case there are mulltiple parameter files

# open the parameter yaml

if params_paths == []:  # TODO refactor exeption handling, too much redundancy
    print(
        f"[report] ERROR: MCMICRO parameter file not found or could not be read. Please check the /qc folder in {data_path_abs} for a valid params.yml and rerun the script."
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

# read the MCMICRO modules that were run from the parameter yaml - currently unused, as parameters are not yet displayed
regex = re.compile(r"^.*?:")
modules = {key: value for key, value in mcmicro_params.items() if "module" in key}
segmentation_module_names = [
    re.sub(regex, "", module[0]) for module in modules["modulesPM"]
]

# assemble path to segmentation files
segmentation_path = data_path + "/segmentation/"

# assemble path to registration files
registration_path = data_path + "/registration/"

# assemble path to quantification files
quantification_path = data_path + "/quantification/"

# assemble path to dearrayed cores + corresponding files
cores_path = data_path + "/dearray/"
core_masks_path = data_path + "/dearray/masks/"
core_centroid_path = data_path + "/qc/coreo/centroidsY-X.txt"


# Check for dearray folder to determine if TMA or not
if Path(cores_path).is_dir() is True:
    print("[report] dearray folder detected. Assuming TMA data.")
    tma_mode = True
else:
    print("[report] No dearray folder detected. Assuming whole slide data.")
    tma_mode = False


# READING FILES


# load quantification csv(s) # TODO account for multiple files
quantification_dict = {}
try:
    for module in segmentation_module_names:
        # find all image paths
        quantification_dict[module] = [
            Path(path) for path in glob(f"{quantification_path}{module}*")
        ]
        # read the images
        quantification_dict[module] = [
            pd.read_csv(path) for path in quantification_dict[module]
        ]
except (OSError, FileNotFoundError):
    print(
        f"[report] ERROR: Registration image not found or could not be read. Please check {str(Path(quantification_path))} for valid csv files and rerun the script."
    )
    quit()


# TEMPORARY SOLUTION UNTIL MULTIPLE IMAGE SELECTION/SEGMENTATION ALGORITHM COMPARISION IS IMPLEMENTED IN THE FRONTEND
# TODO
quantification = list(quantification_dict.values())[0]


# loading images, assuming only one registration image can exist
try:
    whole_image = [Path(path) for path in glob(f"{registration_path}*")]
    image_file_size = os.path.getsize(
        whole_image[0]
    )  # get file size, used for downsampling later
    whole_image = imageio.imread(whole_image[0])
except IndexError:  # TODO refactor exeption handling, too much redundancy
    print(
        f"[report] ERROR: Registration image not found or could not be read. Please check {str(Path(registration_path))} for a valid image and rerun the script."
    )
    quit()

# load cell segementation masks
# use a dictionary to associate the images with different segmentation algorithms
segmentation_dict_cells = {}
try:
    for module in segmentation_module_names:
        # find all image paths
        segmentation_dict_cells[module] = [
            Path(path) for path in glob(f"{segmentation_path}{module}*/cell*")
        ]
        # read the images
        segmentation_dict_cells[module] = [
            imageio.imread(imagepath) for imagepath in segmentation_dict_cells[module]
        ]
except (OSError, FileNotFoundError):
    print(
        f"[report] ERROR: Segmentation images not found or could not be read. Please check {str(Path(segmentation_path))} for valid csv files and rerun the script."
    )
    quit()

# load nuclear segementation masks
# use a dictionary to associate the images with different segmentation algorithms
segmentation_dict_nuclei = {}
try:
    for module in segmentation_module_names:
        # find all image paths
        segmentation_dict_nuclei[module] = [
            Path(path) for path in glob(f"{segmentation_path}{module}*/cell*")
        ]
        # read the images
        segmentation_dict_nuclei[module] = [
            imageio.imread(imagepath) for imagepath in segmentation_dict_nuclei[module]
        ]

except (OSError, FileNotFoundError):
    print(
        f"[report] ERROR: Registration image not found or could not be read. Please check {str(Path(segmentation_path))} for valid csv files and rerun the script."
    )
    quit()

# read coordinates of the TMA core centroids and format them into an array
if tma_mode:
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
if tma_mode:
    num_cores = len(core_centroids)
else:
    num_cores = "-"


# ASSMEBLING FULL IMAGE CORE MASK IF TMA

# read the single core files
if tma_mode:
    # find the filepaths
    cores = [
        Path(path) for path in glob(f"{cores_path}[!_mask]*")
    ]  # TODO Can i use the exclude pattern in the path variable
    # read the images
    cores = [imageio.imread(imagepath) for imagepath in cores]
    # read the single core masks
    single_core_masks = [Path(path) for path in glob(f"{core_masks_path}*")]
    single_core_masks = [imageio.imread(imagepath) for imagepath in single_core_masks]
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
else:
    whole_core_mask = None


# ASSEMBLING FULL IMAGE SEGMENTATION MASK IF TMA


num_cells = 0
num_nuclei = 0

if tma_mode:

    # prepare an empty image to fill with cell/nuclei core in the next steps
    whole_core_mask_cells = np.zeros(whole_image.shape)
    whole_core_mask_nuclei = np.zeros(whole_image.shape)

    # prepare empty dicts to associate the whole image masks with different segmentation algorithms
    segmentation_dict_cells_whole = {}
    segmentation_dict_nuclei_whole = {}

    # loop through the modules
    for module in segmentation_dict_cells.keys():
        if not segmentation_dict_cells[module] == []:
            # loop through the core centroids and stich the mulitple single core masks together to one large core mask for the cells
            for index, core_centroid in enumerate(core_centroids):

                segmentation_dict_cells_whole[module] = overlay_images_at_centroid(
                    whole_core_mask_cells,
                    segmentation_dict_cells[module][index],
                    core_centroids[index][0],
                    core_centroids[index][1],
                )
                # determine number of cells from segmentation masks
                num_cells = num_cells + np.amax(segmentation_dict_cells[module][index])

            for index, core_centroid in enumerate(core_centroids):
                segmentation_dict_nuclei_whole[module] = overlay_images_at_centroid(
                    whole_core_mask_nuclei,
                    segmentation_dict_nuclei[module][index],
                    core_centroids[index][0],
                    core_centroids[index][1],
                )
                # determine number of cores from segmentation masks
                num_nuclei = num_nuclei + np.amax(
                    segmentation_dict_nuclei[module][index]
                )
    # TEMPORARY SOLUTION UNTIL MULTIPLE IMAGE SELECTION/SEGMENTATION ALGORITHM COMPARISION IS IMPLEMENTED IN THE FRONTEND
    # TODO
    segmentation_cells = list(segmentation_dict_cells_whole.values())[0]
    segmentation_nuclei = list(segmentation_dict_nuclei_whole.values())[0]
else:
    segmentation_cells = list(segmentation_dict_cells.values())[0]
    segmentation_nuclei = list(segmentation_dict_nuclei.values())[0]


# SCALING IMAGE


# getting image dimensions
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
if tma_mode:
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

# subset the images + masks with the corner points
high_res_crop = whole_image[p1[1] : p2[1], p1[0] : p2[0]]
high_res_segmentation_mask_cells = segmentation_cells[
    p1[1] : p2[1],
    p1[0] : p2[0],
]
high_res_segmentation_mask_nuclei = segmentation_nuclei[p1[1] : p2[1], p1[0] : p2[0]]


# PLOTTING


# prepare overview figure
Overview_plot = go.Figure()
Overview_plot.add_trace(go.Heatmap(z=whole_image_resized, colorscale="Viridis"))

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
    TMA_text = "TMA mode off or no TMA data could be found. Run MCMICRO--tma to process TMA data."

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
output_path = Path(f"{args.output}/report.html")
with open(output_path, "w") as fh:
    fh.write(report_html_code)

# open it in the default browser
browser.open(output_path)

# stop the timer
stop = timeit.default_timer()
elapsed = "{:.2f}".format(stop - start)
print(f"[report] Report generated in {elapsed} seconds")
