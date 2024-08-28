import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from rasterio.transform import from_origin
import rasterio

from chessboardFunctionsQS import (
    load_image, prepare_tiles, visualize_tiles, 
    visualize_filtered_chessboard, identify_poorly_informed_tiles,  
    iterative_merge_poorly_informed_tiles,
    visualize_modified_tiles, run_simulations
)
from createKernel import createKernel

# Directory parameters
dirPath   = r'path\to\directory'
tiName    = 'ti.h5'     # either tiff or h5
diName    = 'di.h5'     # either tiff or h5
outName   = 'outName'   # without extension

# Chessboard parameters
tile_size   = 1000          # size of chessboard tiles
overlap     = 100           # overlap between tiles
threshold   = 25            # threshold to define poorly informed tiles
maxTileSize = [2000, 2000]  # maximum length of merged tiles, in pixels 

# Variable weights
varWeights = [1, 0.1]

# Kernel parameters
ki_params = {
    'path':          dirPath,   # path to saving directory
    'layer_sizes':  [101, 101], # size of kernel
    'layer_values': [1, 1],        # value for each layer
    'map_type':     [2, 3],        # 0: uniform, 1: gaussian, 2: exponential, 3: NaN except central pixel
    'sigma_gaus':    50,        # sigma for gaussian maps
    'expo_scale':    0.02,      # scaling factor for exponential maps
    'name':         'ki',       # name of file for saving
    'file_format':  ''          # saving format: 'tiff', 'hdf5', or '' not to save
}

# QS parameters
g2s_params = (
    '-a',  'qs',    # algorithm
    '-dt', [1, 0],     # data type, 0: continuous, 1: categorical
    '-k',   1,      # 
    '-n',  [10, 1],    # number of neighbours
    '-j',   0.5     # computing power
)

# GeoTIFF parameters
pixel_size_x = 2                    # x-axis pixel size, positive
pixel_size_y = 2                    # y-axis pixel size, positive
xmin, xmax   = 2565000, 2590000     # x-axis min max coordinates
ymin, ymax   = 1133000, 1175000     # y-axis min max coordinates
crs_str      = 'EPSG:2056'          # EPSG code

# Show plots
plot = False

# Load images
ti_file_path = os.path.join(dirPath, tiName)
ti           = load_image(ti_file_path)
ti           = ti * np.sqrt(varWeights)
di_file_path = os.path.join(dirPath, diName)
di           = load_image(di_file_path)
di           = di * np.sqrt(varWeights)

# Create the kernel
ki = createKernel(**ki_params)

# Prepare tiles, analyze, and identify ignored tiles
tiles, tile_analysis, ignored_tiles, empty_tiles, tiles_no_nans_in_di = prepare_tiles(ti, di, tile_size, overlap)

# Print the number of tiles and ignored tiles
print(f'Total number of tiles: {len(tiles)}')
print(f'Ignored tiles: {len(ignored_tiles)}')
print(f'Tiles to be simulated: {len(tiles) - len(ignored_tiles)}')

if plot == True:
    # Visualize the tiles on the image
    visualize_tiles(di, tiles, ignored_tiles, empty_tiles, tiles_no_nans_in_di)
    # Visualize the chessboard pattern
    visualize_filtered_chessboard(di, tiles, ignored_tiles, tile_size, overlap)

# Identify poorly informed tiles
poorly_informed_tiles, nan_gt_informed_tiles = identify_poorly_informed_tiles(ti, tiles, tile_analysis, empty_tiles, ignored_tiles, threshold, plot)
print(f"{len(poorly_informed_tiles)} tiles without at least {threshold}% informed pixels:", poorly_informed_tiles)
print(f"{len(nan_gt_informed_tiles)} tiles with more NaNs in Di than informed pixels in Ti:", nan_gt_informed_tiles)

# Merge poorly informed tiles
mod_ti_tiles, mod_di_tiles = iterative_merge_poorly_informed_tiles(ti, di, tiles, tile_analysis, poorly_informed_tiles, nan_gt_informed_tiles, empty_tiles, ignored_tiles, tile_size, overlap, threshold, maxTileSize, max_iterations=10)

if plot == True:
    # Visualize the modified tiles
    visualize_modified_tiles(di, tiles, mod_ti_tiles, mod_di_tiles, ignored_tiles, None)  # None shows all modified tiles

# Run simulations based on the modified tiles
final_simulation_result = run_simulations(
    ti, di, mod_ti_tiles, mod_di_tiles, tiles, tile_analysis, ignored_tiles, ki, g2s_params, tile_size, overlap
)

# Save the final result as a GeoTIFF file
output_tiff   = os.path.join(dirPath, f'{outName}.tif')
new_transform = from_origin(xmin, ymax, pixel_size_x, pixel_size_y)
with rasterio.open(
    output_tiff,
    'w',
    driver='GTiff',
    height=final_simulation_result.shape[0],
    width=final_simulation_result.shape[1],
    count=1,  # Assuming the result is a single band
    dtype=final_simulation_result.dtype,
    crs=crs_str,
    transform=new_transform
) as dst:
    dst.write(final_simulation_result, 1)

# Save the final result as an HDF5 file
outputPath = os.path.join(dirPath, f'{outName}.h5')
with h5py.File(outputPath, 'w') as f:
    f.create_dataset('image_data', data=final_simulation_result)

if plot == True:
    # Visualize the result
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
    ax1.imshow(di, cmap='turbo')
    ax1.set_title('Initial state')
    ax1.axis('off')
    ax2.imshow(final_simulation_result, cmap='turbo')
    ax2.set_title('Simulation result')
    ax2.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(dirPath, 'initResultComp.png'), dpi=300, bbox_inches='tight')
