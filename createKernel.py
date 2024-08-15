import numpy as np
import tifffile as tf
import h5py
import os
from scipy.stats import multivariate_normal

def create_3d_matrix(layer_sizes, layer_values):
    height, width = layer_sizes[0], layer_sizes[1]
    num_layers = len(layer_values)
    matrix_3d = np.zeros((height, width, num_layers))
    for i, value in enumerate(layer_values):
        matrix_3d[:, :, i] = value
    return matrix_3d

def create_uniform_map(layer_sizes):
    height, width = layer_sizes[0], layer_sizes[1]
    uniform_map = np.ones((height, width))
    return uniform_map

def create_gaussian_map(layer_sizes, sigma):
    height, width = layer_sizes[0], layer_sizes[1]
    center = (height // 2, width // 2)
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    pos = np.dstack((x, y))
    rv = multivariate_normal(center, sigma)
    return rv.pdf(pos)

def create_exponential_map(layer_sizes, scale):
    height, width = layer_sizes
    center = (height // 2, width // 2)
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    distance_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    exponential_map = np.exp(-distance_from_center * scale)
    return exponential_map

def create_nan_map(layer_sizes):
    height, width = layer_sizes[0], layer_sizes[1]
    nan_map = np.full((height, width), np.nan)
    center = (height // 2, width // 2)
    nan_map[center] = 1
    return nan_map

def create_distribution_map(layer_sizes, sigma_or_scale, distribution_type='gaussian'):
    if distribution_type == 'gaussian':
        return create_gaussian_map(layer_sizes, sigma_or_scale)
    elif distribution_type == 'exponential':
        return create_exponential_map(layer_sizes, sigma_or_scale)
    elif distribution_type == 'uniform':
        return create_uniform_map(layer_sizes)
    else:
        raise ValueError("Unsupported distribution type. Choose either 'gaussian', 'exponential', or 'uniform'.")

def save_as_hdf5(path, name, data):
    outputPath = os.path.join(path, f'{name}.h5')
    with h5py.File(outputPath, 'w') as f:
        f.create_dataset('image_data', data=data)
    print("File saved as:", outputPath)

def save_as_tiff(path, name, data):
    outputPath = os.path.join(path, f'{name}.tif')
    tf.imwrite(outputPath, data)
    print("File saved as:", outputPath)

def createKernel(path, layer_sizes, layer_values, map_type, sigma_gaus, expo_scale, name, file_format):
    ki = create_3d_matrix(layer_sizes, layer_values)

    gaussian_map = create_distribution_map(layer_sizes, sigma_gaus, 'gaussian')
    gaussian_map = (gaussian_map - np.min(gaussian_map)) / (np.max(gaussian_map) - np.min(gaussian_map))

    exponential_map = create_distribution_map(layer_sizes, expo_scale, 'exponential')
    if expo_scale != 0:
        exponential_map = (exponential_map - np.min(exponential_map)) / (np.max(exponential_map) - np.min(exponential_map))

    uniform_map = create_distribution_map(layer_sizes, None, 'uniform')

    nan_map = create_nan_map(layer_sizes)

    for i in range(len(map_type)):
        if map_type[i] == 0:
            ki[:, :, i] = uniform_map * layer_values[i]
        elif map_type[i] == 1:
            ki[:, :, i] = gaussian_map * layer_values[i]
        elif map_type[i] == 2:
            ki[:, :, i] = exponential_map * layer_values[i]
        elif map_type[i] == 3:
            ki[:, :, i] = nan_map * layer_values[i]
    
    if file_format == 'tiff':
        save_as_tiff(path, name, ki)
    elif file_format == 'hdf5':
        save_as_hdf5(path, name, ki)
    else:
        print('ki not saved...')

    return ki # 0 = Uniform, 1 = Gaussian, 2 = Exponential, 3 = NaN
