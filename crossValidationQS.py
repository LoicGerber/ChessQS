import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from scipy.ndimage import uniform_filter, label
from createKernel import createKernel
from crossValFunctionsQS import plot_binned_variogram, group_by_category, plot_histograms, compute_metrics
from g2s import g2s

start_time = time.time()

tile_size = 1001
threshold = 0.65
num_realizations = 10
ring_width = 10
gap = [350, 650, 350, 650] # [y_min, y_max, x_min, x_max]
# Variogram parameters
sample_percentage = 0.5
num_bins = 40
max_lag = 100

# Directory for saving results
dirPath = '/path/to/outputs'
os.makedirs(dirPath, exist_ok=True)

print("Opening .h5 file...")
input = '/path/to/input/file.h5'
with h5py.File(input, 'r') as file:
    img_3d = file['image_data'][:]
    img = img_3d[:, :, 0]

height, width = img.shape

print("Creating mask...")
mask_nan = np.isnan(img)
mask_neg2 = img == -2

combined_mask = mask_nan | mask_neg2
valid_mask = ~combined_mask  # 1 for valid pixels

print("Applying uniform filter to count valid pixels in each tile...")
valid_pixel_count = uniform_filter(valid_mask.astype(float), size=tile_size, mode='constant', cval=0)
print(f"Maximum informed percentage: {np.max(valid_pixel_count) * 100:.2f}%")

print(f"Identifying valid positions where valid pixel count > {threshold}...")
valid_positions = valid_pixel_count > threshold
valid_pixels    = valid_pixel_count[valid_positions]

print(f"Found {len(valid_pixels)} valid positions...")

print("Labeling connected structures...")
structure = np.ones((3, 3), dtype=int)  # 8-connectivity
labeled_array, num_features = label(valid_positions, structure=structure)
print(f"Found {num_features} connected structures.")

print("Plotting the percentage of informed pixels...")
percentage_informed = valid_pixel_count * 100
plt.figure(figsize=(10, 10))
plt.imshow(percentage_informed, cmap='hot', interpolation='none')
plt.colorbar(label='Percentage of Informed Pixels')
plt.title('Percentage of Informed Pixels per Tile')
plt.savefig(os.path.join(dirPath, 'informedPixPercentage.png'))
plt.show()

rgb_img = np.stack([img, img, img], axis=-1)
for label_idx in range(1, num_features + 1):
    structure_mask = labeled_array == label_idx
    rgb_img[structure_mask] = [1, 0, 0]  # Color connected structure in red
# Plot the result
plt.figure(figsize=(10, 10))
plt.imshow(rgb_img, interpolation='none')
plt.title(f"{num_features} connected structures with at least {int(threshold*100)}% of informed pixels")
plt.savefig(os.path.join(dirPath, f'identifiedStructures_t{int(threshold*100)}.png'))
plt.show()

best_informed_positions = []

for label_idx in range(1, num_features + 1):
    structure_mask = labeled_array == label_idx
    structure_pixels = valid_pixel_count[structure_mask]
    # Find the best-informed pixel in the structure (max percentage)
    max_informed_pixel = np.max(structure_pixels)
    best_pixel_index = np.argmax(structure_pixels)
    # Get the coordinates of the best pixel
    structure_coords = np.argwhere(structure_mask)
    best_pixel_coords = structure_coords[best_pixel_index]
    # Reconstruct the tile centered around the best pixel
    y, x = best_pixel_coords
    y_min, y_max = max(0, y - tile_size // 2), min(height, y + tile_size // 2)
    x_min, x_max = max(0, x - tile_size // 2), min(width, x + tile_size // 2)
    reconstructed_tile = img[y_min:y_max, x_min:x_max]
    # Store result
    best_informed_positions.append((y, x, max_informed_pixel, reconstructed_tile))

print("Visualizing the best-informed pixel per structure...")
for idx, (y, x, max_informed_pixel, reconstructed_tile) in enumerate(best_informed_positions):
    print(f"Best pixel in structure {idx + 1} at ({y}, {x}) with {max_informed_pixel * 100:.2f}% informed pixels")
    plt.figure(figsize=(6, 6))
    plt.imshow(reconstructed_tile, cmap='turbo', interpolation='none')
    plt.title(f"Reconstructed Tile for Structure {idx + 1}")
    plt.colorbar()
    plt.show()

ki = createKernel(path         = '',
                  layer_sizes  = [101, 101],
                  layer_values = [1],
                  map_type     = [2], # 0 = Uniform, 1 = Gaussian, 2 = Exponential, 3 = NaN
                  sigma_gaus   = 50,
                  expo_scale   = 0.02,
                  name         = 'ki',
                  file_format  = ''
)

results = []
rmse_variogram_list = []
integral_difference_list = []
all_real_valid = np.empty((gap[1]-gap[0], gap[3]-gap[2], len(best_informed_positions)*num_realizations))
all_sim_valid  = np.empty((gap[1]-gap[0], gap[3]-gap[2], len(best_informed_positions)*num_realizations))

it = 0

for idx, (y, x, max_informed_pixel, reconstructed_tile) in enumerate(best_informed_positions):
    print(f"---\nProcessing tile {idx + 1} at ({y}, {x}) with {max_informed_pixel * 100:.2f}% informed pixels\n---")

    outDir = os.path.join(dirPath,f'tile_{idx+1}')
    #os.makedirs(outDir, exist_ok=True)
    
    tiFull = np.copy(reconstructed_tile)
    tiFull = np.where(tiFull == -2, np.nan, tiFull)
    ti     = np.copy(tiFull)
    ti[gap[0]:gap[1], gap[2]:gap[3]] = np.nan

    diFull = np.copy(reconstructed_tile)
    diFull = np.nan_to_num(diFull, nan=-2)
    di     = np.copy(diFull)
    di[gap[0]:gap[1], gap[2]:gap[3]] = np.where(di[gap[0]:gap[1], gap[2]:gap[3]] != -2, np.nan, di[gap[0]:gap[1], gap[2]:gap[3]])
    
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(7,4))
    ax1.imshow(ti,cmap='turbo')
    ax1.set_title('Training image');
    ax1.axis('off');
    ax2.imshow(di,cmap='turbo')
    ax2.set_title('Simulation setup');
    ax2.axis('off');
    plt.show()
    
    for realization_idx in range(1, num_realizations + 1):
        print(f"Running realization {realization_idx}/{num_realizations}")

        simulation, index, *_ = g2s(
            '-sa', 'mercury.gaia.unil.ch',
            '-a', 'qs',
            '-ti', ti,
            '-di', di,
            '-dt', 1,
            '-ki', ki,
            '-k', 1,
            '-n', 10,
            '-j', 0.5
        )

        outName = f'tile_{idx+1}_realization_{realization_idx}_simulation'
        # tf.imwrite(os.path.join(outDir, f'{outName}.tif'), simulation)

        outputPath = os.path.join(outDir, f'{outName}.h5')
        # with h5py.File(outputPath, 'w') as f:
            # f.create_dataset('image_data', data=simulation)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4))
        ax1.imshow(diFull, cmap='turbo')
        ax1.set_title('Real image')
        ax1.axis('off')
        ax2.imshow(simulation, cmap='turbo')
        ax2.set_title(f'Simulation result - Realization {realization_idx}')
        ax2.axis('off')
        plt.show()
        
        # Define inner square
        inner_square_real = np.where(diFull[gap[0]+ring_width:gap[1]-ring_width, gap[2]+ring_width:gap[3]-ring_width] == -2, np.nan, diFull[gap[0]+ring_width:gap[1]-ring_width, gap[2]+ring_width:gap[3]-ring_width])
        inner_square_sim  = np.where(simulation[gap[0]+ring_width:gap[1]-ring_width, gap[2]+ring_width:gap[3]-ring_width] == -2, np.nan, simulation[gap[0]+ring_width:gap[1]-ring_width, gap[2]+ring_width:gap[3]-ring_width])

        # Whole gap
        real = np.where(diFull[gap[0]:gap[1], gap[2]:gap[3]] == -2, np.nan, diFull[gap[0]:gap[1], gap[2]:gap[3]])
        sim  = np.where(simulation[gap[0]:gap[1], gap[2]:gap[3]] == -2, np.nan, simulation[gap[0]:gap[1], gap[2]:gap[3]])

        # Define the outer ring
        outer_ring_real = np.copy(real)
        outer_ring_real[gap[0]+ring_width:gap[1]-ring_width, gap[2]+ring_width:gap[3]-ring_width] = np.nan
        outer_ring_sim = np.copy(sim)
        outer_ring_sim[gap[0]+ring_width:gap[1]-ring_width, gap[2]+ring_width:gap[3]-ring_width] = np.nan
        
        # Compute for whole gap
        accuracy_whole, kappa_whole, rmse_whole, integral_diff_whole, real_whole, sim_whole = compute_metrics(real, sim, "Whole Gap", sample_percentage, num_bins, max_lag)
        # Compute for outer ring
        accuracy_outer, kappa_outer, rmse_outer, integral_diff_outer, real_outer, sim_outer = compute_metrics(outer_ring_real, outer_ring_sim, "Outer Ring", sample_percentage, num_bins, max_lag)
        # Compute for inner square
        accuracy_inner, kappa_inner, rmse_inner, integral_diff_inner, real_inner, sim_inner = compute_metrics(inner_square_real, inner_square_sim, "Inner Square", sample_percentage, num_bins, max_lag)

        # Store results for this realization
        results.append({
            'tile_index': idx + 1,
            'realization_index': realization_idx,
            'best_pixel_coords': (y, x),
            'max_informed_pixel': max_informed_pixel,
            'simulation_result': simulation,
            'accuracy_whole': accuracy_whole,
            'kappa_whole': kappa_whole,
            'rmse_whole': rmse_whole,
            'integral_diff_whole': integral_diff_whole,
            'accuracy_outer': accuracy_outer,
            'kappa_outer': kappa_outer,
            'rmse_outer': rmse_outer,
            'integral_diff_outer': integral_diff_outer,
            'accuracy_inner': accuracy_inner,
            'kappa_inner': kappa_inner,
            'rmse_inner': rmse_inner,
            'integral_diff_inner': integral_diff_inner
        })
        # Append to global lists
        all_real_valid[:, :, it] = real
        all_sim_valid[:, :, it]  = sim

        print(f"Tile {idx + 1}, realisation {realization_idx}/{num_realizations}, processed. Accuracy: {accuracy_whole:.4f}, Kappa: {kappa_whole:.4f}")
        
        it += 1
        
np.save(os.path.join(dirPath, 'simulation_results.npy'), results)

# Compute mean accuracy and kappa for outer ring and inner square
mean_accuracy_whole = np.mean([result['accuracy_whole'] for result in results])
mean_kappa_whole = np.mean([result['kappa_whole'] for result in results])
print(f"Overall Accuracy: {mean_accuracy_whole:.4f}, Kappa: {mean_kappa_whole:.4f}")
mean_accuracy_outer = np.mean([result['accuracy_outer'] for result in results])
mean_kappa_outer = np.mean([result['kappa_outer'] for result in results])
print(f"Overall Outer Ring Accuracy: {mean_accuracy_outer:.4f}, Kappa: {mean_kappa_outer:.4f}")
mean_accuracy_inner = np.mean([result['accuracy_inner'] for result in results])
mean_kappa_inner = np.mean([result['kappa_inner'] for result in results])
print(f"Overall Inner Square Accuracy: {mean_accuracy_inner:.4f}, Kappa: {mean_kappa_inner:.4f}")

# Group RMSE values by category across all realizations for each region
rmse_whole_by_category = group_by_category(results, 'rmse_whole')
rmse_outer_by_category = group_by_category(results, 'rmse_outer')
rmse_inner_by_category = group_by_category(results, 'rmse_inner')

# Group integral of the difference values by category for each region
integral_whole_by_category = group_by_category(results, 'integral_diff_whole')
integral_outer_by_category = group_by_category(results, 'integral_diff_outer')
integral_inner_by_category = group_by_category(results, 'integral_diff_inner')

# Plot histograms for RMSE
plot_histograms(rmse_whole_by_category, 'RMSE (Whole Gap)', 'RMSE of difference', 'histogram_rmse_whole', dirPath)
plot_histograms(rmse_outer_by_category, 'RMSE (Outer Ring)', 'RMSE of difference', 'histogram_rmse_outer', dirPath)
plot_histograms(rmse_inner_by_category, 'RMSE (Inner Square)', 'RMSE of difference', 'histogram_rmse_inner', dirPath)

# Plot histograms for integral of the difference
plot_histograms(integral_whole_by_category, 'Integral of Difference (Whole Gap)', 'Integral of difference', 'histogram_integral_whole', dirPath)
plot_histograms(integral_outer_by_category, 'Integral of Difference (Outer Ring)', 'Integral of difference', 'histogram_integral_outer', dirPath)
plot_histograms(integral_inner_by_category, 'Integral of Difference (Inner Square)', 'Integral of difference', 'histogram_integral_inner', dirPath)

# Categories proportion
categoriesTiles, countsTiles = np.unique(all_real_valid[~np.isnan(all_real_valid)], return_counts=True)
proportionsTiles = countsTiles / np.sum(countsTiles)
categoriesAll, countsAll = np.unique(img[~np.isnan(img) & (img != -2)], return_counts=True)
proportionsAll = countsAll / np.sum(countsAll)

# Create a new figure
bar_width = 0.4
plt.figure(figsize=(10, 6))
# Adjust the x positions to separate the bars
categories = np.union1d(categoriesTiles, categoriesAll)
x_indexes = np.arange(len(categories))
# Align the bars side-by-side
plt.bar(x_indexes - bar_width/2, proportionsTiles, bar_width, color='blue', edgecolor='black', alpha=0.7, label='Cross-validation tiles')
plt.bar(x_indexes + bar_width/2, proportionsAll, bar_width, color='red', edgecolor='black', alpha=0.7, label='Full Image')
# Add labels and title
plt.xlabel('Categories')
plt.ylabel('Proportion')
plt.title('Proportion of Each Category (Tiles vs Full Image)')
# Add category ticks
plt.xticks(x_indexes, categories)
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(dirPath, f'histogram_proportion_crossValTiles_categories.png'))
plt.show()

print("All results saved!")

end_time = time.time()
total_time = end_time - start_time

# Print the total time taken
print(f"Total time: {total_time:.2f} seconds.")
