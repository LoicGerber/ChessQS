import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import os
import time
from scipy.ndimage import uniform_filter, label
from createKernel import createKernel
from chessFunctionsQS import generate_simulation_path
from crossValFunctionsQS import plot_sub_histograms, group_by_category, plot_histograms, compute_metrics, compute_binned_variogram, apply_nugget_effect, compute_nugget_metrics, compute_rmse_all_tiles
from g2s import g2s

runSim         = False
compute_varios = True
histo          = True
plot_varios    = True

server = 'localhost'  # Server address for g2s

start_time = time.time()

tile_size        = 1001
threshold        = 0.65
num_realizations = 10
ring_width       = 25
gap              = [350, 650, 350, 650] # [y_min, y_max, x_min, x_max]
# Variogram parameters
sample_percentage = 0.5
num_bins          = 40
vario_max_lag     = 100
seed              = 42

metrics_max_lag   = 20  # Maximum lag for metrics computation

numLayers = 5
dt        = [1, 0, 0, 0, 0]
n         = [10, 10, 2, 2, 2]
k         = 1
j         = 0.5
inwardSim  = False
sp_exclude = [-2]

layer_sizes  = [101, 101]
layer_values = [5,1,1,1,1]
map_type     = [2,2,2,2,2] # 0 = Uniform, 1 = Gaussian, 2 = Exponential, 3 = NaN
sigma_gaus   = 50
expo_scale   = 0.02
name         = 'ki'
file_format  = '' # 'tiff', 'hdf5', or ''

# Directory for saving results
dirPath = '/path/to/results/folder'
os.makedirs(dirPath, exist_ok=True)

print("Opening .h5 file...")
input = '/path/to/DI.h5'
with h5py.File(input, 'r') as file:
    img_3d = file['image_data'][:]
    img = img_3d[:, :, :numLayers]

height, width, depth = img.shape

print("Creating mask...")
mask_nan = np.isnan(img[:,:,0])
mask_neg2 = img[:,:,0] == -2

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
plt.figure(figsize=(5, 5))
plt.imshow(percentage_informed, cmap='hot', interpolation='none')
plt.colorbar(label='Percentage of informed pixels')
# plt.title('Percentage of Informed Pixels per Tile')
plt.axis('off')
plt.savefig(os.path.join(dirPath, 'informedPixPercentage.png'))
plt.show()

rgb_img = np.stack([img[:,:,0], img[:,:,0], img[:,:,0]], axis=-1)
for label_idx in range(1, num_features + 1):
    structure_mask = labeled_array == label_idx
    rgb_img[structure_mask] = [1, 0, 0]  # Color connected structure in red
# Plot the result
plt.figure(figsize=(5, 5))
plt.imshow(rgb_img, interpolation='none')
# plt.title(f"{num_features} connected structures with at least {int(threshold*100)}% of informed pixels")
plt.axis('off')
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
    reconstructed_tile = img[y_min:y_max, x_min:x_max, :]
    # Store result
    best_informed_positions.append((y, x, max_informed_pixel, reconstructed_tile))

print("Visualizing the best-informed pixel per structure...")
classes = np.arange(-2, 7)
turbo = plt.cm.turbo
colors = turbo(np.linspace(0, 1, len(classes)))
colors[0] = [0.85, 0.85, 0.85, 1.0]
cmap = ListedColormap(colors)
norm = BoundaryNorm(classes - 0.5, len(classes))
for idx, (y, x, max_informed_pixel, reconstructed_tile) in enumerate(best_informed_positions):
    print(f"Best pixel in structure {idx + 1} at ({y}, {x}) with {max_informed_pixel * 100:.2f}% informed pixels")
    plt.figure(figsize=(5, 5))
    plt.imshow(reconstructed_tile[:,:,0], cmap=cmap, norm=norm, interpolation='none')
    # plt.title(f"Reconstructed Tile for Structure {idx + 1}")
    plt.axis('off')
    # plt.colorbar(ticks=classes, boundaries=classes)# - 0.5)
    plt.show()

ki = createKernel(path         = '',
                  layer_sizes  = layer_sizes,
                  layer_values = layer_values,
                  map_type     = map_type,
                  sigma_gaus   = sigma_gaus,
                  expo_scale   = expo_scale,
                  name         = name,
                  file_format  = file_format # 'tiff' or 'hdf5'
                )

results                  = []
rmse_variogram_list      = []
integral_difference_list = []
all_real_valid           = np.zeros((gap[1]-gap[0], gap[3]-gap[2], len(best_informed_positions)*num_realizations))
all_sim_valid            = np.zeros((gap[1]-gap[0], gap[3]-gap[2], len(best_informed_positions)*num_realizations))
varios                   = []
all_vario_real_whole     = {}
all_vario_real_out       = {}
all_vario_real_in        = {}
all_vario_sim_whole      = {}
all_vario_sim_out        = {}
all_vario_sim_in         = {}
all_vario_nugget_whole   = {}
all_vario_nugget_out     = {}
all_vario_nugget_in      = {}
        
it = 0
if runSim == True:
    for idx, (y, x, max_informed_pixel, reconstructed_tile) in enumerate(best_informed_positions):
        print(f"---\nProcessing tile {idx + 1} at ({y}, {x}) with {max_informed_pixel * 100:.2f}% informed pixels\n---")

        outDir = os.path.join(dirPath,f'tile_{idx+1}')
        #os.makedirs(outDir, exist_ok=True)
        
        tiFull = np.copy(reconstructed_tile)
        tiFull[:,:,0] = np.where(tiFull[:,:,0] == -2, np.nan, tiFull[:,:,0])
        ti     = np.copy(tiFull)
        ti[gap[0]:gap[1], gap[2]:gap[3], 0] = np.nan

        diFull = np.copy(reconstructed_tile)
        diFull[:,:,0] = np.nan_to_num(diFull[:,:,0], nan=-2)
        di     = np.copy(diFull)
        di[gap[0]:gap[1], gap[2]:gap[3], 0] = np.where(di[gap[0]:gap[1], gap[2]:gap[3], 0] != -2, np.nan, di[gap[0]:gap[1], gap[2]:gap[3], 0])
        
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(7,4))
        ax1.imshow(ti[:,:,0], cmap=cmap, norm=norm, interpolation='none')
        ax1.set_title('Training image');
        ax1.axis('off');
        ax2.imshow(di[:,:,0], cmap=cmap, norm=norm, interpolation='none')
        ax2.set_title('Simulation setup');
        ax2.axis('off');
        plt.show()
        
        plt.figure(figsize=(5, 5))
        plt.imshow(ti[:,:,0], cmap=cmap, norm=norm, interpolation='none')
        plt.axis('off')
        plt.show()
               
        for realization_idx in range(1, num_realizations + 1):
            print(f"Running realization {realization_idx}/{num_realizations}")

            sp = generate_simulation_path(di[:,:,0], inwardSim, sp_exclude)
            di_clean = di.copy()
            di_target_clean = di_clean[:,:,0]
            di_target_clean[di_target_clean == sp_exclude] = np.nan  # Replace excluded pixels with NaN
            di_clean[:,:,0] = di_target_clean
            
            simulation, index, *_ = g2s(
                '-sa', server,
                '-a', 'qs',
                '-ti', ti,
                '-di', di_clean,
                '-dt', dt,
                '-ki', ki,
                '-k',  k,
                '-n',  n,
                '-j',  j,
                '-sp', sp
            )
            simulation = simulation[:,:,0]
            simulation[di_target_clean == sp_exclude] = sp_exclude  # Restore excluded pixels to their original value
            
            outName = f'tile_{idx+1}_realization_{realization_idx}_simulation'
            # tf.imwrite(os.path.join(outDir, f'{outName}.tif'), simulation)

            outputPath = os.path.join(outDir, f'{outName}.h5')
            # with h5py.File(outputPath, 'w') as f:
                # f.create_dataset('image_data', data=simulation)
            diFig = np.copy(diFull[:,:,0])
            diFig[diFig == -2] = np.nan

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4))
            ax1.imshow(diFig, cmap='turbo')
            ax1.set_title('Real image')
            ax1.axis('off')
            ax2.imshow(simulation, cmap='turbo')
            ax2.set_title(f'Simulation result - Realization {realization_idx}')
            ax2.axis('off')
            plt.show()
            
            # Whole gap
            real = np.where(diFull[gap[0]:gap[1], gap[2]:gap[3], 0] == -2, np.nan, diFull[gap[0]:gap[1], gap[2]:gap[3], 0])
            sim  = np.where(simulation[gap[0]:gap[1], gap[2]:gap[3]] == -2, np.nan, simulation[gap[0]:gap[1], gap[2]:gap[3]])
            
            # Define inner square
            inner_square_real = np.where(diFull[gap[0]+ring_width:gap[1]-ring_width, gap[2]+ring_width:gap[3]-ring_width, 0] == -2, np.nan, diFull[gap[0]+ring_width:gap[1]-ring_width, gap[2]+ring_width:gap[3]-ring_width, 0])
            inner_square_sim  = np.where(simulation[gap[0]+ring_width:gap[1]-ring_width, gap[2]+ring_width:gap[3]-ring_width] == -2, np.nan, simulation[gap[0]+ring_width:gap[1]-ring_width, gap[2]+ring_width:gap[3]-ring_width])

            # Define the outer ring
            outer_ring_real = np.copy(real)
            outer_ring_real[ring_width:-ring_width, ring_width:-ring_width] = np.nan
            outer_ring_sim = np.copy(sim)
            outer_ring_sim[ring_width:-ring_width, ring_width:-ring_width] = np.nan
            
            # Compute for whole gap
            accuracy_whole, kappa_whole, rmse_whole, integral_diff_whole, real_whole, sim_whole, vario_real, vario_sim = compute_metrics(real, sim, "Whole Gap", sample_percentage, num_bins, vario_max_lag, seed, real)
            # Compute for outer ring
            accuracy_outer, kappa_outer, rmse_outer, integral_diff_outer, real_outer, sim_outer, vario_real, vario_sim = compute_metrics(outer_ring_real, outer_ring_sim, "Outer Ring", sample_percentage, num_bins, vario_max_lag, seed, real)
            # Compute for inner square
            accuracy_inner, kappa_inner, rmse_inner, integral_diff_inner, real_inner, sim_inner, vario_real, vario_sim = compute_metrics(inner_square_real, inner_square_sim, "Inner Square", sample_percentage, num_bins, vario_max_lag, seed, real)

            # Store results for this realization
            results.append({
                'tile_index': idx + 1,
                'realization_index': realization_idx,
                'best_pixel_coords': (y, x),
                'max_informed_pixel': max_informed_pixel,
                'simulation_result': simulation,
                'variogram_real': vario_real,
                'variogram_sim': vario_sim,
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
else:
    results = np.load(os.path.join(dirPath, 'simulation_results.npy'), allow_pickle=True)
    for idx, (y, x, max_informed_pixel, reconstructed_tile) in enumerate(best_informed_positions):
        print(f"---\nProcessing tile {idx + 1}\n---")
        
        diFull = np.copy(reconstructed_tile[:,:,0])
        diFull = np.nan_to_num(diFull, nan=-2)
        
        tile_idx = idx + 1
        
        all_vario_real_whole[tile_idx]   = {}
        all_vario_real_out[tile_idx]     = {}
        all_vario_real_in[tile_idx]      = {}
        all_vario_sim_whole[tile_idx]    = {}
        all_vario_sim_out[tile_idx]      = {}
        all_vario_sim_in[tile_idx]       = {}
        all_vario_nugget_whole[tile_idx] = {}
        all_vario_nugget_out[tile_idx]   = {}
        all_vario_nugget_in[tile_idx]    = {}
        
        if compute_varios == True:
            print('Processing real variogram...')
            # Whole gap
            real = np.where(diFull[gap[0]:gap[1], gap[2]:gap[3]] == -2, np.nan, diFull[gap[0]:gap[1], gap[2]:gap[3]])
            # Define inner square
            inner_square_real = np.where(diFull[gap[0]+ring_width:gap[1]-ring_width, gap[2]+ring_width:gap[3]-ring_width] == -2, np.nan, diFull[gap[0]+ring_width:gap[1]-ring_width, gap[2]+ring_width:gap[3]-ring_width])
            # Define the outer ring
            outer_ring_real = np.copy(real)
            outer_ring_real[ring_width:-ring_width, ring_width:-ring_width] = np.nan
            
            bench        = np.where(diFull[gap[0]:gap[1], gap[2]:gap[3]] != -2, np.nan, diFull[gap[0]:gap[1], gap[2]:gap[3]])
            outBench     = np.copy(diFull)  # copy full tile
            gap_mask     = diFull[gap[0]:gap[1], gap[2]:gap[3]] != -2
            outBench[gap[0]:gap[1], gap[2]:gap[3]][gap_mask] = np.nan
            valid_mask   = (outBench != -2) & (~np.isnan(outBench))
            cats, counts = np.unique(outBench[valid_mask], return_counts=True)
            proportions  = counts / np.sum(counts)
            nugget       = apply_nugget_effect(bench, cats, proportions, seed=42)
            nugget_in    = np.where(nugget[gap[0]+ring_width:gap[1]-ring_width, gap[2]+ring_width:gap[3]-ring_width] == -2, np.nan, nugget[gap[0]+ring_width:gap[1]-ring_width, gap[2]+ring_width:gap[3]-ring_width])
            nugget_out   = np.copy(nugget)
            nugget_out[ring_width:-ring_width, ring_width:-ring_width] = np.nan

            vario_real_whole   = compute_binned_variogram(real, sample_percentage, num_bins, vario_max_lag, seed, real)
            vario_real_out     = compute_binned_variogram(inner_square_real, sample_percentage, num_bins, vario_max_lag, seed, real)
            vario_real_in      = compute_binned_variogram(outer_ring_real, sample_percentage, num_bins, vario_max_lag, seed, real)
            vario_nugget_whole = compute_binned_variogram(nugget, sample_percentage, num_bins, vario_max_lag, seed, nugget)
            vario_nugget_out   = compute_binned_variogram(nugget, sample_percentage, num_bins, vario_max_lag, seed, real)
            vario_nugget_in    = compute_binned_variogram(nugget, sample_percentage, num_bins, vario_max_lag, seed, real)
            
            all_vario_real_whole[tile_idx]   = vario_real_whole
            all_vario_real_out[tile_idx]     = vario_real_out
            all_vario_real_in[tile_idx]      = vario_real_in
            all_vario_nugget_whole[tile_idx] = vario_nugget_whole
            all_vario_nugget_out[tile_idx]   = vario_nugget_out
            all_vario_nugget_in[tile_idx]    = vario_nugget_in
        
        simulations = [sim for sim in results if sim['tile_index'] == tile_idx]
        
        for realization_idx in range(1, num_realizations + 1):
            simulation = [sim for sim in simulations if sim['realization_index'] == realization_idx][0]
            
            real = np.where(diFull[gap[0]:gap[1], gap[2]:gap[3]] == -2, np.nan, diFull[gap[0]:gap[1], gap[2]:gap[3]])
            all_real_valid[:, :, it] = real
            
            sim  = np.where(simulation['simulation_result'][gap[0]:gap[1], gap[2]:gap[3]] == -2, np.nan, simulation['simulation_result'][gap[0]:gap[1], gap[2]:gap[3]])
            all_sim_valid[:, :, it]  = sim
            
            if compute_varios == True:
                print(f'Processing simulated variogram for tile {tile_idx}, realization ({realization_idx}/{num_realizations})...')
                # Whole gap
                sim  = np.where(simulation['simulation_result'][gap[0]:gap[1], gap[2]:gap[3]] == -2, np.nan, simulation['simulation_result'][gap[0]:gap[1], gap[2]:gap[3]])
                # Define inner square
                inner_square_sim  = np.where(simulation['simulation_result'][gap[0]+ring_width:gap[1]-ring_width, gap[2]+ring_width:gap[3]-ring_width] == -2, np.nan, simulation['simulation_result'][gap[0]+ring_width:gap[1]-ring_width, gap[2]+ring_width:gap[3]-ring_width])
                # Define the outer ring
                outer_ring_sim = np.copy(sim)
                outer_ring_sim[ring_width:-ring_width, ring_width:-ring_width] = np.nan
                
                vario_sim_whole = compute_binned_variogram(sim, sample_percentage, num_bins, vario_max_lag, seed, sim)
                vario_sim_out   = compute_binned_variogram(inner_square_sim, sample_percentage, num_bins, vario_max_lag, seed, real)
                vario_sim_in    = compute_binned_variogram(outer_ring_sim, sample_percentage, num_bins, vario_max_lag, seed, real)
                
                all_vario_sim_whole[tile_idx][it] = vario_sim_whole
                all_vario_sim_out[tile_idx][it]   = vario_sim_out
                all_vario_sim_in[tile_idx][it]    = vario_sim_in
            
            it += 1
if compute_varios == True:
    varios.append({
        'vario_real_whole': all_vario_real_whole,
        'vario_real_out':   all_vario_real_out,
        'vario_real_in':    all_vario_real_in,
        'vario_sim_whole':  all_vario_sim_whole,
        'vario_sim_out':    all_vario_sim_out,
        'vario_sim_in':     all_vario_sim_in,
        'vario_nugget_whole': all_vario_nugget_whole,
        'vario_nugget_out':   all_vario_nugget_out,
        'vario_nugget_in':    all_vario_nugget_in
    })
    np.save(os.path.join(dirPath, 'variograms.npy'), varios)

if histo == True:
    varios = np.load(os.path.join(dirPath, 'variograms.npy'), allow_pickle=True)
    
    variograms_real   = [variogram['vario_real_whole']   for variogram in varios]
    variograms_sim    = [variogram['vario_sim_whole']    for variogram in varios]
    variograms_nugget = [variogram['vario_nugget_whole'] for variogram in varios]
    variograms_real   = variograms_real[0]
    variograms_sim    = variograms_sim[0]
    variograms_nugget = variograms_nugget[0]
    
    all_rmses = compute_rmse_all_tiles(variograms_real, variograms_sim, metrics_max_lag)
    rmse_all  = group_by_category(all_rmses)
    # plot_histograms(rmse_all, 'RMSE', 'RMSE of difference', 'histogram_rmse_first_lags', dirPath)
    nugget_whole_rmse, _ = compute_nugget_metrics(variograms_real, variograms_nugget, metrics_max_lag)
    plot_sub_histograms(rmse_all, nugget_whole_rmse, 'RMSE of difference', 'histogram_rmse_first_lags', dirPath)
    mean_rmses = {}
    med_rmses = {}
    for category, values in rmse_all.items():
        mean_value = np.mean(values)
        med_value = np.median(values)
        mean_rmses[category] = mean_value
        med_rmses[category] = med_value
        print(f"Category {category}: mean = {mean_value:.3f}, median = {med_value:.3f}")
    mean_nugget = {}
    med_nugget = {}
    for category, values in nugget_whole_rmse.items():
        mean_value = np.mean(values)
        med_value = np.median(values)
        mean_nugget[category] = mean_value
        med_nugget[category] = med_value
        print(f"Nugget for category {category}: mean = {mean_value:.3f}, median = {med_value:.3f}")

    # Compute mean accuracy and kappa for outer ring and inner square
    # Mean metrics
    mean_accuracy_whole = np.mean([result['accuracy_whole'] for result in results])
    mean_kappa_whole    = np.mean([result['kappa_whole']    for result in results])
    print(f"Mean Accuracy (Whole):       {mean_accuracy_whole:.4f}, Kappa: {mean_kappa_whole:.4f}")
    mean_accuracy_outer = np.mean([result['accuracy_outer'] for result in results])
    mean_kappa_outer    = np.mean([result['kappa_outer']    for result in results])
    print(f"Mean Accuracy (Outer Ring):  {mean_accuracy_outer:.4f}, Kappa: {mean_kappa_outer:.4f}")
    mean_accuracy_inner = np.mean([result['accuracy_inner'] for result in results])
    mean_kappa_inner    = np.mean([result['kappa_inner']    for result in results])
    print(f"Mean Accuracy (Inner Square):{mean_accuracy_inner:.4f}, Kappa: {mean_kappa_inner:.4f}")
    # Median metrics
    median_accuracy_whole = np.median([result['accuracy_whole'] for result in results])
    median_kappa_whole    = np.median([result['kappa_whole']    for result in results])
    print(f"Median Accuracy (Whole):       {median_accuracy_whole:.4f}, Kappa: {median_kappa_whole:.4f}")
    median_accuracy_outer = np.median([result['accuracy_outer'] for result in results])
    median_kappa_outer    = np.median([result['kappa_outer']    for result in results])
    print(f"Median Accuracy (Outer Ring):  {median_accuracy_outer:.4f}, Kappa: {median_kappa_outer:.4f}")
    median_accuracy_inner = np.median([result['accuracy_inner'] for result in results])
    median_kappa_inner    = np.median([result['kappa_inner']    for result in results])
    print(f"Median Accuracy (Inner Square):{median_accuracy_inner:.4f}, Kappa: {median_kappa_inner:.4f}")

    # Group RMSE values by category across all realizations for each region
    rmse_whole_by_category = group_by_category(results, 'rmse_whole')
    rmse_outer_by_category = group_by_category(results, 'rmse_outer')
    rmse_inner_by_category = group_by_category(results, 'rmse_inner')

    # Group integral of the difference values by category for each region
    integral_whole_by_category = group_by_category(results, 'integral_diff_whole')
    integral_outer_by_category = group_by_category(results, 'integral_diff_outer')
    integral_inner_by_category = group_by_category(results, 'integral_diff_inner')
    
    all_vario_nugget_whole = [variogram['vario_nugget_whole'] for variogram in varios]
    all_vario_nugget_out   = [variogram['vario_nugget_out']   for variogram in varios]
    all_vario_nugget_in    = [variogram['vario_nugget_in']    for variogram in varios]
    all_vario_real_whole   = [variogram['vario_real_whole']   for variogram in varios]
    all_vario_nugget_whole = all_vario_nugget_whole[0]
    all_vario_nugget_out   = all_vario_nugget_out[0]
    all_vario_nugget_in    = all_vario_nugget_in[0]
    all_vario_real_whole   = all_vario_real_whole[0]
    
    all_rmses = compute_rmse_all_tiles(variograms_real, variograms_sim, None)
    rmse_all  = group_by_category(all_rmses)
    nugget_whole_rmse, nugget_whole_int = compute_nugget_metrics(all_vario_real_whole, all_vario_nugget_whole, None)
    nugget_outer_rmse, nugget_outer_int = compute_nugget_metrics(all_vario_real_out,   all_vario_nugget_out,   None)
    nugget_inner_rmse, nugget_inner_int = compute_nugget_metrics(all_vario_real_in,    all_vario_nugget_in,    None)
    
    # Plot histograms for RMSE
    plot_histograms(rmse_whole_by_category, 'RMSE (Whole Gap)',    'RMSE of difference', 'histogram_rmse_whole', dirPath)
    plot_histograms(rmse_outer_by_category, 'RMSE (Outer Ring)',   'RMSE of difference', 'histogram_rmse_outer', dirPath)
    plot_histograms(rmse_inner_by_category, 'RMSE (Inner Square)', 'RMSE of difference', 'histogram_rmse_inner', dirPath)

    # Plot histograms for integral of the difference
    plot_histograms(integral_whole_by_category, 'Integral of Difference (Whole Gap)', 'Integral of difference',    'histogram_integral_whole', dirPath)
    plot_histograms(integral_outer_by_category, 'Integral of Difference (Outer Ring)', 'Integral of difference',   'histogram_integral_outer', dirPath)
    plot_histograms(integral_inner_by_category, 'Integral of Difference (Inner Square)', 'Integral of difference', 'histogram_integral_inner', dirPath)

    # Plot all histograms in one figure
    plot_sub_histograms(rmse_all, nugget_whole_rmse, 'RMSE of difference', 'histogram_rmse_whole', dirPath)
    plot_sub_histograms(rmse_outer_by_category, nugget_outer_rmse, 'RMSE of difference', 'histogram_rmse_outer', dirPath)
    plot_sub_histograms(rmse_inner_by_category, nugget_inner_rmse, 'RMSE of difference', 'histogram_rmse_inner', dirPath)

    plot_sub_histograms(integral_whole_by_category, nugget_whole_int, 'Integral of difference', 'histogram_integral_whole', dirPath)
    plot_sub_histograms(integral_outer_by_category, nugget_outer_int, 'Integral of difference', 'histogram_integral_outer', dirPath)
    plot_sub_histograms(integral_inner_by_category, nugget_inner_int, 'Integral of difference', 'histogram_integral_inner', dirPath)

    mean_rmses = {}
    med_rmses = {}
    for category, values in rmse_all.items():
        mean_value = np.mean(values)
        med_value = np.median(values)
        mean_rmses[category] = mean_value
        med_rmses[category] = med_value
        print(f"Category {category}: mean = {mean_value:.3f}, median = {med_value:.3f}")
    mean_nugget = {}
    med_nugget = {}
    for category, values in nugget_whole_rmse.items():
        mean_value = np.mean(values)
        med_value = np.median(values)
        mean_nugget[category] = mean_value
        med_nugget[category] = med_value
        print(f"Nugget for category {category}: mean = {mean_value:.3f}, median = {med_value:.3f}")

    # Categories proportion
    categoriesTiles, countsTiles = np.unique(all_real_valid[~np.isnan(all_real_valid)], return_counts=True)
    proportionsTiles = countsTiles / np.sum(countsTiles)
    categoriesSim, countsSim = np.unique(all_sim_valid[~np.isnan(all_sim_valid)], return_counts=True)
    proportionsSim = countsSim / np.sum(countsSim)
    categoriesAll, countsAll = np.unique(img[:,:,0][~np.isnan(img[:,:,0]) & (img[:,:,0] != -2)], return_counts=True)
    proportionsAll = countsAll / np.sum(countsAll)

    # Create a new figure
    bar_width = 0.3
    plt.figure(figsize=(10, 6))
    # Adjust the x positions to separate the bars
    categories = np.union1d(categoriesTiles, categoriesAll)
    x_indexes = np.arange(len(categories))
    # Align the bars side-by-side
    plt.bar(x_indexes - bar_width, proportionsAll,   bar_width, edgecolor='black', label='Full image')
    plt.bar(x_indexes            , proportionsTiles, bar_width, edgecolor='black', label='Reference (Cross-validation tiles)')
    plt.bar(x_indexes + bar_width, proportionsSim,   bar_width, edgecolor='black', label='Simulations (Cross-validation tiles)')
    # Add labels and title
    plt.xlabel('Categories')
    plt.ylabel('Proportion')
    # plt.title('Proportion of Each Category (Tiles vs Full Image)')
    # Add category ticks
    plt.xticks(x_indexes, categories)
    plt.grid(True)
    plt.legend(framealpha=1)
    plt.savefig(os.path.join(dirPath, f'histogram_proportion_crossValTiles_categories.png'))
    plt.show()

    print("All results saved!")
elif plot_varios == True:
    varios = np.load(os.path.join(dirPath, 'variograms.npy'), allow_pickle=True)
    
    variograms_real   = [variogram['vario_real_whole']   for variogram in varios]
    variograms_sim    = [variogram['vario_sim_whole']    for variogram in varios]
    variograms_nugget = [variogram['vario_nugget_whole'] for variogram in varios]
    variograms_real   = variograms_real[0]
    variograms_sim    = variograms_sim[0]
    variograms_nugget = variograms_nugget[0]
    
    max_lag = metrics_max_lag
    for cat in range(-1, 6, 1):
        plt.figure(figsize=(8, 6))
        for tile_idx in variograms_sim:
            # Plot the real data variogram (red line)
            bin_centers_real, semivariance_real = variograms_real[tile_idx][cat]
            mask_real = bin_centers_real <= max_lag
            plt.plot(
                bin_centers_real[mask_real],
                semivariance_real[mask_real],
                color='red',
                linewidth=2,
                label='Real Data' if tile_idx == list(variograms_sim.keys())[0] else None
            )
            for it in variograms_sim[tile_idx]:
                if cat in variograms_sim[tile_idx][it]:
                    bin_centers, semivariance = variograms_sim[tile_idx][it][cat]
                    mask_sim = bin_centers <= max_lag
                    plt.plot(
                        bin_centers[mask_sim],
                        semivariance[mask_sim],
                        color='grey',
                        alpha=0.5
                    )
            # Plot the nugget variogram (dashed line)
            bin_centers_nugget, semivariance_nugget = variograms_nugget[tile_idx][cat]
            mask_nugget = bin_centers_nugget <= max_lag
            plt.plot(
                bin_centers_nugget[mask_nugget],
                semivariance_nugget[mask_nugget],
                color='blue',  # Adjust color as needed
                linestyle='--',
                linewidth=2,
                label='Nugget' if tile_idx == list(variograms_sim.keys())[0] else None
            )
        plt.xlabel('Lag Distance')
        plt.ylabel('Semivariance')
        plt.ylim(bottom=0)
        plt.title(f'Variograms for Category {cat}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

end_time = time.time()
total_time = end_time - start_time

# Print the total time taken
print(f"Total time: {total_time:.2f} seconds.")



n_images = 14
category_values = np.array([-1, 0, 1, 2, 3, 4, 5])
category_labels = [str(c) for c in category_values]
n_categories = len(category_values)
# Initialize storage
all_errors = {c: [] for c in category_values}
fig, axs = plt.subplots(3, 5, figsize=(20, 10))
axs = axs.flatten()
for img_idx in range(n_images):
    ax = axs[img_idx]
    # Real map
    real_map = all_real_valid[:, :, img_idx * 10]
    valid_mask = (real_map != -2) & (~np.isnan(real_map))
    cats, counts = np.unique(real_map[valid_mask], return_counts=True)
    real_props = counts / np.sum(counts)
    # Structured simulations
    sim_errors = {c: [] for c in category_values}
    for j in range(10):
        sim_map = all_sim_valid[:, :, img_idx * 10 + j]
        cats, counts = np.unique(sim_map[valid_mask], return_counts=True)
        sim_props = counts / np.sum(counts)
        for i, c in enumerate(category_values):
            err = sim_props[i] - real_props[i]
            sim_errors[c].append(err)
            all_errors[c].append(err)
    # Boxplot for structured simulations
    ax.boxplot([sim_errors[c] for c in category_values], labels=category_labels)
    ax.axhline(0, color='red', linestyle='--', lw=1)
    ax.set_title(f'Tile {img_idx+1}')
    # ax.set_ylim(-0.15, 0.25)
    ax.grid(True, linestyle=':', alpha=0.6)
    if img_idx % 5 == 0:
        ax.set_ylabel('Sim - Real Proportion')
    if img_idx >= 10:
        ax.set_xlabel('Category')
# Final subplot: overall boxplot
ax = axs[-1]
ax.boxplot([all_errors[c] for c in category_values], labels=category_labels)
ax.axhline(0, color='red', linestyle='--', lw=1)
ax.set_title('Overall')
# ax.set_ylim(-0.15, 0.25)
ax.set_xlabel('Category')
ax.grid(True, linestyle=':', alpha=0.6)
# Clean up unused axes
for i in range(n_images + 1, len(axs)):
    fig.delaxes(axs[i])
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

for tile in variograms_real:
    real_cats = variograms_real[tile]
    sim_cats = variograms_sim[tile]
    nug_cats = variograms_nugget[tile]

    for category in real_cats:
        real = real_cats[category]
        nugget = nug_cats.get(category)
        if nugget is None:
            continue

        plt.figure(figsize=(10, 6))

        # Plot nugget
        bin_centers_nugget, semivar_nugget = nugget
        plt.plot(bin_centers_nugget, semivar_nugget, 'o-', color='red', linewidth=2, label='Nugget')

        # Plot real
        bin_centers_real, semivar_real = real
        plt.plot(bin_centers_real, semivar_real, 'o-', color='black', linewidth=2, label='Real')

        # Plot all simulations
        for sim_index in sim_cats:
            sim_data = sim_cats[sim_index]
            if category in sim_data:
                bin_centers_sim, semivar_sim = sim_data[category]
                plt.plot(bin_centers_sim, semivar_sim, alpha=0.3, color='black', linewidth=1)

        plt.title(f"Tile {tile} - Category {category}")
        plt.xlabel("Lag Distance")
        plt.ylabel("Semivariance")
        plt.ylim(bottom=0)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
for tile in variograms_real:
    real_cats = variograms_real[tile]
    sim_cats = variograms_sim[tile]
    nug_cats = variograms_nugget[tile]
    
    categories = list(real_cats.keys())
    n_cats = len(categories)
    n_cols = 4
    n_rows = 2

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 9))#, sharey=True)
    fig.suptitle(f"Tile {tile} - Variograms per Category", fontsize=16)
    
    for idx, category in enumerate(categories):
        row, col = divmod(idx, n_cols)
        ax = axs[row][col]

        real = real_cats[category]
        nugget = nug_cats.get(category)
        if nugget is None:
            ax.set_title(f"Category {category} (no nugget)")
            ax.axis('off')
            continue

        # Nugget
        bin_centers_nugget, semivar_nugget = nugget
        ax.plot(bin_centers_nugget, semivar_nugget, 'o-', color='red', linewidth=2, label='Nugget')

        # Real
        bin_centers_real, semivar_real = real
        ax.plot(bin_centers_real, semivar_real, 'o-', color='black', linewidth=2, label='Real')

        # Simulations
        for sim_index in sim_cats:
            sim_data = sim_cats[sim_index]
            if category in sim_data:
                bin_centers_sim, semivar_sim = sim_data[category]
                ax.plot(bin_centers_sim, semivar_sim, alpha=0.3, color='black', linewidth=1)

        ax.set_title(f"Category {category}")
        ax.set_xlabel("Lag Distance")
        ax.set_ylabel("Semivariance")
        ax.set_ylim(bottom=0)
        ax.set_xlim(0,30)
        ax.grid(True)
        ax.legend()

    # Hide any unused subplots
    for i in range(n_cats, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axs[row][col].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make space for suptitle
    plt.show()
