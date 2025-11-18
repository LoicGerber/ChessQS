import os
import numpy as np
import gstools as gs
import math
from scipy.spatial.distance import pdist
from scipy.stats import binned_statistic
from scipy.integrate import simps
from sklearn.metrics import accuracy_score, cohen_kappa_score
import matplotlib.pyplot as plt

def compute_indicator_variogram(image, sample_percentage, num_bins, model_types):
    # Initialize dictionaries to store variograms and models for each category
    variograms = {}
    fitted_models = {}
    all_models = {
        "Gaussian": gs.Gaussian,
        "Exponential": gs.Exponential,
        "Matern": gs.Matern,
        "Stable": gs.Stable,
        "Rational": gs.Rational,
        "Circular": gs.Circular,
        "Spherical": gs.Spherical,
        "SuperSpherical": gs.SuperSpherical,
        "JBessel": gs.JBessel,
    }
    # Filter models based on model_types
    models = {name: cls for name, cls in all_models.items() if model_types.get(name, False)}
    
    scores = {}
    
    # Get the unique categories in the image, excluding nan
    categories = np.unique(image)
    categories = categories[~np.isnan(categories)]
    
    # Generate coordinates for all points in the image
    coords = np.column_stack(np.nonzero(~np.isnan(image)))
    
    for category in categories:
        print(f'Computing experimental variogram for category {int(category)}...')
        # Create an indicator function for the current category (1 for the category, 0 otherwise)
        indicator_values = (image == category).astype(int)
        
        # Flatten the indicator values to match the coordinates
        indicator_values = indicator_values[~np.isnan(image)]
        
        # Calculate the number of points to sample
        num_points = int(len(coords) * sample_percentage)
        print(f'  Selected {num_points} points...')
        
        if num_points < 1:
            print(f'Warning: No value for category {category} in image. Skipping...')
            continue
        
        # Randomly sample a percentage of the points
        indices = np.random.choice(len(coords), size=num_points, replace=False)
        sampled_coords = coords[indices]
        sampled_values = indicator_values[indices]
        
        # Generate the variogram on the sampled data
        if num_bins != 0:
            bin_center, gamma = gs.vario_estimate((sampled_coords[:, 0], sampled_coords[:, 1]), sampled_values, num_bins)
            print('  Maximal bin distance:', max(bin_center))
        else:
            print('  No # bins specified. Automatic binning.')
            bin_center, gamma = gs.vario_estimate((sampled_coords[:, 0], sampled_coords[:, 1]), sampled_values)
            print('  Estimated bin number:', len(bin_center))
            print('  Maximal bin distance:', max(bin_center))
        
        # plot the estimated variogram
        plt.figure()
        plt.scatter(bin_center, gamma, color="k", label="data")
        plt.title(f'Model fitting comparison for category {int(category)}')
        plt.grid(True)
        plt.xlabel('Lag distance')
        plt.ylabel('Semivariance')
        ax = plt.gca()

        if models:
            # Initialize variables to keep track of the best model
            best_model = None
            best_score = -float('inf')
            
            # fit all models to the estimated variogram
            for model in models:
                fit_model = models[model](dim=2)
                para, pcov, r2 = fit_model.fit_variogram(bin_center, gamma, return_r2=True)
                fit_model.plot(x_max=max(bin_center), ax=ax)
                scores[model] = r2
                # Update the best model if the current one has a higher score
                if r2 > best_score:
                    best_score = r2
                    best_model = fit_model
            
            ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            print("RANKING by Pseudo-r2 score")
            for i, (model, score) in enumerate(ranking, 1):
                print(f"{i:>6}. {model:>15}: {score:.5}")

            # Fit the variogram with a stable model
            # print(f'Fitting variogram with stable model for category {int(category)}...')
            # fit_model = gs.Stable(dim=2)
            # fit_model.fit_variogram(bin_center, gamma)
            
            fitted_models[category] = best_model
        else:
            fitted_models[category] = []
        
        # Store the results in the dictionaries
        variograms[category] = (bin_center, gamma)
    
    return variograms, fitted_models

def plot_variograms(variograms_real, variograms_sim, models_real, models_sim, categories):
    for category in categories:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        
        # Plot experimental variograms and fitted models for real and simulated data
        for data_type, color, marker, model_dict, label in [('real', 'red', 'o', models_real, 'Real data'), 
                                                                   ('sim', 'black', 'x', models_sim, 'Simulated data')]:
            if category in (variograms_real if data_type == 'real' else variograms_sim):
                bin_center, gamma = (variograms_real if data_type == 'real' else variograms_sim)[category]
                scatter = ax.scatter(bin_center, gamma, marker=marker, color=color, label=label)
                
                model = model_dict[category]
                if model:
                    model_handle, = ax.plot([], [], color=color, linestyle='-', linewidth=1)
                    model.plot(ax=ax, x_max=max(bin_center), color=color, linestyle='-', linewidth=1)
                
        ax.set_title(f'Variogram for Category {int(category)}')
        ax.set_xlabel('Lag distance')
        ax.set_ylabel('Semivariance')
        ax.set_ylim([0, None])
        ax.grid(True)
        ax.legend(loc='best', fontsize='small')
        
        plt.tight_layout()
        # plt.savefig(f'variogram_category_{int(category)}.png')
    
def compute_rmse(real, sim):
    if real.shape != sim.shape:
        raise ValueError("Input images must have the same dimensions.")
    mask = ~np.isnan(real) & ~np.isnan(sim)
    rmse = np.sqrt(np.nanmean((real[mask] - sim[mask]) ** 2))

    return rmse

def compute_hamming_distance(real, sim):
    if real.shape != sim.shape:
        raise ValueError("Input images must have the same dimensions.")
    real = real.astype(int)
    sim = sim.astype(int)
    hamming_distance = np.sum(real != sim) / np.size(real)
    
    return hamming_distance

def compute_binned_variogram_OLD(image, sample_percentage, num_bins, max_lag, seed, real):
    if seed is not None:
        np.random.seed(seed)
    variograms = {}
    # Get the unique categories in the image, excluding NaN
    categories = np.unique(image[~np.isnan(image)])
    # Generate coordinates for all valid (non-NaN) points in the image
    coords = np.column_stack(np.nonzero(~np.isnan(image)))
    for category in categories:
        # print(f'Computing variogram for category {int(category)}...')
        # Create an indicator function for the current category (1 for category, 0 otherwise)
        indicator_values = (image == category).astype(int)
        real_values = (real == category).astype(int)
        # Flatten indicator values to match available coordinates
        indicator_values = indicator_values[~np.isnan(image)]
        real_values = real_values[~np.isnan(real)]
        # Calculate the number of points to sample
        num_points = max(1, int(len(coords) * sample_percentage))  # Ensure at least 1 point is sampled
        # Randomly sample a subset of the points
        indices = np.random.choice(len(coords), size=num_points, replace=False)
        sampled_coords = coords[indices]
        sampled_values = indicator_values[indices]
        # Compute pairwise distances (lags)
        pairwise_distances = pdist(sampled_coords)
        # Compute indicator semivariance directly: (I(x) - I(y))^2
        i_diff = (sampled_values[:, None] != sampled_values[None, :]) * 1
        # i_diff = sampled_values[:, None] - sampled_values[None, :]
        pairwise_semivariance = 0.5 * (i_diff ** 2)[np.triu_indices(num_points, k=1)]
        # Filter by max_lag
        valid_indices = pairwise_distances <= max_lag
        filtered_distances = pairwise_distances[valid_indices]
        filtered_semivariance = pairwise_semivariance[valid_indices]
        if len(filtered_distances) == 0:
            print(f'Warning: No pairs found within max lag for category {category}.')
            continue
        # Bin distances and compute mean semivariance per bin
        bin_edges = np.linspace(0, max_lag, num_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        binned_semivariance, _, _ = binned_statistic(filtered_distances, filtered_semivariance, statistic='mean', bins=bin_edges)
        realVar = np.nanvar(real_values)
        # Store the binned variogram for this category
        variograms[category] = (bin_centers, binned_semivariance/realVar)
    return variograms

def compute_binned_variogram(image, sample_percentage, num_bins, max_lag, seed, real=None):
    if seed is not None:
        np.random.seed(seed)

    variograms = {}
    valid_mask = ~np.isnan(image)
    coords = np.column_stack(np.nonzero(valid_mask))
    categories = np.unique(image[valid_mask])

    for category in categories:
        # Indicator values: 1 for this category, 0 elsewhere
        indicator_full = (image == category).astype(int)
        indicator = indicator_full[valid_mask]
        if real is not None:
            real_indicator_full = (real == category).astype(int)
            real_indicator = real_indicator_full[valid_mask]

        # Sample points
        num_points = max(1, int(len(coords) * sample_percentage))
        idx = np.random.choice(len(coords), size=num_points, replace=False)
        sampled_coords = coords[idx]
        sampled_values = indicator[idx]

        # Pairwise distances and semivariances
        dists = pdist(sampled_coords)
        diff_matrix = (sampled_values[:, None] != sampled_values[None, :]).astype(float)
        semivariances = 0.5 * diff_matrix[np.triu_indices(num_points, k=1)]

        # Filter for max lag
        valid = dists <= max_lag
        if not np.any(valid):
            print(f"Warning: no pairs within max_lag for category {category}")
            continue

        # Bin and normalize
        bin_edges = np.linspace(0, max_lag, num_bins+1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        binned_gamma, _, _ = binned_statistic(
            dists[valid], semivariances[valid],
            statistic="mean", bins=bin_edges
        )
        if real is not None:
            real_var = np.nanvar(real_indicator)
            
        # Normalize by variance p(1-p) of the real indicator image to obtain normalized indicator variogram
        variograms[category] = (bin_centers, (binned_gamma / real_var) if real is not None else binned_gamma)

    return variograms

def plot_binned_variogram(variograms_real, variograms_sim, outDir):
    categories_real = variograms_real.keys()
    categories_sim  = variograms_sim.keys()

    # Ensure both real and simulated variograms have the same categories
    common_categories = set(categories_real).intersection(set(categories_sim))
    
    if not common_categories:
        print("No common categories found between real and simulated variograms.")
        return

    for category in common_categories:
        outName = os.path.join(outDir,f'variograms_cat_{category}.png')
        # Extract the data for the real and simulated variograms
        bin_centers_real, binned_semivariance_real = variograms_real[category]
        bin_centers_sim,  binned_semivariance_sim  = variograms_sim[category]
        
        plt.figure()
        # Plot real variogram
        plt.plot(bin_centers_real, binned_semivariance_real, 'o', color="blue", label="Real values", alpha=0.7)
        # Plot simulated variogram
        plt.plot(bin_centers_sim,  binned_semivariance_sim,  's', color="red", label="Simulated values", alpha=0.7)
        
        # Add title and labels
        plt.title(f'Binned Variogram Comparison for category {int(category)}')
        plt.grid(True)
        plt.xlabel('Lag distance')
        plt.ylabel('Semivariance')
        plt.ylim(0, None)
        plt.legend()
        plt.savefig(outName)
        plt.show()

def compute_rmse_all_tiles(variograms_real, variograms_sim, max_lag=None):
    all_rmses = []

    for tile_index in variograms_real:
        real_categories = variograms_real[tile_index]
        sim_realisations = variograms_sim[tile_index]

        tile_rmses = []

        for realisation in sim_realisations:
            rmse = compute_rmse_per_category(real_categories, sim_realisations[realisation], max_lag)
            tile_rmses.append(rmse)

        all_rmses.append(tile_rmses)

    return all_rmses


def compute_rmse_per_category(variograms_real, variograms_sim, max_lag=None):
    rmse_per_category = {}

    for category in variograms_real.keys():
        real_lags, real_values = variograms_real[category]
        sim_lags, sim_values = variograms_sim.get(category, (None, None))

        if sim_values is None:
            print(f"Warning: Category {category} missing in simulated variograms.")
            continue

        # Remove NaNs from both real and sim values (safely)
        mask = ~np.isnan(real_values) & ~np.isnan(sim_values)
        real_lags = real_lags[mask]
        real_values = real_values[mask]
        sim_values = sim_values[mask]

        if max_lag is not None:
            lag_mask    = real_lags <= max_lag
            real_lags   = real_lags[lag_mask]
            real_values = real_values[lag_mask]
            sim_values  = sim_values[lag_mask]

        if len(real_values) == 0 or len(sim_values) != len(real_values):
            print(f"Warning: Category {category} has no valid data after filtering.")
            continue

        rmse = np.sqrt(np.nanmean((real_values - sim_values) ** 2))
        rmse_per_category[category] = rmse

    return rmse_per_category

def compute_integral_diff_per_category(variograms_real, variograms_sim):
    int_diff_category = {}
    for category in variograms_real.keys():
        real_lags, real_values = variograms_real[category]
        sim_lags, sim_values = variograms_sim.get(category, (None, None))

        if sim_values is None or len(sim_values) != len(real_values):
            print(f"Warning: Category {category} missing or size mismatch in simulated variograms.")
            continue

        # Compute integral of difference for this category
        diff = real_values - sim_values
        int_diff = simps(diff, real_lags)
        int_diff_category[category] = int_diff
    return int_diff_category

def group_by_category(results, key=None):
    grouped_by_category = {}
    if key is not None:
        for result in results:
            if key in result:
                for category, value in result[key].items():
                    grouped_by_category.setdefault(category, []).append(value)
    else:
        for tile in results:
            for realisation in tile:
                for category, value in realisation.items():
                    grouped_by_category.setdefault(category, []).append(value)

    return grouped_by_category

def plot_histograms(grouped_data, title_prefix, xlabel, file_prefix, dirPath):
    for category, values in grouped_data.items():
        plt.figure(figsize=(8, 6))
        plt.hist(values, bins=10, edgecolor='black', alpha=0.7)
        plt.title(f'{title_prefix} for Category {int(category)}')
        plt.xlabel(xlabel)
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # Save the histogram to the specified directory
        plt.savefig(os.path.join(dirPath, f'{file_prefix}_category_{int(category)}.png'))
        plt.show()

def compute_metrics(region_real, region_sim, label, sample_percentage, num_bins, max_lag, seed, real):
        print(f"{label}: Real image")
        variograms_real = compute_binned_variogram(region_real, sample_percentage, num_bins, max_lag, seed, real)
        print(f"{label}: Simulated image")
        variograms_sim = compute_binned_variogram(region_sim, sample_percentage, num_bins, max_lag, seed, real)

        # Compute RMSE for variograms
        rmse_variogram = compute_rmse_per_category(variograms_real, variograms_sim)

        # Compute integral of difference for variograms
        integral_difference = compute_integral_diff_per_category(variograms_real, variograms_sim)

        real_flat = region_real.flatten()
        sim_flat = region_sim.flatten()
        mask = ~np.isnan(real_flat) & ~np.isnan(sim_flat)
        real_valid = real_flat[mask]
        sim_valid = sim_flat[mask]

        # Compute overall accuracy
        accuracy = accuracy_score(real_valid, sim_valid)

        # Compute Cohen's Kappa
        kappa = cohen_kappa_score(real_valid, sim_valid)

        return accuracy, kappa, rmse_variogram, integral_difference, real_valid, sim_valid, variograms_real, variograms_sim

def plot_sub_histograms(grouped_data, nugget, xlabel, file_name, dirPath):
    # Determine the number of rows and columns for subplots
    num_categories = len(grouped_data)
    cols = math.ceil(math.sqrt(num_categories))
    rows = math.ceil(num_categories / cols)

    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    # Plot each category in its subplot
    for idx, (category, values) in enumerate(grouped_data.items()):
        ax = axes[idx]
        ax.hist(values, bins=20, edgecolor='black', alpha=0.7)
        # Compute mean nugget metric for this category
        mean_nugget = np.nanmean(nugget[category])

        # Add vertical line for nugget integral
        # ax.axvline(mean_nugget_integral, color='red', linestyle='dashed', linewidth=2, label='Mean Nugget Integral')

        ax.set_title(f'Category {int(category)} - Random sim {xlabel}: {mean_nugget:.3f}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency')
        ax.set_xlim(left=0)
        ax.grid(True)

    # Hide any unused subplots
    for idx in range(len(grouped_data), len(axes)):
        axes[idx].axis('off')

    # Adjust layout and save the figure
    fig.tight_layout()
    output_path = os.path.join(dirPath, f'{file_name}.png')
    plt.savefig(output_path)
    plt.show()

    return output_path

def apply_nugget_effect(image, categories, proportions, seed):
    if seed is not None:
        np.random.seed(seed)
    modified_image = image.copy()
    nan_mask = np.isnan(modified_image)
    random_values = np.random.choice(categories, size=nan_mask.sum(), p=proportions)
    modified_image[nan_mask] = random_values
    
    return modified_image

def compute_nugget_metrics(all_vario_real_whole, all_vario_nugget, max_lag=None):
    nugget_rmse = {category: [] for category in range(-1, 6)}
    nugget_int  = {category: [] for category in range(-1, 6)}

    for tile in all_vario_real_whole.keys():
        if tile not in all_vario_nugget:
            print(f"Warning: Tile {tile} not found in nugget variograms.")
            continue

        for category in all_vario_real_whole[tile].keys():
            if category not in all_vario_nugget[tile]:
                print(f"Warning: Category {category} missing in tile {tile}.")
                continue

            # Extract bin centers and semivariances
            real_lags, real_values = all_vario_real_whole[tile][category]
            sim_lags,  sim_values  = all_vario_nugget[tile].get(category, (None, None))

            # Ensure valid values
            if sim_values is None or len(real_values) != len(sim_values):
                print(f"Warning: Category {category} missing or size mismatch in tile {tile}.")
                continue
            
            # Ensure lag arrays match before integration
            if not np.array_equal(real_lags, sim_lags):
                print(f"Warning: Mismatched lag values for category {category} in tile {tile}.")
                continue
            
            # Filter out NaNs
            mask = ~np.isnan(real_values) & ~np.isnan(sim_values)
            real_lags = real_lags[mask]
            real_values = real_values[mask]
            sim_values = sim_values[mask]

            # If max_lag is provided, filter all lags <= max_lag
            if max_lag is not None:
                lag_mask    = real_lags <= max_lag
                real_lags   = real_lags[lag_mask]
                real_values = real_values[lag_mask]
                sim_values  = sim_values[lag_mask]

            # Compute RMSE
            rmse = np.sqrt(np.nanmean((real_values - sim_values) ** 2))

            # Compute integral of the absolute difference
            int_diff = simps(np.abs(real_values - sim_values), real_lags)

            # Store results in category-wise lists
            nugget_rmse[category].append(rmse)
            nugget_int[category].append(int_diff)
    
    return nugget_rmse, nugget_int


