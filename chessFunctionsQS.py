import random
import numpy as np
import tifffile as tf
import h5py
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from g2s import g2s

def load_image(file_path):
    file_ext = os.path.splitext(file_path)[1]
    if file_ext == '.tif' or file_ext == '.tiff':
        return tf.imread(file_path)
    elif file_ext == '.h5':
        with h5py.File(file_path, 'r') as f:
            image_data = f['/image_data'][:]
            if image_data.ndim == 3:
                return np.transpose(image_data, (1, 0, 2))
            else:
                return image_data
    else:
        raise ValueError("Unsupported file format. Choose either 'tif' or 'h5'.")

def prepare_tiles(ti, di, tile_size, overlap):
    tiles = create_chessboard_tiles(ti.shape[:2], tile_size, overlap)
    tile_analysis = analyze_tiles(tiles, ti, di)
    
    # Identify different types of ignored tiles
    tiles_with_nothing = [
        idx for idx, analysis in enumerate(tile_analysis)
        if analysis['num_informed_ti'] == 0 and analysis['num_nan_di'] == 0
    ]
    tiles_no_nans_in_di = [
        idx for idx, analysis in enumerate(tile_analysis)
        if analysis['num_nan_di'] == 0
    ]
    
    # Combine all ignored tiles into a single list
    ignored_tiles = list(set(tiles_with_nothing + tiles_no_nans_in_di))
    
    return tiles, tile_analysis, ignored_tiles, tiles_with_nothing, tiles_no_nans_in_di

def create_chessboard_tiles(image_shape, tile_size, overlap):
    step = tile_size - overlap
    tiles = []
    for i in range(0, image_shape[0], step):
        for j in range(0, image_shape[1], step):
            tiles.append((i, j, min(i + tile_size, image_shape[0]), min(j + tile_size, image_shape[1])))
    return tiles

def analyze_tiles(tiles, ti, di):
    results = []
    for idx, (i_start, j_start, i_end, j_end) in enumerate(tiles):
        ti_tile = ti[i_start:i_end, j_start:j_end]
        di_tile = di[i_start:i_end, j_start:j_end]
        # Count the number of informed pixels (non-NaN) in ti
        num_informed_ti = np.count_nonzero(~np.isnan(ti_tile))
        # Count the number of NaNs in di
        num_nan_di = np.count_nonzero(np.isnan(di_tile))
        # Calculate the total number of pixels in the tile
        total_pixels = ti_tile.size
        # Calculate percentages
        percent_informed_ti = (num_informed_ti / total_pixels) * 100
        percent_nan_di = (num_nan_di / total_pixels) * 100
        results.append({
            "tile_index": idx,
            "num_informed_ti": num_informed_ti,
            "percent_informed_ti": percent_informed_ti,
            "num_nan_di": num_nan_di,
            "percent_nan_di": percent_nan_di,
            "total_pixels": total_pixels
        })
    return results

def visualize_tiles(image, tiles, ignored_tiles, tiles_with_nothing, tiles_no_nans_in_di, map, tileNum):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image, cmap=map)
    
    for idx, (i_start, j_start, i_end, j_end) in enumerate(tiles):
        rect = patches.Rectangle((j_start, i_start), j_end - j_start, i_end - i_start,
                                 linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        
        # Highlight ignored tiles in different colors
        if idx in tiles_with_nothing:
            rect.set_edgecolor('gray')
        elif idx in tiles_no_nans_in_di:
            rect.set_edgecolor('red')
        
        if tileNum == True:
            center_i = i_start + (i_end - i_start) / 2
            center_j = j_start + (j_end - j_start) / 2
            if idx in ignored_tiles:
                ax.text(center_j, center_i, f'{idx}', color='white', fontsize=8, ha='center', va='center', weight='bold')
            else:
                ax.text(center_j, center_i, f'{idx}', color='red', fontsize=8, ha='center', va='center', weight='bold')
            
    # ax.set_title('Chessboard Tiling Visualization with Ignored Tiles Highlighted')
    ax.axis('off')
    plt.show()

def visualize_filtered_chessboard(di, tiles, ignored_tiles, tile_size, overlap, map, tileNum):
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(di, cmap=map)

    # Generate the chessboard pattern
    white_tiles, black_tiles = generate_chessboard_pattern(di.shape, tiles, tile_size, overlap)
    
    # Convert ignored_tiles to a set
    ignored_tiles_set = set(ignored_tiles)

    # Filter out the ignored tiles
    white_tiles = white_tiles - ignored_tiles_set
    black_tiles = black_tiles - ignored_tiles_set

    # Highlight all tiles with their index
    for idx, (i_start, j_start, i_end, j_end) in enumerate(tiles):
        rect = patches.Rectangle((j_start, i_start), j_end - j_start, i_end - i_start,
                                 linewidth=1, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

        if tileNum == True:
            center_i = i_start + (i_end - i_start) / 2
            center_j = j_start + (j_end - j_start) / 2
            ax.text(center_j, center_i, f'{idx}', color='white', fontsize=8, ha='center', va='center', weight='bold')

    # Highlight white chessboard tiles
    for idx in white_tiles:
        (i_start, j_start, i_end, j_end) = tiles[idx]
        rect = patches.Rectangle((j_start, i_start), j_end - j_start, i_end - i_start,
                                 linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        
        if tileNum == True:
            center_i = i_start + (i_end - i_start) / 2
            center_j = j_start + (j_end - j_start) / 2
            ax.text(center_j, center_i, f'{idx}', color='lime', fontsize=8, ha='center', va='center', weight='bold')

    # Highlight black chessboard tiles
    for idx in black_tiles:
        (i_start, j_start, i_end, j_end) = tiles[idx]
        rect = patches.Rectangle((j_start, i_start), j_end - j_start, i_end - i_start,
                                 linewidth=2, edgecolor='magenta', facecolor='none')
        ax.add_patch(rect)
        
        if tileNum == True:
            center_i = i_start + (i_end - i_start) / 2
            center_j = j_start + (j_end - j_start) / 2
            ax.text(center_j, center_i, f'{idx}', color='magenta', fontsize=8, ha='center', va='center', weight='bold')

    # ax.set_title('Filtered Chessboard Tile Movement Visualization')
    ax.axis('off')
    plt.show()

def generate_chessboard_pattern(image_shape, tiles, tile_size, overlap):
    white_tiles = set()
    black_tiles = set()
    
    step = tile_size - overlap
    tiles_in_row = (image_shape[1] + step - 1) // step  # Calculate how many tiles fit in one row

    for idx in range(len(tiles)):
        row = idx // tiles_in_row
        col = idx % tiles_in_row

        if (row + col) % 2 == 0:
            white_tiles.add(idx)
        else:
            black_tiles.add(idx)
    
    return white_tiles, black_tiles

def identify_poorly_informed_tiles(image, tiles, tile_analysis, empty_tiles, ignored_tiles, threshold, plot):
    image_shape = image.shape
    # Generate maps and statistics
    (informed_map, nan_map, map_nan_gt_informed, 
            map_nan_no_informed, map_low_informed_ti, 
            map_low_informed_ti_no_nan_di,
            min_informed_ti, max_informed_ti, min_nan_di, max_nan_di,
            tiles_nan_gt_informed, tiles_nan_no_informed, tiles_low_informed_ti,
            tiles_low_informed_ti_no_nan_di, total_tiles) = generate_maps(image_shape, tiles, tile_analysis, empty_tiles, threshold)
    
    # Replace tiles_low_informed_ti calculation to ensure it contains indices
    tiles_low_informed_ti_idx = []
    tiles_nan_gt_informed_ti_idx = []
    for idx, analysis in enumerate(tile_analysis):
        if idx in empty_tiles or idx in ignored_tiles:
            continue  # Skip the tiles that are empty or ignored
        if analysis['percent_informed_ti'] < threshold:
            tiles_low_informed_ti_idx.append(idx)
        if analysis['percent_informed_ti'] < analysis['percent_nan_di']:
            tiles_nan_gt_informed_ti_idx.append(idx)

    if plot == True:
        # Define a colormap and boundaries for binary (0 or 1) values
        cmap = mcolors.ListedColormap(['black', 'red'])  # Change colors as desired
        bounds = [-0.5, 0.5, 1.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Plot 1: Percentage of Informed Pixels in TI
        plt.figure(figsize=(5, 5))
        plt.imshow(informed_map, cmap='turbo')
        plt.title(f'Percentage of Informed Pixels in TI\nMin: {min_informed_ti:.2f}%, Max: {max_informed_ti:.2f}%')
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04, label='Percentage')
        plt.show()

        # Plot 2: Percentage of NaN Pixels in DI
        plt.figure(figsize=(5, 5))
        plt.imshow(nan_map, cmap='turbo')
        plt.title(f'Percentage of NaN Pixels in DI\nMin: {min_nan_di:.2f}%, Max: {max_nan_di:.2f}%')
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04, label='Percentage')
        plt.show()

        # Plot 3: NaNs in DI > Informed in TI
        plt.figure(figsize=(5, 5))
        plt.imshow(map_nan_gt_informed, cmap=cmap, norm=norm)
        plt.title(f'NaNs in DI > Informed in TI\n{tiles_nan_gt_informed} / {total_tiles} tiles ({tiles_nan_gt_informed/total_tiles:.2%})')
        plt.axis('off')
        cbar = plt.colorbar(ticks=[0, 1], fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels(['False', 'True'])
        plt.show()

        # Plot 4: NaNs in DI but No Informed in TI
        plt.figure(figsize=(5, 5))
        plt.imshow(map_nan_no_informed, cmap=cmap, norm=norm)
        plt.title(f'NaNs in DI but No Informed in TI\n{tiles_nan_no_informed} / {total_tiles} tiles ({tiles_nan_no_informed/total_tiles:.2%})')
        plt.axis('off')
        cbar = plt.colorbar(ticks=[0, 1], fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels(['False', 'True'])
        plt.show()

        # Plot 5: Less than 25% Informed Pixels in TI
        plt.figure(figsize=(5, 5))
        plt.imshow(map_low_informed_ti, cmap=cmap, norm=norm)
        plt.title(f'Less than {threshold}% Informed Pixels in TI\n{tiles_low_informed_ti} / {total_tiles} tiles ({tiles_low_informed_ti/total_tiles:.2%})')
        plt.axis('off')
        cbar = plt.colorbar(ticks=[0, 1], fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels(['False', 'True'])
        plt.show()

        # Plot 6: Low Informed TI and NaNs in DI
        plt.figure(figsize=(5, 5))
        plt.imshow(map_low_informed_ti_no_nan_di, cmap=cmap, norm=norm)
        plt.title(f'Less than {threshold}% Informed TI and NaNs in DI\n{tiles_low_informed_ti_no_nan_di} / {total_tiles} tiles ({tiles_low_informed_ti_no_nan_di/total_tiles:.2%})')
        plt.axis('off')
        cbar = plt.colorbar(ticks=[0, 1], fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels(['False', 'True'])
        plt.show()

    # Return the indices of poorly informed tiles
    return tiles_low_informed_ti_idx, tiles_nan_gt_informed_ti_idx

def generate_maps(image_shape, tiles, tile_analysis, empty_tiles, threshold):
    informed_map = np.full(image_shape[:2], np.nan)
    nan_map = np.full(image_shape[:2], np.nan)
    map_nan_gt_informed = np.full(image_shape[:2], np.nan)
    map_nan_no_informed = np.full(image_shape[:2], np.nan)
    map_low_informed_ti = np.full(image_shape[:2], np.nan)
    map_low_informed_ti_no_nan_di = np.full(image_shape[:2], np.nan)
    
    min_informed_ti = float('inf')
    max_informed_ti = float('-inf')
    min_nan_di = float('inf')
    max_nan_di = float('-inf')
    
    tiles_nan_gt_informed = 0
    tiles_nan_no_informed = 0
    tiles_low_informed_ti = 0
    tiles_low_informed_ti_no_nan_di = 0
    
    total_tiles = 0
    
    for idx, (i_start, j_start, i_end, j_end) in enumerate(tiles):
        if idx in empty_tiles:
            continue  # Skip empty tiles
        
        total_tiles += 1
        analysis = tile_analysis[idx]
        
        # Update min/max for informed pixels in TI
        min_informed_ti = min(min_informed_ti, analysis['percent_informed_ti'])
        max_informed_ti = max(max_informed_ti, analysis['percent_informed_ti'])
        
        # Update min/max for NaN pixels in DI
        min_nan_di = min(min_nan_di, analysis['percent_nan_di'])
        max_nan_di = max(max_nan_di, analysis['percent_nan_di'])
        
        informed_map[i_start:i_end, j_start:j_end] = analysis['percent_informed_ti']
        nan_map[i_start:i_end, j_start:j_end] = analysis['percent_nan_di']
        
        # Condition 1: NaNs > Informed pixels
        if analysis['percent_nan_di'] > analysis['percent_informed_ti']:
            map_nan_gt_informed[i_start:i_end, j_start:j_end] = 1
            tiles_nan_gt_informed += 1
        else:
            map_nan_gt_informed[i_start:i_end, j_start:j_end] = 0

        # Condition 2: NaNs present but no informed pixels
        if analysis['percent_nan_di'] > 0 and analysis['percent_informed_ti'] == 0:
            map_nan_no_informed[i_start:i_end, j_start:j_end] = 1
            tiles_nan_no_informed += 1
        else:
            map_nan_no_informed[i_start:i_end, j_start:j_end] = 0

        # Condition 4: Less than threshold informed pixels in TI
        if analysis['percent_informed_ti'] < threshold:
            map_low_informed_ti[i_start:i_end, j_start:j_end] = 1
            tiles_low_informed_ti += 1
        else:
            map_low_informed_ti[i_start:i_end, j_start:j_end] = 0
        
        # less than threshold in ti and no nans in di
        if analysis['percent_informed_ti'] < threshold and analysis['percent_nan_di'] > 0:
            map_low_informed_ti_no_nan_di[i_start:i_end, j_start:j_end] = 1
            tiles_low_informed_ti_no_nan_di += 1
        else:
            map_low_informed_ti_no_nan_di[i_start:i_end, j_start:j_end] = 0
    
    return (informed_map, nan_map, map_nan_gt_informed, 
            map_nan_no_informed, map_low_informed_ti, 
            map_low_informed_ti_no_nan_di,
            min_informed_ti, max_informed_ti, min_nan_di, max_nan_di,
            tiles_nan_gt_informed, tiles_nan_no_informed, tiles_low_informed_ti,
            tiles_low_informed_ti_no_nan_di, total_tiles)

def iterative_merge_poorly_informed_tiles(ti, di, tiles, tile_analysis, poorly_informed_tiles, nan_gt_informed_tiles, empty_tiles, ignored_tiles, tile_size, overlap, threshold, maxTileSize, max_iterations):
    grid = create_tile_index_grid(ti.shape, tile_size, overlap)
    mod_ti_tiles = tiles.copy()
    mod_di_tiles = tiles.copy()

    initial_poorly_informed = poorly_informed_tiles.copy()
    initial_nan_gt_informed = nan_gt_informed_tiles.copy()
    
    initial_mod_di_tiles = mod_di_tiles.copy()
    initial_mod_ti_tiles = mod_ti_tiles.copy()

    iteration = 0

    while iteration < max_iterations and (poorly_informed_tiles or nan_gt_informed_tiles):
        print(f"ITERATION {iteration + 1}/{max_iterations}")
        print(f"    {len(poorly_informed_tiles)} poorly informed `ti` tiles")
        print(f"    {len(nan_gt_informed_tiles)} `di` tiles with more NaN than informed pixels in `ti`")
        
        # Track the number of poorly informed tiles and nan_gt_informed tiles at the start of the iteration
        initial_poorly_informed_count = len(poorly_informed_tiles)
        initial_nan_gt_informed_count = len(nan_gt_informed_tiles)
        
        initial_mod_di_tiles = mod_di_tiles.copy()
        initial_mod_ti_tiles = mod_ti_tiles.copy()
        
        merged_ti_tiles = {}
        merged_di_tiles = {}

        # Attempt to merge each poorly informed tile in `ti` with its best neighbor
        for idx in poorly_informed_tiles:
            neighbors = find_neighbors(grid, idx, tiles, empty_tiles, tile_size, overlap)

            if not neighbors:
                print(f"        No valid neighbors found for `ti` tile {idx}...")
                continue
            
            # Sort neighbors by the highest percentage of informed pixels in `ti`
            sorted_neighbors = sorted(neighbors, key=lambda n_idx: tile_analysis[n_idx]['percent_informed_ti'], reverse=True)
            i_start, j_start, i_end, j_end = mod_ti_tiles[idx]

            for best_neighbor_idx in sorted_neighbors:
                # Merge the `ti` tile with its best neighbor
                ni_start, nj_start, ni_end, nj_end = tiles[best_neighbor_idx]

                if (i_start <= ni_start <= i_end and i_start <= ni_end <= i_end and
                j_start <= nj_start <= j_end and j_start <= nj_end <= j_end):
                    # Skip to the next neighbor if within the current tile
                    continue
    
                # Create a merged tile by expanding the boundaries
                merged_tile = (
                    min(i_start, ni_start), min(j_start, nj_start),
                    max(i_end, ni_end), max(j_end, nj_end)
                )
                
                # Calculate the size of the merged tile
                merged_height = merged_tile[2] - merged_tile[0]
                merged_width = merged_tile[3] - merged_tile[1]

                # Check if the merged tile exceeds the maximum size
                if merged_height <= maxTileSize[0] and merged_width <= maxTileSize[1]:
                    merged_ti_tiles[idx] = merged_tile
                    break
                else:
                    print(f"        `ti` tile {idx} and neighbor {best_neighbor_idx} exceed the maximum allowed size after merging. Trying next best neighbor...")

        # Attempt to merge each tile in `di` that has more NaN values than informed pixels in `ti`
        for idx in nan_gt_informed_tiles:
            neighbors = find_neighbors(grid, idx, tiles, empty_tiles, tile_size, overlap)

            if not neighbors:
                print(f"        No valid neighbors found for `di` tile {idx}...")
                continue
            
            # Sort neighbors by the highest percentage of informed pixels in `ti`
            sorted_neighbors = sorted(neighbors, key=lambda n_idx: tile_analysis[n_idx]['percent_informed_ti'], reverse=True)
            i_start, j_start, i_end, j_end = mod_di_tiles[idx]

            for best_neighbor_idx in sorted_neighbors:
                # Merge the `di` tile with its best neighbor
                ni_start, nj_start, ni_end, nj_end = tiles[best_neighbor_idx]

                if (i_start <= ni_start <= i_end and i_start <= ni_end <= i_end and
                j_start <= nj_start <= j_end and j_start <= nj_end <= j_end):
                    # Skip to the next neighbor if within the current tile
                    continue
                
                # Create a merged tile by expanding the boundaries
                merged_tile = (
                    min(i_start, ni_start), min(j_start, nj_start),
                    max(i_end, ni_end), max(j_end, nj_end)
                )

                # Calculate the size of the merged tile
                merged_height = merged_tile[2] - merged_tile[0]
                merged_width = merged_tile[3] - merged_tile[1]

                # Check if the merged tile exceeds the maximum size
                if merged_height <= maxTileSize[0] and merged_width <= maxTileSize[1]:
                    merged_di_tiles[idx] = merged_tile
                    break
                else:
                    print(f"        `di` tile {idx} and neighbor {best_neighbor_idx} exceed the maximum allowed size after merging. Trying next best neighbor...")

        # Update the modified tiles with the newly merged tiles
        for idx, merged_tile in merged_ti_tiles.items():
            mod_ti_tiles[idx] = merged_tile

        for idx, merged_tile in merged_di_tiles.items():
            mod_di_tiles[idx] = merged_tile

        # Reanalyze tiles after merging
        ti_analysis = analyze_tiles(mod_ti_tiles, ti, di)
        di_analysis = analyze_tiles(mod_di_tiles, ti, di)

        # Validate tiles to ensure they now meet the threshold
        poorly_informed_tiles, _ = validate_tiles(ti_analysis, empty_tiles, ignored_tiles, threshold)
        _, nan_gt_informed_tiles = validate_tiles(di_analysis, empty_tiles, ignored_tiles, threshold)

        print(f"    {len(poorly_informed_tiles)} poorly informed `ti` tiles remaining...")
        print(f"    {len(nan_gt_informed_tiles)} `di` tiles with more NaN than informed pixels in `ti` remaining...")

        # Check if there's been any improvement
        if initial_mod_di_tiles == mod_di_tiles and initial_mod_ti_tiles == mod_ti_tiles:
            print(f"No improvement detected in iteration {iteration + 1}. Stopping early...")
            merged_poor = [idx for idx in initial_poorly_informed if idx not in poorly_informed_tiles]
            merged_nan_gt = [idx for idx in initial_nan_gt_informed if idx not in nan_gt_informed_tiles]
            print(f"    {len(merged_poor)} `ti` tiles merged: {merged_poor}")
            print(f"    {len(merged_nan_gt)} `di` tiles merged: {merged_nan_gt}")
            print(f"    {len(poorly_informed_tiles)} poorly informed `ti` tiles still do not meet the {threshold}% threshold: {poorly_informed_tiles}")
            print(f"    {len(nan_gt_informed_tiles)} `di` tiles with more NaN than informed pixels in `ti`: {nan_gt_informed_tiles}")
            break
        
        iteration += 1
        if not poorly_informed_tiles and not nan_gt_informed_tiles:
            print(f"All tiles meet the {threshold}% threshold after {iteration} iteration(s).")
            break

    if iteration >= max_iterations and (poorly_informed_tiles or nan_gt_informed_tiles):
        print(f"Max iterations reached ({max_iterations}).")
        merged_poor = [idx for idx in initial_poorly_informed if idx not in poorly_informed_tiles]
        merged_nan_gt = [idx for idx in initial_nan_gt_informed if idx not in nan_gt_informed_tiles]
        print(f"    {len(merged_poor)} `ti` tiles merged: {merged_poor}")
        print(f"    {len(merged_nan_gt)} `di` tiles merged: {merged_nan_gt}")
        print(f"    {len(poorly_informed_tiles)} poorly informed `ti` tiles still do not meet the {threshold}% threshold: {poorly_informed_tiles}")
        print(f"    {len(nan_gt_informed_tiles)} `di` tiles with more NaN than informed pixels in `ti`: {nan_gt_informed_tiles}")

    return mod_ti_tiles, mod_di_tiles

def create_tile_index_grid(image_shape, tile_size, overlap):
    height, width = image_shape[:2]
    step = tile_size - overlap
    rows = (height + step - 1) // step
    cols = (width + step - 1) // step
    
    # Initialize the grid with -1 (indicating no tile)
    grid = np.full((rows, cols), -1, dtype=int)

    idx = 0
    for row in range(rows):
        for col in range(cols):
            grid[row, col] = idx
            idx += 1

    return grid

def find_neighbors(grid, tile_idx, tiles, ignored_tiles, tile_size, overlap):
    # Get the grid index of the tile
    tile_coords = tiles[tile_idx]
    i_start, j_start, i_end, j_end = tile_coords

    # Calculate the grid coordinates
    row = i_start // (tile_size - overlap)
    col = j_start // (tile_size - overlap)
    
    neighbors = []
    
    # Possible neighbor positions (up, down, left, right)
    possible_neighbors = [
        (row-1, col),  # top
        (row+1, col),  # bottom
        (row, col-1),  # left
        (row, col+1)   # right
    ]
    
    # Validate the neighboring positions
    for r, c in possible_neighbors:
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
            neighbor_idx = r * grid.shape[1] + c
            if neighbor_idx != tile_idx and neighbor_idx not in ignored_tiles:
                neighbors.append(neighbor_idx)
    
    return neighbors

def validate_tiles(updated_analysis, empty_tiles, ignored_tiles, threshold):
    invalid_ti_tiles = []
    invalid_di_tiles = []

    # Create a set for faster lookup
    ignored_tiles_set = set(empty_tiles + ignored_tiles)

    for analysis in updated_analysis:
        idx = analysis['tile_index']
        if idx in ignored_tiles_set:
            continue  # Skip ignored tiles

        if analysis['percent_informed_ti'] < threshold:
            invalid_ti_tiles.append(idx)
        if analysis['percent_informed_ti'] < analysis['percent_nan_di']:
            invalid_di_tiles.append(idx)

    return invalid_ti_tiles, invalid_di_tiles

def visualize_modified_tiles(image, original_tiles, mod_ti_tiles, mod_di_tiles, ignored_tiles, map, tile_index):
    # Generate the list of differing tiles for `ti`
    differing_ti_tiles = [idx for idx, (orig, mod) in enumerate(zip(original_tiles, mod_ti_tiles)) 
                          if orig != mod and idx not in ignored_tiles]

    # Generate the list of differing tiles for `di`
    differing_di_tiles = [idx for idx, (orig, mod) in enumerate(zip(original_tiles, mod_di_tiles)) 
                          if orig != mod and idx not in ignored_tiles]

    # Check if tile_index is provided and is valid for `ti`
    if tile_index is not None:
        if tile_index in differing_ti_tiles:
            differing_ti_tiles = [tile_index]  # Only keep the specific differing tile for `ti`
        elif tile_index in differing_di_tiles:
            differing_di_tiles = [tile_index]  # Only keep the specific differing tile for `di`
        else:
            print(f"Tile index {tile_index} is not a differing tile in either `ti` or `di`.")
            return

    # Plot `ti` tiles
    plot_tiles(image, original_tiles, mod_ti_tiles, ignored_tiles, differing_ti_tiles, map,
               'TI')

    # Plot `di` tiles
    plot_tiles(image, original_tiles, mod_di_tiles, ignored_tiles, differing_di_tiles, map,
               'DI')

def plot_tiles(image, original_tiles, modified_tiles, ignored_tiles, differing_tiles, map, title):
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image, cmap=map)

    # Generate a colormap with as many colors as there are differing tiles
    num_differing_tiles = len(differing_tiles)
    cmap = plt.colormaps['tab20']
    color_mapping = {idx: mcolors.rgb2hex(cmap(i / max(num_differing_tiles - 1, 1))[:3]) 
                        for i, idx in enumerate(differing_tiles)}

    # Highlight the ignored tiles in gray
    # for idx in ignored_tiles:
    #     i_start, j_start, i_end, j_end = original_tiles[idx]
    #     rect = patches.Rectangle((j_start, i_start), j_end - j_start, i_end - i_start,
    #                                 linewidth=2, edgecolor='gray', facecolor='none')
    #     ax.add_patch(rect)

    # Highlight the original tiles that are not ignored in black
    # for idx, (i_start, j_start, i_end, j_end) in enumerate(original_tiles):
    #     if idx not in ignored_tiles:
    #         rect = patches.Rectangle((j_start, i_start), j_end - j_start, i_end - i_start,
    #                                     linewidth=2, edgecolor='black', facecolor='none')
    #         ax.add_patch(rect)

    # Highlight only the modified tiles that differ from the original in red (or other colors from the colormap)
    for idx in differing_tiles:
        i_start, j_start, i_end, j_end = modified_tiles[idx]
        rect = patches.Rectangle((j_start, i_start), j_end - j_start, i_end - i_start,
                                    linewidth=2, edgecolor=color_mapping[idx], facecolor='none')
        ax.add_patch(rect)

    # Add a title and show the plot
    ax.set_title(title)
    ax.axis('off')
    plt.show()

def run_simulations(ti, di, mod_ti_tiles, mod_di_tiles, tiles, tile_analysis, ignored_tiles, ki, g2s_params, tile_size, overlap):
    cumulative_simulation = di.copy()
    
    # Generate chessboard pattern
    white_tiles, black_tiles = generate_chessboard_pattern(di.shape, tiles, tile_size, overlap)
    
    # Filter out ignored tiles
    white_tiles = sorted(white_tiles - set(ignored_tiles))
    black_tiles = sorted(black_tiles - set(ignored_tiles))

    # Shuffle the order of white and black tiles randomly
    random.shuffle(white_tiles)
    random.shuffle(black_tiles)

    # Run simulations on white tiles first
    for idx in white_tiles:
        # if idx in nan_gt_informed_tiles:            # special case for when NaNs in Di > informed pixels in Ti
        #     tile_coords = mod_ti_tiles[idx]         # Use updated coordinates for ti
        #     original_coords = mod_di_tiles[idx]     # Use updated coordinates for di 
        # else:
        #     tile_coords = mod_ti_tiles[idx]         # Use updated coordinates for ti
        #     original_coords = tiles[idx]            # Use original coordinates for di
        ti_coords = mod_ti_tiles[idx]
        di_coords = mod_di_tiles[idx]
        analysis  = tile_analysis[idx]

        if analysis['num_nan_di'] == 0:
            continue

        print(f"Running simulation on white tile {idx}.")
        simulation = run_tile_simulation(ti_coords, di_coords, ti, cumulative_simulation, ki, g2s_params)

        # Update cumulative_simulation with the result of this simulation
        i_start, j_start, i_end, j_end = di_coords
        cumulative_simulation[i_start:i_end, j_start:j_end] = simulation
    
    # Run simulations on black tiles
    for idx in black_tiles:
        # if idx in nan_gt_informed_tiles:            # special case for when NaNs in Di > informed pixels in Ti
        #     tile_coords = mod_ti_tiles[idx]         # Use updated coordinates for ti
        #     original_coords = mod_di_tiles[idx]     # Use updated coordinates for di
        # else:
        #     tile_coords = mod_ti_tiles[idx]         # Use updated coordinates for ti
        #     original_coords = tiles[idx]            # Use original coordinates for di
        ti_coords = mod_ti_tiles[idx]
        di_coords = mod_di_tiles[idx]
        analysis  = tile_analysis[idx]

        if analysis['num_nan_di'] == 0:
            continue

        print(f"Running simulation on black tile {idx}.")
        simulation = run_tile_simulation(ti_coords, di_coords, ti, cumulative_simulation, ki, g2s_params)

        # Update cumulative_simulation with the result of this simulation
        i_start, j_start, i_end, j_end = di_coords
        cumulative_simulation[i_start:i_end, j_start:j_end] = simulation

    return cumulative_simulation

def run_tile_simulation(mod_coords, og_coords, ti, di, ki, params):
    mod_i_start, mod_j_start, mod_i_end, mod_j_end = mod_coords
    og_i_start, og_j_start, og_i_end, og_j_end = og_coords
    ti_tile = ti[mod_i_start:mod_i_end, mod_j_start:mod_j_end]
    di_tile = di[og_i_start:og_i_end, og_j_start:og_j_end] 
    args = ['-ti', ti_tile, '-di', di_tile, '-ki', ki]
    params_list = list(params)
    args.extend(params_list)
    simulation, index, *_ = g2s(*args)
    return simulation

