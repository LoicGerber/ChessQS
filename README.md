# __chessQS__: A QS-Based Gap-Filling Framework Using Chessboard Tiling for Very Large and Complex Images

This repository provides a workflow, __chessQS__, for spatial gap-filling and simulation using the **QuickSampling (QS)** algorithm. The method subdivides the domain into overlapping chessboard-like tiles, evaluates data availability, merges poorly informed tiles, and applies QS simulations to reconstruct missing regions. Optional systematic cross-validation is included, reproducing the results used in the associated research paper (in preparation).

---

## Repository Structure
chessQS.py

  -> Main file where parameters are defined and the simulations are run

createKernel.py

  -> Function file for the creation of the kernel used by QS

chessFunctionsQS.py

  -> Functions file with all necessary functions used by __chessQS__

crossValidationQS.py

  -> Main file for cross-validation of the method, with parameters definition, simulation runs, and figure production

crossValFunctionsQS.py

  -> Functions file with all necessary functions used for the cross-validation

---

## Overview of the Workflow

### 0. Prerequisites
The QS algorithm must be installed and operational for __chessQS__ to run. __chessQS__ does not include QS; it only calls an existing QS installation.

All documentation related to QS installation, compilation, and execution is available at:
https://gaia-unil.github.io/G2S/briefOverview.html

QS can be executed locally or on a remote machine. Ensure that the QS executable is accessible before running the workflow.

### 1. Input Images

Two categorical rasters are required:

- **TI (Training Image)** — the reference map  
- **DI (Damaged Image)** — the image/map to gapfill, containing missing values (`NaN` or `−2`)

Supported formats: **GeoTIFF** and **HDF5 (.h5)**.

---

### 2. Chessboard Tiling

The domain is divided into overlapping tiles (e.g. `1000 × 1000` pixels with `100`-pixel overlap).  
Each tile is evaluated based on the proportion of informed pixels in TI and DI.

Tiles with insufficient informed pixels (e.g. < 25%) are flagged as *poorly informed*.

---

### 3. Tile Merging

Tiles that are poorly informed are iteratively merged with neighboring tiles, subject to constraints on:

- Maximum merged tile size (e.g. `2000 × 2000`)  
- Maximum number of merging iterations  

This step ensures adequate spatial context for QS simulations.

---

### 4. Kernel Construction

QS requires a multi-layer spatial kernel.  
`createKernel.py` constructs kernels with configurable:

- Layer sizes  
- Layer weights  
- Kernel types (uniform, Gaussian, exponential, or identity)  
- Gaussian sigma or exponential decay parameters  

---

### 5. QS Simulations

Each tile is simulated with QS, using TI tiles as training data and DI tiles as simulation domains.

Requirements:

- A functional installation of QS  
- QS running either locally or on a remote machine

Simulated tiles are reassembled into a full-domain reconstruction.  
Final outputs are provided in both:

- **GeoTIFF** (with CRS and pixel size metadata; e.g. EPSG:2056)  
- **HDF5**

---

### 6. Cross-Validation (Optional)

`crossValidationQS.py` performs systematic cross-validation:

- Selects tiles as synthetic gaps  
- Runs QS simulations  
- Computes metrics such as:
  - RMSE  
  - Variogram differences
  - Overall Accuracy
  - Cohen's Kappa

Cross-validation functions are implemented in `crossValFunctionsQS.py`.

*Note: this part of the workflow is not fully refactored but reproduces the results used in the corresponding research paper.*

---

## Usage

### Running the Main Workflow


This performs:

1. Loading TI and DI  
2. Chessboard tiling  
3. Identification and merging of poorly informed tiles  
4. Kernel construction  
5. QS simulations  
6. Export of `.tif` and `.h5` outputs  

### Running Cross-Validation


Generates:

- Reconstructed tiles  
- Validation metrics  
- Variograms and histograms  
- Reproducible numerical outputs  

---

## Citation

If you use this workflow in research, please cite the associated paper when available.

---

## Contact

For questions or suggestions, please open an issue or contact the author.
