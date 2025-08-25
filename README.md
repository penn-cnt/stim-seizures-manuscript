# stim-seizures
## ⚡️ Can electrical stimulation replace spontaneous seizures?
<img src="https://github.com/penn-cnt/stim-seizures-manuscript/blob/main/cover_fig.png" width="750">

### 👨‍🍳 Authors
William K.S. Ojemann [1,2], Caren Armstrong [3,4], Akash Pattnaik [1,2], Nina Petillo [1,5], Mariam Josyula [1,5], Alexander Daum [6], Daniel J. Zhou [1,5], Joshua LaRocque [1,5,7], Jacob Korzun [5], Catherine V. Kulick-Soper [5], Eli J. Cornblath [1,5], Sarita Damaraju [1,8], Russell T. Shinohara [9,10], Eric D. Marsh [3], Kathryn A. Davis [1,2,5], Brian Litt [1,2,5], Erin C. Conrad [1,5]

### 🏦 Affiliations
1. Center for Neuroengineering and Therapeutics, University of Pennsylvania, Philadelphia, PA 19104, USA
2. Department of Bioengineering, University of Pennsylvania, Philadelphia, PA 19104, USA
3. Children's Hospital of Philadelphia, Division of Neurology, Philadelphia, PA 19104, USA
4. Department of Neurology, University of California, Davis, Sacramento, CA, 95817, USA	
5. Department of Neurology, University of Pennsylvania, Philadelphia, PA 19104, USA
6. Department of Neuroscience, University of Pennsylvania, Philadelphia, PA 19104, USA
7. Department of Neurology, Medical College of Wisconsin, Milwaukee, WI 53226, USA
8. Perelman School of Medicine, University of Pennsylvania, Philadelphia, PA 19104, USA
9. Penn Statistics in Imaging and Visualization Center, Department of Biostatistics, Epidemiology and Informatics, University of Pennsylvania, Philadelphia, PA 19104, USA
10. AI2D: Center for AI and Data Science For Integrated Diagnostics, University of Pennsylvania, Philadelphia, PA 19104, USA

### 🧪 Preprint: will be made available upon preprint!
### 🔬 Manuscript: will be made available upon publication!

## 🤖 Prerequisites/Dependencies
- **OS**: Linux (tested). macOS should work; Windows is untested.
- **Python (stim-env)**: Python 3.10.x (tested with 3.10.16). Saved artifacts use pickle protocol 5, so use Python ≥3.8 (3.10 recommended) to ensure compatibility.
- **Environment setup**:
  - Create a virtual environment (recommended name: `stim-env`).
  - Install Python deps via the pinned `requirements.txt` at the repo root.
  - Additional packages not pinned in `requirements.txt` but required by the code:
    - `ieeg` (Python client for iEEG.org). Install from source at `https://github.com/ieeg-portal/ieegpy`
- **GPU (optional)**:
  - The pinned wheels target CUDA 12.x (e.g., torch 2.2.0 + cu12 and TensorFlow 2.16.1). Ensure a compatible NVIDIA driver and CUDA runtime if using GPU. CPU-only runs are supported but slower.
- **Jupyter**: JupyterLab 4.x is included in `requirements.txt` for running notebooks.
- **MNE-BIDS**: `mne`, `mne-bids` are pinned and required for BIDS I/O and analyses.
- **Stats/ML libraries**: `numpy`, `scipy`, `scikit-learn`, `statsmodels`, `seaborn`, `matplotlib`, and others are pinned in `requirements.txt`.
- **R (for mixed-effects models and stats in `code/*.R`)**: R ≥4.1 with the following packages installed:
  - `lme4`, `lmerTest`, `pbkrtest`, `dplyr`, `emmeans`, `multcomp`, `car`, `nlme`
- **Access to iEEG.org**:
  - An iEEG.org account is required to download raw EEG.
  - The config expects `IEEG_USR` (username) and an `IEEG_PWD` path to a binary file containing the password.

## Quick start
- Create and activate the environment, then install Python deps:
  ```bash
  python3.10 -m venv stim-env
  source stim-env/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  pip install git@github.com:ieeg-portal/ieegpy.git  # if not already present in your env
  ```
- Install R dependencies (one-time):
  ```r
  install.packages(c("lme4","lmerTest","pbkrtest","dplyr","emmeans","multcomp","car","nlme","jsonlite"))
  ```
- Download data checkpoints from (link here soon!) 
- Update config.py
    - usr - *ieeg.org username*
    - passpath - *Path to ieeg.org login binary password file*
    - RAW_DATA - *Path to raw data and localizations, this folder is not required for the quick run*
    - PROCESSED_DATA - *Path to processed data checkpoints, required*
    - METADATA - *Path to patient and seizure metadata, required*
    - figures - *Path to saved figures, required*
- Run config.py
    - This creates a r_config.json file for path information in the statistical analysis scripts
- Run the following scripts/notebooks to generate the figures/analyses/statistics in the manuscript from intermediate data checkpoints
    - annotation_analysis_and_consensus.ipynb

          
## Data
all raw EEG data is publically available on iEEG.org and will be uploaded in BIDS format as a publcially available dataset on pennsive.io upon publication. The script BIDS_seizure_saving.py contains code used to save the raw EEG data into BIDS format from ieeg.org.

## Analysis pipeline
* annotation_analysis_and_consensys.ipynb
notebook to generate consensus thresholds and perform seizure similarity analysis (Figure SX). Run this first to generate threshold_tuning_consensus_v2.pkl, which contains the consensus annotations for each annotated seizure.
* BIDS_seizure_saving.py
script to pull seizure data from iEEG.org and save it in iEEG-BIDS format using mne python. Saves all seizures in seizure_information-LF that are spontaneous (stim == 0) or low-frequency stim-induced (stim == 1)
* BIDS_interictal_saving.py
script to pull the designated interictal data from iEEG.org and save it in iEEG-BIDS format using mne python. Saves all interictal clips as designated in the config, one for each patient.
* seizure_detection_pretrain.py
script to generate seizure detection values for each spontaneous and stimulation induced seizure using the NDD model and two benchmarks: Absolute slope, and a wavenet-based univariate seizure detector.
* val_generate_model_annotations.py
script to generate ueo channel and 10 second spread channel annotations for each seizure that had human annotations at each threshold in a parameter sweep.
* val_threshold_sweep_merging.py
script for merging human annotations and model annotations at each of the swept thresholds
* tuning_model_thresholds.ipynb
notebook containing code to visualize model performances and compare them to model performances
* lme_model_analysis.R
* val_generate_optimized_model_annotations.py
* model_prediction_analysis_cursor.ipynb
* lme_recruitment_analysis.R
* lme_semiology_mtle_analysis.R
* lme_spont_spread_analysis.R
* utils
    * utils.py
    * stim_seizure_preprocessing_utils.py
* visualizations
    * end2end_sandbox.ipynb
