### SAVING INTERICTAL TRAINING DATA IN BIDS TO LEIF
"""
Script to convert interictal (non-seizure) data to BIDS format for training.
Processes baseline/background iEEG recordings, applies minimal preprocessing, 
and saves in standardized BIDS format for machine learning model training.
"""

import numpy as np
import pandas as pd
from os.path import join as ospj
from utils import *
import scipy as sc
import warnings

from tqdm import tqdm

# BIDS imports
import mne
from mne_bids import BIDSPath, write_raw_bids

# Loading CONFIG - get paths and patient information
from config import Config
usr,passpath,datapath,prodatapath,metapath,figpath,patient_table,rid_hup,pt_list = Config.deal()

# Setting Seed for reproducibility
np.random.seed(171999)

# Target sampling frequency after downsampling
TARGET = 512
# Whether to overwrite existing BIDS files
OVERWRITE = False

def main():
    """
    Main function to process interictal data and convert to BIDS format.
    
    Workflow:
    1. Load seizure metadata to get IEEG IDs for consistency
    2. Set up BIDS directory structure for interictal data
    3. For each patient with interictal training data:
       - Extract 60-second interictal clips from specified recordings
       - Apply minimal preprocessing (downsample only, no filtering)
       - Create MNE Raw object with interictal annotations
       - Save in BIDS format
    """
    
    # Setting up BIDS targets - define BIDS directory structure and metadata
    bids_path_kwargs = {
        "root": ospj(datapath,'BIDS'),    # Root BIDS directory
        "datatype": "ieeg",               # Intracranial EEG data type
        "extension": ".edf",              # European Data Format
        "suffix": "ieeg",                 # BIDS suffix for iEEG data
        "task": "interictal",             # Task name (interictal = non-seizure)
        "session": "clinical01",          # Clinical recording session
    }
    bids_path = BIDSPath(**bids_path_kwargs)
    
    # iEEG.org authentication credentials
    ieeg_kwargs = {
        "username": usr,
        "password_bin_file": passpath,
    }

    # Loading seizure data to get consistent IEEG IDs across ictal and interictal data
    seizures_df = pd.read_csv(ospj(metapath,"stim_seizure_information_BIDS.csv"))

    # Process each patient's interictal training data
    for _,row in tqdm(
        patient_table.iterrows(),
        total=len(patient_table),
        desc="Patients",
        position=0,
    ):
        # Skip patients without interictal training data specified
        if len(row.interictal_training) == 0:
            continue
            
        pt = row.ptID
        
        # Extract interictal training parameters from patient table
        ieeg_name = row.interictal_training[0]  # iEEG filename for interictal data
        onset = row.interictal_training[1]      # Start time for interictal clip (seconds)
        offset = onset + 60                     # End time (60-second clips)
        
        # Get consistent IEEG ID from seizure data to maintain run numbering across tasks
        # Note: Using mode() to handle potential multiple entries for same recording
        ieegid = int(seizures_df.loc[seizures_df.IEEGname == ieeg_name,'IEEGID'].mode())
        
        # Create BIDS path for this interictal clip
        clip_bids_path = bids_path.copy().update(
            subject=pt,                        # Patient ID
            run=ieegid,                        # iEEG recording file number (consistent with seizure data)
            task=f"interictal{int(onset)}",    # Task with onset time appended
        )

        # Skip if file already exists and not overwriting
        if clip_bids_path.fpath.exists() and not OVERWRITE:
            continue

        # Calculate clip duration (should be 60 seconds)
        duration = offset-onset

        # Extract iEEG data for the specified interictal period
        data, fs = get_iEEG_data(
            iEEG_filename=ieeg_name,
            start_time_usec= onset * 1e6,   # Start time in microseconds
            stop_time_usec= offset * 1e6,   # End time in microseconds
            **ieeg_kwargs,
        )

        # Remove channels with flat line (constant signal) as they may cause save issues
        data = data[data.columns[data.min(axis=0) != data.max(axis=0)]]

        # Clean electrode labels for consistency
        data.columns = clean_labels(data.columns, pt=pt)

        # Remove duplicate channel labels (keep first occurrence)
        data = data.loc[:, ~data.columns.duplicated()]
        
        # Determine channel types (e.g., SEEG, ECoG) for MNE
        ch_types = check_channel_types(list(data.columns))
        ch_types.set_index("name", inplace=True, drop=True)

        # Replace NaN values with 0 for stable processing
        data.fillna(0, inplace=True)

        # Minimal preprocessing
        data_np = data.to_numpy().T                                    # Convert to numpy array (channels x samples)
        signal_len = int(data_np.shape[1]/fs*TARGET)                   # Calculate new length for downsampling
        data_np_ds = sc.signal.resample(data_np,signal_len,axis=1)     # Downsample to target frequency
        fs = TARGET                                                    # Update sampling frequency

        # Create MNE Raw object for BIDS conversion
        data_info = mne.create_info(
            ch_names=list(data.columns), 
            sfreq=fs, 
            ch_types= "seeg",  # Default to SEEG, will be updated below
            verbose=False
        )
        
        # Create Raw object (convert µV to V for MNE standard)
        raw = mne.io.RawArray(
            data_np_ds / 1e6,  # Convert microvolts to volts
            data_info,
            verbose=False,
        )
        
        # Set correct channel types based on electrode analysis
        raw.set_channel_types(ch_types.type)
        
        # Create interictal annotation for the entire clip
        annots = mne.Annotations(
            onset=0,              # Annotation starts at beginning of clip
            duration=[duration],  # Entire clip duration (60 seconds)
            description="interictal",  # Mark as interictal (non-seizure) data
        )

        # Add annotations to raw data and save in BIDS format
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress MNE warnings
            raw.set_annotations(annots)

            # Write to BIDS format
            write_raw_bids(
                raw,
                clip_bids_path,
                overwrite=OVERWRITE,
                verbose=False,
                allow_preload=True,
                format="EDF",  # Save as European Data Format
            )

if __name__ == "__main__":
    main()