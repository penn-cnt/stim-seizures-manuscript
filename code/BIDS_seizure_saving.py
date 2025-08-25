### SAVING SEIZURES AS BIDS FORMAT TO LEIF
"""
Script to convert seizure data to BIDS (Brain Imaging Data Structure) format.
Processes seizure recordings from iEEG data, applies minimal preprocessing, and saves
in standardized BIDS format for further analysis.
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
    Main function to process seizure data and convert to BIDS format.
    
    Workflow:
    1. Load seizure metadata from CSV
    2. Set up BIDS directory structure
    3. For each patient and seizure:
       - Extract iEEG data with buffer around seizure
       - Preprocess data (filter, downsample, clean channels)
       - Create MNE Raw object with annotations
       - Save in BIDS format
    """
    
    # Setting up BIDS targets - define BIDS directory structure and metadata
    bids_path_kwargs = {
        "root": ospj(datapath,'BIDS'),  # Root BIDS directory
        "datatype": "ieeg",             # Intracranial EEG data type
        "extension": ".edf",            # European Data Format
        "suffix": "ieeg",               # BIDS suffix for iEEG data
        "task": "ictal",                # Task name (ictal = seizure)
        "session": "clinical01",        # Clinical recording session
    }
    bids_path = BIDSPath(**bids_path_kwargs)
    
    # iEEG.org authentication credentials
    ieeg_kwargs = {
        "username": usr,
        "password_bin_file": passpath,
    }

    # Loading in all seizure data from annotation CSV
    seizures_df = pd.read_csv(ospj(metapath,"stim_seizure_information - LF_seizure_annotation.csv"))
    seizures_df.dropna(axis=0,how='all',inplace=True)  # Remove completely empty rows
    
    # Fill missing onset times with backup columns in order of preference
    seizures_df['approximate_onset'].fillna(seizures_df['UEO'],inplace=True)  # Use UEO if available
    seizures_df['approximate_onset'].fillna(seizures_df['EEC'],inplace=True)  # Then EEC
    seizures_df['approximate_onset'].fillna(seizures_df['Other_onset_description'],inplace=True)  # Finally other descriptions
    
    # Filter data: exclude high frequency stim induced seizures (stim=2) and keep only specified patients
    seizures_df = seizures_df[seizures_df.stim != 2]
    seizures_df = seizures_df[seizures_df.Patient.isin(pt_list)]
    
    # Buffer time (seconds) to save before and after seizure for context
    buffer = 120
    
    # Process each patient's seizures
    for pt, group in tqdm(
        seizures_df.groupby('Patient'),
        total=seizures_df.Patient.nunique(),
        desc="Patients",
        position=0,
    ):
        # Assign unique IEEG ID for each recording file within patient
        ieegid = group.groupby('IEEGname').ngroup().astype(int)
        seizures_df.loc[ieegid.index,'IEEGID'] = ieegid
        group.loc[ieegid.index,'IEEGID'] = ieegid
        
        # Sort seizures by recording file and onset time for consistent processing
        group = group.sort_values(["IEEGID","approximate_onset"])
        group.reset_index(inplace=True, drop=True)
        
        # Process each seizure in the patient
        for _, row in tqdm(
            group.iterrows(), total=group.shape[0], desc="seizures", position=1, leave=False
        ):
            # Skip high frequency induced seizures (defensive check)
            if row.stim == 2:
                continue
                
            # Define task names: 0=ictal (spontaneous), 1=stim (stimulation-induced)
            task_names = ['ictal','stim']
            onset = row.approximate_onset  # Seizure start time (seconds)
            offset = row.end               # Seizure end time (seconds)
            
            # Create BIDS path for this specific seizure
            sz_clip_bids_path = bids_path.copy().update(
                subject=pt,                                    # Patient ID
                run=int(row["IEEGID"]),                       # iEEG recording file number
                task=f"{task_names[int(row.stim)]}{int(onset)}", # Task with onset time appended
            )

            # Skip if file already exists and not overwriting
            if sz_clip_bids_path.fpath.exists() and not OVERWRITE:
                continue

            # Skip problematic seizure in CHOP037 (too large for processing)
            if (pt == 'CHOP037') & (onset == 962082.12):
                continue

            # Calculate seizure duration
            duration = offset-onset

            # Extract iEEG data with buffer around seizure
            data, fs = get_iEEG_data(
                iEEG_filename=row["IEEGname"],
                start_time_usec=(onset - buffer) * 1e6,  # Start buffer seconds before seizure
                stop_time_usec=(offset + buffer) * 1e6,   # End buffer seconds after seizure
                **ieeg_kwargs,
            )

            # Remove channels with flat line (constant signal) as they may cause save issues
            data = data[data.columns[data.min(axis=0) != data.max(axis=0)]]

            # Clean electrode labels for consistency
            data.columns = clean_labels(data.columns, pt=pt)
            
            # Remove scalp and EKG electrodes (keep only intracranial electrodes)
            no_scalp_labels = remove_scalp_electrodes(data.columns)
            data = data.loc[:,no_scalp_labels]

            # Remove duplicate channel labels (keep first occurrence)
            data = data.loc[:, ~data.columns.duplicated()]
            
            # Determine channel types (e.g., SEEG, ECoG) for MNE
            ch_types = check_channel_types(list(data.columns))
            ch_types.set_index("name", inplace=True, drop=True)

            # Replace NaN values with 0 for stable processing
            data.fillna(0, inplace=True)

            # Preprocessing pipeline
            data_np = data.to_numpy().T              # Convert to numpy array (channels x samples)
            data_np_notch = notch_filter(data_np,fs) # Apply notch filter (remove line noise)
            
            # Downsample to target frequency
            signal_len = int(data_np_notch.shape[1]/fs*TARGET)  # Calculate new length
            data_np_ds = sc.signal.resample(data_np_notch,signal_len,axis=1)  # Resample
            fs = TARGET  # Update sampling frequency

            # Create MNE Raw object for BIDS conversion
            data_info = mne.create_info(
                ch_names=list(data.columns), 
                sfreq=fs, 
                ch_types="eeg",  # Default to EEG, will be updated below
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
            
            # Create seizure annotation for the clip
            annots = mne.Annotations(
                onset=[buffer],        # Seizure starts after buffer period
                duration=[duration],   # Seizure duration
                description=task_names[int(row.stim)],  # Seizure type (ictal/stim)
            )

            # Add annotations to raw data and save in BIDS format
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress MNE warnings
                raw.set_annotations(annots)

                # Write to BIDS format
                write_raw_bids(
                    raw,
                    sz_clip_bids_path,
                    overwrite=OVERWRITE,
                    verbose=False,
                    allow_preload=True,
                    format="EDF",  # Save as European Data Format
                )
    
    # Save updated seizure dataframe with BIDS information
    seizures_df.to_csv(ospj(metapath,"stim_seizure_information_BIDS.csv"))

if __name__ == "__main__":
    main()
