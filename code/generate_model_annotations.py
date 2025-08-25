"""
Validation script for generating model annotations on seizure data.

This script processes seizure prediction data from machine learning models (LSTM, AbsSlp, WVNT)
and generates predicted channel annotations by applying various thresholds to probability matrices.
The script evaluates model performance across different hyperparameters and saves the results
for threshold tuning and validation analysis.
"""

# Scientific computing imports
import numpy as np
import scipy as sc
import pandas as pd
from tqdm import tqdm

# OS imports
from os.path import join as ospj
from utils import *

# Custom imports
from config import Config

def main():
    """
    Main function to generate model annotations for seizure prediction validation.
    
    Processes seizure data through multiple ML models, applies threshold sweeps,
    and identifies predicted onset/spread channels for comparison with manual annotations.
    """
    # Load configuration and metadata
    datapath, prodatapath, metapath, patient_table = Config.deal(['datapath','prodatapath','metapath','patient_table'])

    # Load seizure information and consensus annotations
    seizures_df = pd.read_csv(ospj(metapath, "stim_seizure_information_BIDS.csv"))
    annotations_df = pd.read_pickle(ospj(prodatapath, "threshold_tuning_consensus_v2.pkl"))

    # Configuration parameters
    montage = 'bipolar'
    mdl_strs = ['LSTM', 'AbsSlp', 'WVNT']  # Model types to evaluate
    
    # Hyperparameter sweep - optimal parameters based on previous tuning:
    # 10 epochs, no demin, moving mean with window size 20 on probability matrices
    for epochs in [10]:
        for demin in [False]:  # Whether to subtract minimum value
            for movtype in ['mean']:  # Type of moving window filter
                for movwin in [20]:  # Moving window size
                    for movdata in ['prob']:  # Data type to apply moving window to
                        print(f"Processing: demin={demin}, movtype={movtype}, movwin={movwin}, movdata={movdata}")
                        
                        # Initialize results dictionary
                        predicted_channels = {
                            'Patient': [],
                            'iEEG_ID': [],
                            'model': [],
                            'stim': [],
                            'approximate_onset': [],
                            'ueo_time_consensus': [],
                            'threshold': [],
                            'ueo_chs_strict': [],     # Strict UEO channel matching
                            'ueo_chs_loose': [],      # Loose UEO channel matching
                            'sec_chs_strict': [],     # Strict secondary channel matching
                            'sec_chs_loose': [],      # Loose secondary channel matching
                            'to_annotate': []
                        }
                        
                        # Process each patient
                        pbar = tqdm(patient_table.iterrows(), total=len(patient_table))
                        for _, row in pbar:
                            pt = row.ptID
                            pbar.set_description(desc=f"Patient -- {pt}", refresh=True)
                            
                            # Skip patients without interictal training data
                            if len(row.interictal_training) == 0:
                                continue
                            
                            # Only process seizures that have been annotated
                            seizure_times = seizures_df[
                                (seizures_df.Patient == pt) & (seizures_df.to_annotate == 1)
                            ]

                            # Process each seizure for this patient
                            qbar = tqdm(seizure_times.iterrows(), total=len(seizure_times), 
                                      desc='Seizures', leave=False)
                            for _, sz_row in qbar:
                                # Get BIDS data path information
                                _, _, _, _, task, run = get_data_from_bids(
                                    ospj(datapath, "BIDS"), pt, str(int(sz_row.approximate_onset)),
                                    return_path=True, verbose=0
                                )
                                
                                # Process each model type
                                for mdl_str in mdl_strs:
                                    clf_fs = 128  # Classifier sampling frequency
                                    
                                    # Construct probability matrix file path
                                    if (epochs == 100) & (mdl_str == 'LSTM'):
                                        prob_path = (f"pretrain_probability_matrix_nosmooth_nepochs-{epochs}_"
                                                   f"mdl-{mdl_str}_fs-{clf_fs}_montage-{montage}_"
                                                   f"task-{task}_run-{run}.pkl")
                                    else:
                                        prob_path = (f"pretrain_probability_matrix_nosmooth_"
                                                   f"mdl-{mdl_str}_fs-{clf_fs}_montage-{montage}_"
                                                   f"task-{task}_run-{run}.pkl")
                                    
                                    # Load seizure probability matrix
                                    sz_prob = pd.read_pickle(ospj(prodatapath, pt, prob_path))
                                    time_wins = sz_prob.time.to_numpy()
                                    sz_prob.drop('time', axis=1, inplace=True)
                                    prob_chs = sz_prob.columns.to_numpy()
                                    sz_prob = sz_prob.to_numpy().T

                                    # Apply moving window smoothing to probability data
                                    if movdata == 'prob':
                                        if movtype == 'mean':
                                            sz_prob = sc.ndimage.uniform_filter1d(
                                                sz_prob, size=movwin, mode='nearest', axis=1, origin=0
                                            )
                                        if movtype == 'med':
                                            sz_prob = sc.ndimage.median_filter(
                                                sz_prob, size=movwin, mode='nearest', axes=1, origin=0
                                            )
                                    
                                    # Apply demin (subtract minimum) if specified
                                    if demin:
                                        sz_prob = sz_prob - np.min(sz_prob)

                                    # Match seizure with consensus annotations
                                    task_time = int(task[np.where([s.isnumeric() for s in task])[0][0]:])
                                    approx_time = sz_row.approximate_onset
                                    
                                    if task_time in annotations_df.approximate_onset.astype(int):
                                        annot_row = annotations_df[
                                            annotations_df.approximate_onset.astype(int) == task_time
                                        ]
                                        consensus_time = annot_row.ueo_time_consensus.item()
                                    else:
                                        consensus_time = approx_time
                                    
                                    # Calculate time difference between annotator and approximate time
                                    time_diff = consensus_time - approx_time
                                    onset_time = 120  # Standard onset time in seconds
                                    
                                    # Find indices for consensus onset and spread times
                                    onset_index = np.argmin(np.abs((time_wins - onset_time) + time_diff))
                                    spread_index = np.argmin(np.abs((time_wins - (onset_time + 10)) + time_diff))
                                    
                                    # Sweep through threshold values (0 to 4 in 750 steps)
                                    for final_thresh in np.linspace(0, 4, 750):
                                        # Store metadata for this threshold
                                        predicted_channels['Patient'].append(sz_row.Patient)
                                        predicted_channels['iEEG_ID'].append(sz_row.IEEGname)
                                        predicted_channels['model'].append(mdl_str)
                                        predicted_channels['stim'].append(sz_row.stim)
                                        predicted_channels['approximate_onset'].append(sz_row.approximate_onset)
                                        predicted_channels['ueo_time_consensus'].append(consensus_time)
                                        predicted_channels['to_annotate'].append(sz_row.to_annotate)
                                        predicted_channels['threshold'].append(final_thresh)

                                        # Apply threshold to get binary classification
                                        sz_prob_reject = sz_prob  # Future: could reject late-seizing channels
                                        prob_chs_reject = prob_chs
                                        sz_clf = sz_prob_reject > final_thresh

                                        # Apply moving window to classification if specified
                                        if movdata == 'clf':
                                            if movtype == 'mean':
                                                sz_clf_final = sc.ndimage.uniform_filter1d(
                                                    sz_clf, size=movwin, mode='nearest', axis=1, origin=0
                                                )
                                            if movtype == 'med':
                                                sz_clf_final = sc.ndimage.median_filter(
                                                    sz_clf, size=movwin, mode='nearest', axes=1, origin=0
                                                )
                                        else:
                                            sz_clf_final = sz_clf
                                        
                                        # Identify UEO (Earliest Onset) channels
                                        # Channels that are consistently active during onset window (5 time points, or the first 2 seconds)
                                        mdl_ueo_idx = np.all(sz_clf_final[:, onset_index:onset_index+5], axis=1)
                                        mdl_ueo_ch_bp = prob_chs_reject[mdl_ueo_idx]
                                        
                                        # Extract channel names (strict = first electrode in montage, loose = all electrodes in montage)
                                        mdl_ueo_ch_strict = np.array([s.split("-")[0] for s in mdl_ueo_ch_bp]).flatten()
                                        mdl_ueo_ch_loose = np.unique(np.array([s.split("-") for s in mdl_ueo_ch_bp]).flatten())
                                        predicted_channels['ueo_chs_strict'].append(mdl_ueo_ch_strict)
                                        predicted_channels['ueo_chs_loose'].append(mdl_ueo_ch_loose)

                                        # Identify secondary/spread channels
                                        # Channels active during spread window (10 seconds after onset)
                                        mdl_sec_idx = np.all(sz_clf_final[:, spread_index:spread_index+5], axis=1)
                                        mdl_sec_ch_bp = prob_chs_reject[mdl_sec_idx]
                                        mdl_sec_ch_strict = np.array([s.split("-")[0] for s in mdl_sec_ch_bp]).flatten()
                                        mdl_sec_ch_loose = np.unique(np.array([s.split("-") for s in mdl_sec_ch_bp]).flatten())
                                        predicted_channels['sec_chs_strict'].append(mdl_sec_ch_strict)
                                        predicted_channels['sec_chs_loose'].append(mdl_sec_ch_loose)

                        # Convert results to DataFrame and save
                        predicted_channels = pd.DataFrame(predicted_channels)
                        output_filename = (f"pretrain_predicted_channels_epoch-{epochs}_min-{str(demin)}_"
                                         f"mov-{movtype}-{str(movwin)}-{movdata}_newptsv2.pkl")
                        predicted_channels.to_pickle(ospj(prodatapath, output_filename))

if __name__ == "__main__":
    main()