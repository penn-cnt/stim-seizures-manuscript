"""
This script has utility functions for the iEEG data analysis.
It has the following functions:
- get_iEEG_data
- clean_labels
- check_channel_types
- get_channel_types
- get_channel_labels
- get_channel_coords
- get_rpath
- surgical_parcellation
- surgical_parcelate
- get_cnt_inventory
- get_pt_coords
- get_data_from_bids
- _shade_y_ticks_background
- plot_iEEG_data
- make_surf_transforms
- cohens_d
- notch_filter
- bandpass_filter
- artifact_removal
- detect_bad_channels
- num_wins
- bipolar_montage
- ar_one
- preprocess_for_detection
- remove_scalp_electrodes
- MovingWinClips
- dice_score
- set_seed
- in_parallel
- calculate_seizure_similarity
- calculate_spread_similarity
- plot_seizure_similarity
"""
# %%
# pylint: disable-msg=C0103
# pylint: disable-msg=W0703

# standard imports
import os
from os.path import join as ospj
import pickle
from numbers import Number
import time
import re
from glob import glob
import logging
import warnings
import random
from joblib import Parallel, delayed

# data IO imports
import mne_bids
from mne_bids import BIDSPath, read_raw_bids

# nonstandard imports
from ieeg.auth import Session
import pandas as pd
import numpy as np
import torch

from scipy.signal import sosfiltfilt, butter, filtfilt

import scipy as sc

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator

# Agreement metrics
from sklearn.metrics import cohen_kappa_score, f1_score, matthews_corrcoef
warnings.filterwarnings("ignore")

########################################## Data I/O ##########################################
def _pull_iEEG(ds, start_usec, duration_usec, channel_ids):
    """
    Pull iEEG data with automatic retry logic to handle connection errors.
    
    This function attempts to retrieve iEEG data from a dataset with built-in error handling
    and retry logic. It will retry up to 50 times with 1-second delays between attempts.
    
    Args:
        ds: iEEG dataset object with get_data method
        start_usec (int): Start time in microseconds
        duration_usec (int): Duration in microseconds
        channel_ids (list): List of channel indices to retrieve
        
    Returns:
        numpy.ndarray: Retrieved iEEG data, or None if all retry attempts failed
        
    Notes:
        - Logs error message if all 50 retry attempts fail
        - Uses 1-second sleep between retry attempts
    """
    i = 0
    while True:
        if i == 50:
            logger = logging.getLogger()
            logger.error(
                f"failed to pull data for {ds.name}, {start_usec / 1e6}, {duration_usec / 1e6}, {len(channel_ids)} channels"
            )
            return None
        try:
            data = ds.get_data(start_usec, duration_usec, channel_ids)
            return data
        except Exception as _:
            time.sleep(1)
            i += 1


def get_iEEG_data(
    username: str,
    password_bin_file: str,
    iEEG_filename: str,
    start_time_usec: float,
    stop_time_usec: float,
    select_electrodes=None,
    ignore_electrodes=None,
    outputfile=None,
    force_pull = False
):
    """
    Retrieve iEEG data from the iEEG.org portal with flexible electrode selection.
    
    This function connects to iEEG.org, opens a dataset, and retrieves data for specified
    time ranges and electrodes. It handles large data requests by automatically chunking
    the data either temporally (for long time periods) or spatially (for many channels).
    
    Args:
        username (str): iEEG.org username
        password_bin_file (str): Path to file containing iEEG.org password
        iEEG_filename (str): Name of the iEEG dataset on iEEG.org
        start_time_usec (float): Start time in microseconds
        stop_time_usec (float): Stop time in microseconds
        select_electrodes (list, optional): List of electrode names or indices to include.
            Can be strings (electrode names) or integers (electrode indices)
        ignore_electrodes (list, optional): List of electrode names or indices to exclude.
            Can be strings (electrode names) or integers (electrode indices)
        outputfile (str, optional): Path to save the data as a pickle file
        force_pull (bool): If True, continues even if some selected electrodes are missing
        
    Returns:
        tuple: (pandas.DataFrame, float) containing:
            - DataFrame with iEEG data (time x channels)
            - Sampling frequency in Hz
            
    Notes:
        - Automatically chunks data for large requests:
            * Temporal chunking for clips > 120 seconds with < 100 channels
            * Channel chunking for > 100 channels 
        - Uses clean_labels() to standardize electrode names
        - Retries connection up to 50 times with 1-second delays
        
    Raises:
        ValueError: If dataset cannot be opened after 50 attempts
        ValueError: If selected electrodes not found and force_pull=False
    """
    start_time_usec = int(start_time_usec)
    stop_time_usec = int(stop_time_usec)
    duration = stop_time_usec - start_time_usec

    with open(password_bin_file, "r") as f:
        pwd = f.read()

    iter = 0
    while True:
        try:
            if iter == 50:
                raise ValueError("Failed to open dataset")
            s = Session(username, pwd)
            ds = s.open_dataset(iEEG_filename)
            all_channel_labels = ds.get_channel_labels()
            break
            
        except Exception as e:
            time.sleep(1)
            iter += 1
    all_channel_labels = clean_labels(all_channel_labels, iEEG_filename)
    
    if select_electrodes is not None:
        if isinstance(select_electrodes[0], Number):
            channel_ids = select_electrodes
            channel_names = [all_channel_labels[e] for e in channel_ids]
        elif isinstance(select_electrodes[0], str):
            select_electrodes = clean_labels(select_electrodes, iEEG_filename)
            if any([i not in all_channel_labels for i in select_electrodes]):
                if force_pull:
                    select_electrodes = [e for e in select_electrodes
                                          if e in all_channel_labels]
                else:
                    raise ValueError("Channel not in iEEG")

            channel_ids = [
                i for i, e in enumerate(all_channel_labels) if e in select_electrodes
            ]
            channel_names = select_electrodes
        else:
            print("Electrodes not given as a list of ints or strings")

    elif ignore_electrodes is not None:
        if isinstance(ignore_electrodes[0], int):
            channel_ids = [
                i
                for i in np.arange(len(all_channel_labels))
                if i not in ignore_electrodes
            ]
            channel_names = [all_channel_labels[e] for e in channel_ids]
        elif isinstance(ignore_electrodes[0], str):
            ignore_electrodes = clean_labels(ignore_electrodes, iEEG_filename)
            channel_ids = [
                i
                for i, e in enumerate(all_channel_labels)
                if e not in ignore_electrodes
            ]
            channel_names = [
                e for e in all_channel_labels if e not in ignore_electrodes
            ]
        else:
            print("Electrodes not given as a list of ints or strings")

    else:
        channel_ids = np.arange(len(all_channel_labels))
        channel_names = all_channel_labels

    # if clip is small enough, pull all at once, otherwise pull in chunks
    if (duration < 120 * 1e6) and (len(channel_ids) < 100):
        data = _pull_iEEG(ds, start_time_usec, duration, channel_ids)
    elif (duration > 120 * 1e6) and (len(channel_ids) < 100):
        # clip is probably too big, pull chunks and concatenate
        clip_size = 60 * 1e6

        clip_start = start_time_usec
        data = None
        while clip_start + clip_size < stop_time_usec:
            if data is None:
                data = _pull_iEEG(ds, clip_start, clip_size, channel_ids)
            else:
                new_data = _pull_iEEG(ds, clip_start, clip_size, channel_ids)
                data = np.concatenate((data, new_data), axis=0)
            clip_start = clip_start + clip_size

        last_clip_size = stop_time_usec - clip_start
        new_data = _pull_iEEG(ds, clip_start, last_clip_size, channel_ids)
        data = np.concatenate((data, new_data), axis=0)
    else:
        # there are too many channels, pull chunks and concatenate
        channel_size = 20
        channel_start = 0
        data = None
        while channel_start + channel_size < len(channel_ids):
            if data is None:
                data = _pull_iEEG(
                    ds,
                    start_time_usec,
                    duration,
                    channel_ids[channel_start : channel_start + channel_size],
                )
            else:
                new_data = _pull_iEEG(
                    ds,
                    start_time_usec,
                    duration,
                    channel_ids[channel_start : channel_start + channel_size],
                )
                data = np.concatenate((data, new_data), axis=1)
            channel_start = channel_start + channel_size

        last_channel_size = len(channel_ids) - channel_start
        new_data = _pull_iEEG(
            ds,
            start_time_usec,
            duration,
            channel_ids[channel_start : channel_start + last_channel_size],
        )
        data = np.concatenate((data, new_data), axis=1)

    df = pd.DataFrame(data, columns=channel_names)
    fs = ds.get_time_series_details(ds.ch_labels[0]).sample_rate  # get sample rate

    if outputfile:
        with open(outputfile, "wb") as f:
            pickle.dump([df, fs], f)
    else:
        return df, fs


def clean_labels(channel_li: list, pt: str) -> list:
    """
    Standardize and clean electrode channel labels for consistent naming across patients.
    
    This function standardizes electrode naming conventions by:
    - Removing hyphens and standardizing grid names
    - Extracting lead names and contact numbers using regex
    - Applying patient-specific naming conventions and mappings
    - Formatting contact numbers with zero-padding
    
    Args:
        channel_li (list): List of raw channel labels from the iEEG dataset
        pt (str): Patient identifier (e.g., 'HUP75_phaseII', 'sub-RID0065')
        
    Returns:
        list: List of cleaned and standardized channel labels
        
    Notes:
        - Uses regex pattern to extract lead name and contact number
        - Applies patient-specific transformations for known naming inconsistencies
        - Standardizes contact numbers to 2-digit zero-padded format (e.g., "01", "02")
        - Handles special cases like bipolar references and grid electrodes
        
    Patient-specific transformations:
        - HUP75: "Grid" -> "G"
        - HUP78: "Grid" -> "LG" 
        - HUP86: Multiple lead name mappings
        - HUP93: Grid electrode standardization
        - HUP89: Grid and lead name mappings
        - HUP99: "G" -> "RG"
        - HUP112: Preserves bipolar reference format
        - HUP116: Removes hyphens
        - HUP123: Lead name mappings for "RS" and "GTP"
        - HUP189: "LG" -> "LGr"
    """
    new_channels = []
    for i in channel_li:
        i = i.replace("-", "")
        i = i.replace("GRID", "G")  # mne has limits on channel name size
        # standardizes channel names
        pattern = re.compile(r"([A-Za-z0-9]+?)(\d+)$")
        regex_match = pattern.match(i)

        if regex_match is None:
            new_channels.append(i)
            continue

        # if re.search('Cz|Fz|C3|C4|EKG',i):
        #     continue
        lead = regex_match.group(1).replace("EEG", "").strip()
        contact = int(regex_match.group(2))

        if pt in ("HUP75_phaseII", "HUP075", "sub-RID0065"):
            if lead == "Grid":
                lead = "G"

        if pt in ("HUP78_phaseII", "HUP078", "sub-RID0068"):
            if lead == "Grid":
                lead = "LG"

        if pt in ("HUP86_phaseII", "HUP086", "sub-RID0018"):
            conv_dict = {
                "AST": "LAST",
                "DA": "LA",
                "DH": "LH",
                "Grid": "LG",
                "IPI": "LIPI",
                "MPI": "LMPI",
                "MST": "LMST",
                "OI": "LOI",
                "PF": "LPF",
                "PST": "LPST",
                "SPI": "RSPI",
            }
            if lead in conv_dict:
                lead = conv_dict[lead]
        
        if pt in ("HUP93_phaseII", "HUP093", "sub-RID0050"):
            if lead.startswith("G"):
                lead = "G"
    
        if pt in ("HUP89_phaseII", "HUP089", "sub-RID0024"):
            if lead in ("GRID", "G"):
                lead = "RG"
            if lead == "AST":
                lead = "AS"
            if lead == "MST":
                lead = "MS"

        if pt in ("HUP99_phaseII", "HUP099", "sub-RID0032"):
            if lead == "G":
                lead = "RG"

        if pt in ("HUP112_phaseII", "HUP112", "sub-RID0042"):
            if "-" in i:
                new_channels.append(f"{lead}{contact:02d}-{i.strip().split('-')[-1]}")
                continue
        if pt in ("HUP116_phaseII", "HUP116", "sub-RID0175"):
            new_channels.append(f"{lead}{contact:02d}".replace("-", ""))
            continue

        if pt in ("HUP123_phaseII_D02", "HUP123", "sub-RID0193"):
            if lead == "RS": 
                lead = "RSO"
            if lead == "GTP":
                lead = "RG"
        
        new_channels.append(f"{lead}{contact:02d}")

        if pt in ("HUP189", "HUP189_phaseII", "sub-RID0520"):
            conv_dict = {"LG": "LGr"}
            if lead in conv_dict:
                lead = conv_dict[lead]
                
    return new_channels


def get_apn_dkt(
    fname="/mnt/sauce/littlab/users/pattnaik/ictal_patterns/data/metadata/apn_dkt_labels.txt",
) -> dict:
    """
    Load antsPyNet DKT (Desikan-Killiany-Tourville) atlas labels from text file.
    
    Parses a text file containing DKT atlas region labels and their corresponding
    numerical IDs, returning a dictionary mapping region IDs to region names.
    
    Args:
        fname (str): Path to the antsPyNet DKT labels text file
        
    Returns:
        dict: Dictionary mapping region IDs (int) to region names (str)
        
    Notes:
        - Expects text file format with lines starting with "Label"
        - Parses lines like "Label 1001: Left-cerebral-white-matter"
        - Used for mapping segmentation values to anatomical region names
        
    Raises:
        ValueError: If file format is unexpected or cannot be parsed
    """
    with open(fname, "r") as f:
        lines = f.readlines()

    dkt = {}
    for line in lines:
        if line.startswith("Label"):
            words = line.strip().split()
            reg_id = int(words[1][:-1])
            reg_name = " ".join(words[2:])
            dkt[reg_id] = reg_name

    return dkt


def check_channel_types(ch_list, threshold=24):
    """
    Classify electrode channels by type (ECoG, sEEG, EEG, ECG, misc) based on naming patterns.
    
    Analyzes channel names to determine electrode types using lead name patterns and
    contact counts. Channels with many contacts (>threshold) are classified as ECoG,
    while those with fewer contacts are classified as sEEG.
    
    Args:
        ch_list (list): List of channel names to classify
        threshold (int, optional): Contact count threshold for ECoG vs sEEG classification. 
            Defaults to 24.
            
    Returns:
        pandas.DataFrame: DataFrame with columns:
            - name: Original channel name
            - lead: Lead/electrode name (e.g., "LG", "RH")
            - contact: Contact number on the lead
            - type: Channel type ("ecog", "seeg", "eeg", "ecg", "misc")
            
    Notes:
        - Uses regex to extract lead names and contact numbers
        - EEG channels identified by standard 10-20 system names (C, Cz, F, etc.)
        - ECG channels identified by "ECG" or "EKG" in name
        - ECoG vs sEEG distinction based on contact count per lead
        - Channels that don't match patterns are classified as "misc"
    """
    ch_df = []
    for i in ch_list:
        regex_match = re.match(r"([A-Za-z0-9]+)(\d{2})$", i)
        if regex_match is None:
            ch_df.append({"name": i, "lead": i, "contact": 0, "type": "misc"})
            continue
        lead = regex_match.group(1)
        contact = int(regex_match.group(2))
        ch_df.append({"name": i, "lead": lead, "contact": contact})
    ch_df = pd.DataFrame(ch_df)
    for lead, group in ch_df.groupby("lead"):
        if lead in ["ECG", "EKG"]:
            ch_df.loc[group.index, "type"] = "ecg"
            continue
        if lead in [
            "C",
            "Cz",
            "CZ",
            "F",
            "Fp",
            "FP",
            "Fz",
            "FZ",
            "O",
            "P",
            "Pz",
            "PZ",
            "T",
        ]:
            ch_df.loc[group.index.to_list(), "type"] = "eeg"
            continue
        if len(group) > threshold:
            ch_df.loc[group.index.to_list(), "type"] = "ecog"
        else:
            ch_df.loc[group.index.to_list(), "type"] = "seeg"
    return ch_df

def electrode_wrapper(pt,rid_hup,datapath):
    """
    Wrapper function to load electrode localizations for either HUP or CHOP patients.
    
    This function routes electrode localization loading to the appropriate method
    based on patient type (HUP vs CHOP), handling different data formats and
    directory structures for each hospital system.
    
    Args:
        pt (str): Patient identifier (e.g., "HUP075" or "CHOP001")
        rid_hup (pandas.DataFrame): Mapping table between HUP subject numbers and RIDs
        datapath (str): Base path to patient data directories
        
    Returns:
        tuple: (electrode_localizations, electrode_regions) where:
            - electrode_localizations: DataFrame with matter type labels (gray/white matter)
            - electrode_regions: DataFrame with anatomical region labels
            
    Notes:
        - For HUP patients: Uses IEEG_recon pipeline output with optimize_localizations()
        - For CHOP patients: Uses Excel files with choptimize_localizations()
        - Handles RID lookup for HUP patients using rid_hup mapping table
        - Searches multiple possible directory structures for recon data
        - Returns standardized format regardless of source data format
    """
    if pt[:3] == 'HUP':
        hup_no = pt[3:]
        rid = rid_hup[rid_hup.hupsubjno == hup_no].record_id.to_numpy()[0]
        rid = str(rid)
        if len(rid) < 4:
            rid = '0' + rid
        recon_path = ospj('/mnt','sauce','littlab','data',
                            'Human_Data','CNT_iEEG_BIDS',
                            f'sub-RID{rid}','derivatives','ieeg_recon',
                            'module3/')
        if not os.path.exists(recon_path):
            recon_path =  ospj('/mnt','sauce','littlab','data',
                            'Human_Data','recon','BIDS_penn',
                            f'sub-RID{rid}','derivatives','ieeg_recon',
                            'module3/')
        electrode_localizations,electrode_regions = optimize_localizations(recon_path,rid)
        return electrode_localizations,electrode_regions
    else:
        recon_path = ospj(datapath,pt,f'{pt}_locations.xlsx')
        electrode_localizations,electrode_regions = choptimize_localizations(recon_path,pt)
        return electrode_localizations,electrode_regions

def optimize_localizations(path_to_recon,RID):
    """
    Optimize electrode localization labels using tissue type and anatomical probabilities.
    
    This function refines electrode localizations from the IEEG_recon pipeline by:
    1. Loading probability data for both tissue types (atropos) and brain regions (DKT)
    2. Applying probabilistic thresholds to assign more accurate labels
    3. Prioritizing gray matter over other tissue types when overlap exceeds 5%
    4. Ensuring white matter electrodes are properly labeled regardless of region
    
    Args:
        path_to_recon (str): Path to IEEG_recon module3 output directory
        RID (str): Patient RID number (e.g., "0031")
        
    Returns:
        tuple: (modified_atropos_df, modified_dkt_df) where:
            - modified_atropos_df: DataFrame with optimized tissue type labels
            - modified_dkt_df: DataFrame with optimized anatomical region labels
            
    Notes:
        - Uses 5% overlap threshold for tissue type assignment
        - Prioritizes gray matter > white matter > other tissue types
        - Preserves white matter labels even in anatomical regions
        - Handles both line-delimited and standard JSON formats
        - Uses helper functions _apply_matter_function and _apply_region_function
        
    File inputs:
        - *_atlas-atropos_*_coordinates.json: Tissue probability data
        - *_atlas-DKTantspynet_*_coordinates.json: Anatomical region probability data
    """
    try:
        atropos_probs = pd.read_json(path_to_recon + f'sub-RID{RID}_ses-clinical01_space-T00mri_atlas-atropos_radius-2_desc-vox_coordinates.json',lines=True)
        dkt_probs = pd.read_json(path_to_recon + f'sub-RID{RID}_ses-clinical01_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.json',lines=True)
    except:
        dkt_probs = pd.read_json(path_to_recon + f'sub-RID{RID}_ses-clinical01_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.json')
        atropos_probs = pd.read_json(path_to_recon + f'sub-RID{RID}_ses-clinical01_space-T00mri_atlas-atropos_radius-2_desc-vox_coordinates.json')

    def _apply_matter_function(x):
        """
        Apply tissue type assignment based on probability thresholds.
        
        Helper function that assigns tissue labels (gray matter, white matter)
        based on overlap probabilities above 5% threshold.
        """
        # look in labels sorted and see if it contains gray matter
        # if gray matter is greater than 5% then set label to gray matter
        x = pd.DataFrame(x).transpose()
        for i,label in enumerate(x['labels_sorted'].to_numpy()[0]):
            if (label == 'gray matter') and (x['percent_assigned'].to_numpy()[0][i] > 0.05):
                x['label'] = label
                x['index'] = 2
                break
            elif (label == 'white matter') and (x['percent_assigned'].to_numpy()[0][i] > 0.05):
                x['label'] = label
                x['index'] = 3
                break
        
        return x
    def _apply_region_function(x):
        """
        Apply anatomical region assignment based on probability thresholds.
        
        Helper function that assigns anatomical region labels based on
        overlap probabilities above 5% threshold, excluding EmptyLabel.
        """
        # look in labels sorted and see if it contains gray matter
        # if gray matter is greater than 5% then set label to gray matter
        x = pd.DataFrame(x).transpose()
        for i,label in enumerate(x['labels_sorted'].to_numpy()[0]):
            if (label != 'EmptyLabel') and (x['percent_assigned'].to_numpy()[0][i] > 0.05):
                x['label'] = label
                break
        return x
        
    modified_atropos = atropos_probs.iloc[:,:].apply(lambda x: _apply_matter_function(x), axis = 1)
    modified_atropos_df = pd.DataFrame(np.squeeze(np.array(modified_atropos.to_list())),columns=atropos_probs.columns)

    modified_dkt = dkt_probs.iloc[:,:].apply(lambda x: _apply_region_function(x),axis = 1)
    modified_dkt_df = pd.DataFrame(np.squeeze(np.array(modified_dkt.to_list())),columns=dkt_probs.columns)
    modified_dkt_df[modified_atropos_df.label == 'white matter']['label'] = 'white matter'

    return modified_atropos_df,modified_dkt_df

def choptimize_localizations(recon_path,chopid):
    """
    Load and standardize electrode localizations for CHOP (Children's Hospital) patients.
    
    This function processes electrode localization data from Excel files used at CHOP,
    converting the format to match the standardized output format used for HUP patients.
    
    Args:
        recon_path (str): Path to the Excel file containing electrode locations
        chopid (str): CHOP patient identifier
        
    Returns:
        tuple: (electrode_locals, electrode_regions) where:
            - electrode_locals: DataFrame with tissue type information (gray/white matter)
            - electrode_regions: DataFrame with anatomical region information
            
    Notes:
        - Reads Excel file with patient-specific sheet name
        - Standardizes channel names using clean_labels()
        - Maps "Unknown" brain areas to "EmptyLabel" for consistency
        - Creates separate DataFrames for tissue types and anatomical regions
        - Standardizes column names and formats to match HUP pipeline output
        
    Column mappings:
        - "grey"/"white" matter -> "gray matter"/"white matter" labels
        - brain_area -> anatomical region labels
        - Preserves coordinate information (x, y, z)
    """
    electrode_locals = pd.read_excel(recon_path,f'{chopid}_locations')
    electrode_locals.loc[:,'name'] = clean_labels(electrode_locals.full_label,chopid)
    electrode_locals.loc[:,'index'] = pd.NA
    electrode_locals.loc[electrode_locals.brain_area == 'Unknown','brain_area'] = 'EmptyLabel'
    electrode_regions = electrode_locals.copy()
    electrode_regions["label"] = electrode_locals["brain_area"]
    col_list = ["name","x","y","z","label","isgrey"]
    electrode_regions = electrode_regions[col_list]
    mapping = {"grey": "gray matter","white": "white matter","Unknown": "EmptyLabel"}
    electrode_locals["label"] = electrode_locals["matter"].replace(mapping)
    col_list = ["name","x","y","z","label","isgrey"]
    electrode_locals = electrode_locals[col_list]
    return electrode_locals,electrode_regions

def get_rpath(prodatapath,pt):
    """
    Generate the file path for electrode localization pickle files based on patient type.
    
    This function returns the appropriate file path for saved electrode localization
    data, using different naming conventions for CHOP vs other patients.
    
    Args:
        prodatapath (str): Base path to processed data directory
        pt (str): Patient identifier
        
    Returns:
        str: Full file path to the electrode localization pickle file
        
    Notes:
        - CHOP patients use "electrode_localizations_CHOPR.pkl"
        - All other patients use "electrode_localizations_dkt.pkl"
        - Used for consistent file naming across different patient types
    """
    if pt[:3] == 'CHO':
        region_path = ospj(prodatapath,pt,'electrode_localizations_CHOPR.pkl')
    else:
        region_path = ospj(prodatapath,pt,'electrode_localizations_dkt.pkl')
    return region_path

def surgical_parcellation(electrode_regions):
    """
    Assign electrodes to clinically relevant surgical regions based on anatomical labels.
    
    This function groups detailed anatomical region labels into broader surgical categories
    that are clinically meaningful for epilepsy surgery planning. It provides a simplified
    anatomical classification focused on surgical targets.
    
    Args:
        electrode_regions (pandas.DataFrame): DataFrame with electrode anatomical labels
        
    Returns:
        pandas.DataFrame: Modified DataFrame with surgical region labels replacing
            detailed anatomical labels
            
    Surgical region mappings:
        - "EmptyLabel": Areas with no clear anatomical assignment or white matter
        - "left/right mesial temporal": Amygdala and hippocampus regions
        - "left/right temporal neocortex": Temporal, fusiform, entorhinal, parahippocampal
        - "left/right other neocortex": All other cortical regions
        
    Notes:
        - Preserves laterality (left vs right) from original labels
        - Handles NaN/float values by converting to "EmptyLabel"
        - Case-insensitive string matching for robust label assignment
        - Modifies DataFrame in-place and returns the modified version
    """
    for i,row in electrode_regions.iterrows():
        label = row.label
        if isinstance(label,float):
            label = "EmptyLabel"
        label = label.lower()
        if ("emptylabel" in label) or ("white" in label):
            surgical_label = "EmptyLabel"
        elif ("amygdala" in label) or ("hippocampus" in label):
            if "left" in label:
                surgical_label = 'left mesial temporal'
            else:
                surgical_label = 'right mesial temporal'
        elif ("temporal" in label) or ("fusiform" in label) or ("entorhinal" in label) or ("parahippocampal" in label):
            if "left" in label:
                surgical_label = 'left temporal neocortex'
            else:
                surgical_label = 'right temporal neocortex'
        else:
            if "left" in label:
                surgical_label = 'left other neocortex'
            else:
                surgical_label = 'right other neocortex'
        electrode_regions.loc[i,"label"] = surgical_label
    return electrode_regions
    
def surgical_parcelate(region_list):
    """
    Convert a list of anatomical region labels to surgical region categories.
    
    This function applies the same surgical parcellation logic as surgical_parcellation()
    but operates on a simple list of region names rather than a full DataFrame.
    
    Args:
        region_list (list): List of anatomical region label strings
        
    Returns:
        list: List of corresponding surgical region labels
        
    See surgical_parcellation() for detailed mapping rules.
        
    Notes:
        - Functional equivalent to surgical_parcellation() for list inputs
        - Useful when you only need to convert labels without electrode coordinates
        - Handles the same edge cases (NaN values, case sensitivity)
    """
    surgical_labels = []
    for label in region_list:
        if isinstance(label,float):
            label = "EmptyLabel"
        label = label.lower()
        if ("emptylabel" in label) or ("white" in label):
            surgical_label = "EmptyLabel"
        elif ("amygdala" in label) or ("hippocampus" in label):
            if "left" in label:
                surgical_label = 'left mesial temporal'
            else:
                surgical_label = 'right mesial temporal'
        elif ("temporal" in label) or ("fusiform" in label) or ("entorhinal" in label) or ("parahippocampal" in label):
            if "left" in label:
                surgical_label = 'left temporal neocortex'
            else:
                surgical_label = 'right temporal neocortex'
        else:
            if "left" in label:
                surgical_label = 'left other neocortex'
            else:
                surgical_label = 'right other neocortex'
        surgical_labels.append(surgical_label)
    return surgical_labels
######################## BIDS ########################
BIDS_DIR = "/mnt/leif/littlab/data/Human_Data/CNT_iEEG_BIDS"
BIDS_INVENTORY = "/mnt/leif/littlab/users/pattnaik/ieeg_recon/migrate/cnt_ieeg_bids.csv"


def get_cnt_inventory(bids_inventory=BIDS_INVENTORY):
    """
    Load and process the CNT iEEG BIDS dataset inventory.
    
    This function reads a CSV file that tracks which datasets and processing steps
    are available for each patient in the CNT iEEG BIDS directory structure.
    
    Args:
        bids_inventory (str, optional): Path to the BIDS inventory CSV file.
            Defaults to global BIDS_INVENTORY constant.
            
    Returns:
        pandas.DataFrame: Boolean DataFrame indicating data availability, where:
            - Rows represent patients/datasets
            - Columns represent different data types or processing steps
            - Values are True/False indicating availability ("yes"/"no" in CSV)
            
    Notes:
        - Converts string values ("yes"/"no") to boolean (True/False)
        - Used to check data availability before processing
        - Helps identify which patients have complete datasets
    """
    inventory = pd.read_csv(bids_inventory, index_col=0)
    inventory = inventory == "yes"
    return inventory


def get_pt_coords(pt):
    """
    Load electrode coordinates for a specific patient from BIDS directory structure.
    
    This function locates and loads the DKT antsPyNet coordinate file for a patient
    from the standardized BIDS derivatives directory structure.
    
    Args:
        pt (str): Patient identifier (e.g., "sub-RID0031")
        
    Returns:
        pandas.DataFrame: DataFrame containing electrode coordinates and anatomical labels
        
    Notes:
        - Searches in BIDS_DIR/pt/derivatives/ieeg_recon/module3/
        - Uses glob to find DKT coordinate CSV files 
        - Assumes standardized BIDS naming convention for coordinate files
        - Returns DataFrame with electrode positions and DKT atlas labels
        
    Raises:
        IndexError: If no coordinate file found for the patient
        FileNotFoundError: If BIDS directory structure doesn't exist
    """
    coords_path = glob(
        ospj(BIDS_DIR, pt, "derivatives", "ieeg_recon", "module3", "*DKTantspynet*csv")
    )[0]
    return pd.read_csv(coords_path, index_col=0)

def get_data_from_bids(root,subject,task_key,run=None,return_path=False,verbose=0):
    """
    Load iEEG data from BIDS-formatted dataset with automatic task and run detection.
    
    This function provides a high-level interface to load iEEG data from BIDS-structured
    datasets using MNE-BIDS. It automatically identifies the appropriate task and run
    based on partial string matching and returns both the data and sampling frequency.
    
    Args:
        root (str): Path to the BIDS dataset root directory
        subject (str): Subject identifier (without "sub-" prefix)
        task_key (str): Partial task name to search for (e.g., "rest" matches "task-rest")
        run (str, optional): Specific run identifier. If None, uses first available run.
        return_path (bool, optional): If True, also returns BIDS path components.
            Defaults to False.
        verbose (int, optional): MNE verbosity level. Defaults to 0 (quiet).
        
    Returns:
        tuple: Depending on return_path parameter:
            - If return_path=False: (data_df, fs) where:
                * data_df: DataFrame with iEEG data (channels as columns)
                * fs: Sampling frequency in Hz
            - If return_path=True: (data_df, fs, root, subject, task, run)
                * Additional path components for reference
                
    Notes:
        - Automatically filters out other subjects when searching for tasks/runs
        - Uses first matching task that contains task_key substring
        - Assumes session "clinical01" (standard for clinical iEEG data)
        - Removes time column from MNE DataFrame output
        - Calculates sampling frequency from time differences
        
    Example:
        >>> data, fs = get_data_from_bids("/data/bids", "RID0031", "ictal")
        >>> # Loads ictal task data for subject sub-RID0031
    """
    # setting subject
    all_subjects = mne_bids.get_entity_vals(root,'subject')
    ignore_subjects = [s for s in all_subjects if s != subject]

    # extracting task
    task_list = mne_bids.get_entity_vals(root, 'task', ignore_subjects=ignore_subjects)
    task = [t for t in task_list if task_key in t][0]
    ignore_tasks = [t for t in task_list if t != task]

    # Should be just one run per task
    if run is None:
        run = mne_bids.get_entity_vals(root, 'run', 
                                    ignore_tasks = ignore_tasks,
                                    ignore_subjects=ignore_subjects)[0]
    
    bidspath = BIDSPath(root=root,subject=subject,task=task,run=run,session='clinical01')
    data_raw = read_raw_bids(bidspath,verbose=verbose)
    data_df = data_raw.to_data_frame()
    fs = 1/data_df.time.diff().mode().item()
    if return_path:
        return data_df.drop('time',axis=1), int(fs), root, subject, task, run
    return data_df.drop('time',axis=1), int(fs)

################################################ Plotting and Visualization ################################################
def _shade_y_ticks_background(ax, y_ticks, colors, alpha=1):
    """
    Add colored background shading to a plot based on y-axis tick positions.
    
    This function creates horizontal colored bands behind plot data, with each band
    centered on a y-tick position. Useful for visually grouping or highlighting
    different data series in multi-channel plots.
    
    Args:
        ax (matplotlib.axes.Axes): The axis object to modify
        y_ticks (list or array): Y-tick values to center the shading bands on
        colors (list, array, or str): Colors for each shading band. Can be:
            - List of colors (one per y-tick)
            - Single color string (applied to all y-ticks)
            - None values skip shading for that y-tick
        alpha (float, optional): Transparency level (0=transparent, 1=opaque). 
            Defaults to 1.
            
    Notes:
        - Shading extends halfway to adjacent y-ticks on each side
        - For edge ticks, extends same distance as to nearest neighbor
        - Uses axhspan() to create horizontal spans across full x-axis width
        - Automatically sorts y-ticks to ensure proper ordering
        
    Raises:
        ValueError: If y_ticks and colors arrays have different lengths (when colors is a list)
        
    Example:
        >>> fig, ax = plt.subplots()
        >>> y_positions = [0, 1, 2, 3]
        >>> channel_colors = ['red', 'blue', None, 'green']  # No shading for channel 2
        >>> shade_y_ticks_background(ax, y_positions, channel_colors, alpha=0.3)
    """
    if isinstance(colors,str):
        colors = [colors]*len(y_ticks)
    if len(y_ticks) != len(colors):
        raise ValueError("The length of y_ticks and colors must be the same.")

    # Sort y_ticks and colors together to ensure proper ordering
    sorted_indices = np.argsort(y_ticks)
    y_ticks = np.array(y_ticks)[sorted_indices]
    colors = np.array(colors)[sorted_indices]

    # Add shading between each pair of y-ticks
    for i in range(len(y_ticks)):
        if colors[i] is not None:
            if i == 0:
                # First tick: shade from this tick down to halfway to the next tick
                lower_bound = y_ticks[i] - (y_ticks[i + 1] - y_ticks[i]) / 2
            else:
                # Shade from halfway between this tick and the previous one
                lower_bound = (y_ticks[i - 1] + y_ticks[i]) / 2

            if i == len(y_ticks) - 1:
                # Last tick: shade up to halfway to the previous tick
                upper_bound = y_ticks[i] + (y_ticks[i] - y_ticks[i - 1]) / 2
            else:
                # Shade up to halfway between this tick and the next one
                upper_bound = (y_ticks[i] + y_ticks[i + 1]) / 2

            # Add a colored rectangle spanning the full x-axis width, avoiding overlap
            ax.axhspan(lower_bound, upper_bound, color=colors[i], alpha=alpha, linewidth=0)

def plot_iEEG_data(
    data,#: Union[pd.DataFrame, np.ndarray], 
    fs=None,
    t=None,
    t_offset=0,
    colors=None,
    plot_color = 'k',
    shade_color = None,
    shade_alpha = 0.3,
    empty=False,
    dr=None,
    fig_size=None,
    minmax=False
):
    """
    Create a multi-channel iEEG data plot with customizable styling and layout.
    
    This function generates a standard "waterfall" or "butterfly" plot for multi-channel
    iEEG data, where each channel is plotted at a different y-offset for easy visualization.
    Supports various customization options including color coding, background shading,
    and automatic scaling.
    
    Args:
        data (pandas.DataFrame or numpy.ndarray): iEEG data matrix
            - If DataFrame: Uses column names as channel labels
            - Shape should be (time_points, channels) or (channels, time_points)
        fs (float, optional): Sampling frequency in Hz. Required if t is None.
        t (numpy.ndarray, optional): Time vector. If None, generated from fs.
        t_offset (float, optional): Time offset to add to time vector. Defaults to 0.
        colors (list, optional): List of colors for y-tick labels (channel names).
        plot_color (str, optional): Color for the data traces. Defaults to 'k' (black).
        shade_color (list, optional): Colors for background shading. See shade_y_ticks_background().
        shade_alpha (float, optional): Alpha transparency for background shading. Defaults to 0.3.
        empty (bool, optional): If True, removes plot borders and ticks for clean appearance.
        dr (float, optional): Vertical spacing between channels. If None, auto-calculated.
        fig_size (tuple, optional): Figure size (width, height). If None, auto-calculated.
        minmax (bool, optional): If True, z-score normalizes the data. Defaults to False.
        
    Returns:
        tuple: (fig, ax) - matplotlib figure and axis objects
        
    Notes:
        - Automatically transposes data if dimensions don't match time vector
        - Auto-calculates figure size based on duration and channel count
        - Channel labels appear as y-tick labels (if DataFrame input)
        - Supports both normalized and raw data display
        - Can overlay colored background shading for channel grouping
        
    Example:
        >>> data = pd.DataFrame(ieeg_data, columns=channel_names)
        >>> fig, ax = plot_iEEG_data(data, fs=500, colors=channel_colors)
        >>> plt.show()
    """
    if minmax:
        data = data.apply(sc.stats.zscore)
    if t is None:
        t = np.arange(len(data))/fs

    t += t_offset
    if data.shape[0] != np.size(t):
        data = data.T

    n_rows = data.shape[1]
    duration = t[-1] - t[0]
    
    if fig_size is not None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig, ax = plt.subplots(figsize=(duration / 3, n_rows / 5))
        
    sns.despine()

    ticklocs = []
    ax.set_xlim(t[0], t[-1])

    dmin = data.min().min()
    dmax = data.max().min()

    if dr is None:
        dr = (dmax - dmin) * 0.8  # Crowd them a bit.
    # if minmax and (dr is None):
    #     dr = 1

    y0 = dmin - dr
    y1 = (n_rows-1) * dr + dmax + dr/2
    ax.set_ylim([y0,y1])
    segs = []
    
    for i in range(n_rows):
        if isinstance(data, pd.DataFrame):
            segs.append(np.column_stack((t, data.iloc[:, i])))
        elif isinstance(data, np.ndarray):
            segs.append(np.column_stack((t, data[:, i])))
        else:
            print("Data is not in valid format")

    for i in reversed(range(n_rows)):
        ticklocs.append(i * dr)

    offsets = np.zeros((n_rows, 2), dtype=float)
    offsets[:, 1] = ticklocs

    # # Set the yticks to use axes coordinates on the y axis
    ax.set_yticks(ticklocs)
    if isinstance(data, pd.DataFrame):
        ax.set_yticklabels(data.columns)

    if colors:
        for col, lab in zip(colors, ax.get_yticklabels()):
            if col is None:
                col = 'black'
            lab.set_color(col)

    ax.set_xlabel("Time (s)")

    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    ax.plot(t, data + ticklocs, color=plot_color, lw=0.4)

    if shade_color is not None:    
        _shade_y_ticks_background(ax, ticklocs, shade_color, alpha=shade_alpha)

    if empty:
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Optionally, remove grid lines if present
        ax.grid(False)

        # Keep tick labels but remove tick markers
        ax.tick_params(axis='both', which='both', length=0)

    return fig, ax

def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size between two groups.
    
    Cohen's d is a standardized measure of effect size that indicates the magnitude
    of difference between two groups in terms of standard deviations. It's useful
    for interpreting the practical significance of statistical differences.
    
    Args:
        group1 (array-like): First group of values
        group2 (array-like): Second group of values
        
    Returns:
        float: Cohen's d effect size
        
    Effect size interpretation:
        - Small effect: |d| ≈ 0.2
        - Medium effect: |d| ≈ 0.5  
        - Large effect: |d| ≈ 0.8
        - Positive d: group1 mean > group2 mean
        - Negative d: group1 mean < group2 mean
        
    Notes:
        - Uses pooled standard deviation for denominator
        - Applies Bessel's correction (ddof=1) for sample standard deviation
        - Assumes approximately normal distributions for interpretation
        
    Formula:
        d = (mean1 - mean2) / pooled_std
        where pooled_std = sqrt(((n1-1)*std1² + (n2-1)*std2²) / (n1+n2-2))
        
    Example:
        >>> group1 = [1, 2, 3, 4, 5]
        >>> group2 = [3, 4, 5, 6, 7] 
        >>> effect_size = cohens_d(group1, group2)
        >>> print(f"Cohen's d: {effect_size:.3f}")
    """
    # Calculating means of the two groups
    mean1, mean2 = np.mean(group1), np.mean(group2)
     
    # Calculating pooled standard deviation
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
     
    # Calculating Cohen's d
    d = (mean1 - mean2) / pooled_std
     
    return d

################################################ Preprocessing ################################################
def notch_filter(data: np.ndarray, fs: float) -> np.array:
    """
    Apply notch filtering to remove 60Hz and 120Hz power line noise from iEEG data.
    
    This function removes electrical interference from power lines using bandstop filters
    centered at 60Hz (fundamental frequency) and 120Hz (first harmonic). Uses 
    zero-phase filtfilt for no temporal distortion.
    
    Args:
        data (numpy.ndarray): Input iEEG data array
            - Shape: (channels, time_points) or (time_points, channels)
        fs (float): Sampling frequency in Hz
        
    Returns:
        numpy.ndarray: Filtered data with same shape as input
        
    Notes:
        - Uses 4th-order Butterworth bandstop filters
        - Filter ranges: 58-62Hz and 118-122Hz  
        - Zero-phase filtering preserves temporal relationships
        - Applied along last axis (-1) by default
        - Filters are applied sequentially (60Hz first, then 120Hz)
        
    Filter specifications:
        - 60Hz notch: 58-62Hz stopband
        - 120Hz notch: 118-122Hz stopband  
        - Both use 4th-order Butterworth design
        
    Example:
        >>> clean_data = notch_filter(raw_data, fs=500)
        >>> # Removes 60Hz and 120Hz power line interference
    """
    # remove 60Hz noise
    # b, a = iirnotch(60, 15, fs)
    # d, c = iirnotch(120, 15, fs)
    b, a = butter(4,(58,62),'bandstop',fs=fs)
    d, c = butter(4,(118,122),'bandstop',fs=fs)

    data_filt = filtfilt(b, a, data, axis=-1)
    data_filt_filt = filtfilt(d, c, data_filt, axis = -1)
    # TODO: add option for causal filter
    # TODO: add optional argument for order

    return data_filt_filt


def bandpass_filter(data: np.ndarray, fs: float, order=3, lo=1, hi=150) -> np.array:
    """
    Apply bandpass filtering to iEEG data to retain specific frequency ranges.
    
    This function applies a Butterworth bandpass filter to isolate frequency content
    of interest while removing low-frequency drift and high-frequency noise.
    Uses zero-phase filtering to preserve temporal relationships.
    
    Args:
        data (numpy.ndarray): Input iEEG data array
            - Shape: (channels, time_points) or (time_points, channels)  
        fs (float): Sampling frequency in Hz
        order (int, optional): Filter order (higher = steeper rolloff). Defaults to 3.
        lo (int, optional): Low-frequency cutoff in Hz. Defaults to 1.
        hi (int, optional): High-frequency cutoff in Hz. Defaults to 150.
        
    Returns:
        numpy.ndarray: Bandpass filtered data with same shape as input
        
    Notes:
        - Uses scipy.signal.butter for Butterworth filter design
        - Zero-phase sosfiltfilt preserves phase relationships
        - Applied along last axis (-1) by default
        - Cutoff frequencies are -3dB points
        - Higher order = steeper transition but more ripple
        
    Common frequency bands:
        - Broadband: 1-150Hz (default)
        - Clinical: 1-70Hz  
        - Research: 0.5-500Hz
        - Gamma: 30-100Hz
        - High gamma: 70-150Hz
        
    Example:
        >>> # Filter for high gamma band
        >>> hg_data = bandpass_filter(data, fs=1000, lo=70, hi=150)
        >>> # Filter for clinical analysis  
        >>> clinical_data = bandpass_filter(data, fs=500, lo=1, hi=70)
    """
    # TODO: add causal function argument
    # TODO: add optional argument for order
    sos = butter(order, [lo, hi], output="sos", fs=fs, btype="bandpass")
    data_filt = sosfiltfilt(sos, data, axis=-1)
    return data_filt


def artifact_removal(
    data: np.ndarray, fs: float, discon=1 / 12, noise=15000, win_size=1
) -> np.ndarray:
    """
    Detect and mark artifacts in iEEG data using amplitude and noise thresholds.
    
    This function identifies artifacts by detecting periods of disconnection 
    (very low amplitude) and excessive noise (high derivative) in windowed segments.
    Returns a boolean mask indicating artifact-contaminated time points.
    
    Args:
        data (numpy.ndarray): Input iEEG data array
            - Shape: (time_points, channels)
        fs (float): Sampling frequency in Hz
        discon (float, optional): Disconnection threshold (low amplitude). 
            Defaults to 1/12 ≈ 0.083.
        noise (int, optional): High-frequency noise threshold. Defaults to 15000.
        win_size (int, optional): Window size for analysis in seconds. Defaults to 1.
        
    Returns:
        numpy.ndarray: Boolean artifact mask with same shape as input
            - True: Artifact detected
            - False: Clean data
            
    Artifact detection criteria:
        1. Disconnection: Sum of absolute values < discon threshold in window
        2. High-frequency noise: RMS of differences > noise threshold in window
        3. NaN values: Automatically marked as artifacts
        
    Notes:
        - Processes data in non-overlapping windows
        - Artifacts in any window affect entire window duration
        - Designed for clinical iEEG with typical amplitude ranges
        - Thresholds may need adjustment for different recording systems
        - NaN values are preserved and marked as artifacts
        
    Window-based processing:
        - Divides data into win_size second windows
        - Applies thresholds within each window
        - Marks entire window as artifact if criteria met
        
    Example:
        >>> artifact_mask = artifact_removal(data, fs=500, win_size=2)
        >>> clean_data = data.copy()
        >>> clean_data[artifact_mask] = np.nan  # Mark artifacts as NaN
    """
    win_size = int(win_size * fs)
    
    n_wins = np.ceil(data.shape[0]/win_size)
    max_inds = n_wins*win_size
     
    all_inds = np.arange(max_inds)
    all_inds[data.shape[0]:] = np.nan
    ind_overlap = np.reshape(all_inds, (-1, int(win_size)))
    
    artifacts = np.empty_like(data)

    # mask indices with nan values
    artifacts = np.isnan(data)

    for win_inds in ind_overlap:
        win_inds = win_inds[~np.isnan(win_inds)].astype(int)
        is_disconnected = np.sum(np.abs(data[win_inds,:]), axis=0) < discon

        is_noise = (
            np.sqrt(np.sum(np.power(np.diff(data[win_inds,:], axis=0), 2), axis=0))
            > noise
        )

        artifacts[win_inds, :] = np.logical_or(
            artifacts[win_inds, :].any(axis=0), np.logical_or(is_disconnected, is_noise)
        )

    return artifacts


def detect_bad_channels(data,fs,lf_stim = False):
    """
    Automatically identify bad channels in iEEG data using multiple artifact detection criteria.
    
    This comprehensive function identifies problematic channels based on various
    artifact signatures including disconnection, excessive noise, flat-lining,
    high 60Hz content, and extreme variance. Designed for clinical iEEG recordings.
    
    Args:
        data (numpy.ndarray): iEEG data array
            - Shape: (time_points, channels)
            - Units: typically microvolts (μV)
        fs (float): Sampling frequency in Hz
        lf_stim (bool, optional): If True, relaxes criteria for stimulation datasets.
            Defaults to False.
            
    Returns:
        tuple: (channel_mask, details) where:
            - channel_mask: Boolean array (True = good channel, False = bad channel)
            - details: Dictionary with lists of bad channels by category
            
    Bad channel detection criteria:
        1. NaN channels: >50% NaN values
        2. Zero channels: >50% zero values  
        3. Flat channels: >2% consecutive identical values + high amplitude
        4. High voltage: >10 samples above 5000μV threshold
        5. High variance: Values in 99th percentile with extreme outliers
        6. High 60Hz: >70% power in 58-62Hz range
        7. High std: Standard deviation >10x median across channels
        
    Args details:
        - lf_stim=True: Relaxes high voltage and high variance criteria
        - Designed for μV-scale data (multiplies by 1000 internally)
        - Uses FFT analysis for 60Hz noise detection
        
    Returns details dictionary keys:
        - 'noisy': High 60Hz content channels
        - 'nans': High NaN content channels  
        - 'zeros': High zero content channels
        - 'flat': Flat-lining channels
        - 'var': High variance channels
        - 'higher_std': Extremely high standard deviation channels
        - 'high_voltage': High amplitude artifact channels
        
    Example:
        >>> mask, bad_info = detect_bad_channels(data*1e6, fs=500)  # Convert to μV
        >>> print(f"Rejected {sum(~mask)} channels:")
        >>> for category, channels in bad_info.items():
        >>>     if channels: print(f"  {category}: {len(channels)} channels")
    """
    values = data.copy()
    which_chs = np.arange(values.shape[1])
    ## Parameters to reject super high variance
    tile = 99
    mult = 10
    num_above = 1
    abs_thresh = 5e3

    ## Parameter to reject high 60 Hz
    percent_60_hz = 0.7

    ## Parameter to reject electrodes with much higher std than most electrodes
    mult_std = 10

    bad = []
    high_ch = []
    nan_ch = []
    zero_ch = []
    flat_ch = []
    high_var_ch = []
    noisy_ch = []
    all_std = np.empty((len(which_chs),1))
    all_std[:] = np.nan
    details = {}

    for i in range(len(which_chs)):       
        ich = which_chs[i]
        eeg = values[:,ich]
        bl = np.nanmedian(eeg)
        all_std[i] = np.nanstd(eeg)
        
        ## Remove channels with nans in more than half
        if sum(np.isnan(eeg)) > 0.5*len(eeg):
            bad.append(ich)
            nan_ch.append(ich)
            continue
        
        ## Remove channels with zeros in more than half
        if sum(eeg == 0) > (0.5 * len(eeg)):
            bad.append(ich)
            zero_ch.append(ich)
            continue

        ## Remove channels with extended flat-lining
        if (sum(np.diff(eeg,1) == 0) > (0.02 * len(eeg))) and (sum(abs(eeg - bl) > abs_thresh) > (0.02 * len(eeg))):
            bad.append(ich)
            flat_ch.append(ich)
        
        ## Remove channels with too many above absolute thresh
        if sum(abs(eeg - bl) > abs_thresh) > 10:
            if not lf_stim:
                bad.append(ich)
            high_ch.append(ich)
            continue

        ## Remove channels if there are rare cases of super high variance above baseline (disconnection, moving, popping)
        pct = np.percentile(eeg,[100-tile,tile])
        thresh = [bl - mult*(bl-pct[0]), bl + mult*(pct[1]-bl)]
        sum_outside = sum(((eeg > thresh[1]) + (eeg < thresh[0])) > 0)
        if sum_outside >= num_above:
            if not lf_stim:
                bad.append(ich)
            high_var_ch.append(ich)
            continue
        
        ## Remove channels with a lot of 60 Hz noise, suggesting poor impedance
        # Calculate fft
        Y = np.fft.fft(eeg-np.nanmean(eeg))
        
        # Get power
        P = abs(Y)**2
        freqs = np.linspace(0,fs,len(P)+1)
        freqs = freqs[:-1]
        
        # Take first half
        P = P[:np.ceil(len(P)/2).astype(int)]
        freqs = freqs[:np.ceil(len(freqs)/2).astype(int)]
        
        P_60Hz = sum(P[(freqs > 58) * (freqs < 62)])/sum(P)
        if P_60Hz > percent_60_hz:
            bad.append(ich)
            noisy_ch.append(ich)
            continue

    ## Remove channels for whom the std is much larger than the baseline
    median_std = np.nanmedian(all_std)
    higher_std = which_chs[(all_std > (mult_std * median_std)).squeeze()]
    bad_std = higher_std
    # for ch in bad_std:
    #     if ch not in bad:
    #         if ~lf_stim:
    #             bad.append(ch)
    channel_mask = np.ones((values.shape[1],),dtype=bool)
    channel_mask[bad] = False
    details['noisy'] = noisy_ch
    details['nans'] = nan_ch
    details['zeros'] = zero_ch
    details['flat'] = flat_ch
    details['var'] = high_var_ch
    details['higher_std'] = bad_std
    details['high_voltage'] = high_ch
    
    return channel_mask,details


def num_wins(xLen, fs, winLen, winDisp):
    """
    Calculate the number of non-overlapping analysis windows for given data length.
    
    This utility function determines how many complete analysis windows can be 
    extracted from a data segment with specified window length and displacement.
    
    Args:
        xLen (int): Length of data in samples
        fs (float): Sampling frequency in Hz  
        winLen (float): Window length in seconds
        winDisp (float): Window displacement/step size in seconds
        
    Returns:
        int: Number of complete windows that fit in the data
        
    Notes:
        - Calculates based on non-overlapping windows (displacement = window length)
        - For overlapping windows, winDisp < winLen
        - For non-overlapping windows, winDisp = winLen
        - Rounds down to ensure complete windows only
        
    Formula:
        n_windows = floor((data_duration - winLen + winDisp) / winDisp)
        where data_duration = xLen / fs
        
    Example:
        >>> # 10 seconds of data at 500Hz, 2-second windows, 1-second steps
        >>> n_wins = num_wins(5000, 500, 2.0, 1.0)
        >>> # Returns 9 (windows at 0-2s, 1-3s, 2-4s, ..., 8-10s)
    """
    return int(((xLen/fs - winLen + winDisp) - ((xLen/fs - winLen + winDisp)%winDisp))/winDisp)


def bipolar_montage(data: np.ndarray, ch_types: pd.DataFrame) -> np.ndarray:
    """
    Convert iEEG data to bipolar montage by subtracting adjacent electrode contacts.
    
    Bipolar montage reduces common-mode artifacts and emphasizes local field potentials
    by computing differences between adjacent contacts on the same electrode lead.
    This is standard practice in clinical iEEG analysis.
    
    Args:
        data (numpy.ndarray): Raw iEEG data
            - Shape: (channels, time_points)
        ch_types (pandas.DataFrame): Channel information with columns:
            - 'name': Channel name (e.g., "LH01", "LH02") 
            - 'type': Channel type ("ecog", "seeg", "eeg", "ecg", "misc")
            - 'lead': Lead/electrode name (e.g., "LH")
            - 'contact': Contact number on lead (e.g., 1, 2, 3)
            
    Returns:
        tuple: (new_data, new_ch_types) where:
            - new_data: Bipolar montage data (fewer channels than input)
            - new_ch_types: DataFrame describing bipolar channel pairs
            
    Bipolar montage creation:
        - Only applies to 'ecog' and 'seeg' channels
        - Creates pairs: contact_n - contact_(n+1) for each lead
        - Results in channel names like "LH01-LH02", "LH02-LH03", etc.
        - Reduces number of channels by ~half
        
    Notes:
        - Skips channels that aren't ecog/seeg (preserves EEG, ECG channels)
        - Requires consecutive contact numbering within each lead
        - Last contact on each lead cannot form a pair (discarded)
        - Common mode rejection reduces artifacts and volume conduction
        
    Example:
        >>> # Convert to bipolar montage
        >>> bp_data, bp_channels = bipolar_montage(raw_data, channel_info)
        >>> print(f"Reduced from {raw_data.shape[0]} to {bp_data.shape[0]} channels")
    """
    n_ch = len(ch_types)
    new_ch_types = []
    for ind, row in ch_types.iterrows():
        # do only if type is ecog or seeg
        if row["type"] not in ["ecog", "seeg"]:
            continue

        ch1 = row["name"]

        ch2 = ch_types.loc[
            (ch_types["lead"] == row["lead"])
            & (ch_types["contact"] == row["contact"] + 1),
            "name",
        ]
        if len(ch2) > 0:
            ch2 = ch2.iloc[0]
            entry = {
                "name": ch1 + "-" + ch2,
                "type": row["type"],
                "idx1": ind,
                "idx2": ch_types.loc[ch_types["name"] == ch2].index[0],
            }
            new_ch_types.append(entry)

    new_ch_types = pd.DataFrame(new_ch_types)
    # apply montage to data
    new_data = np.empty((len(new_ch_types), data.shape[1]))
    for ind, row in new_ch_types.iterrows():
        new_data[ind, :] = data[row["idx1"], :] - data[row["idx2"], :]

    return new_data, new_ch_types

def ar_one(data):
    """
    Apply AR(1) autoregressive whitening to reduce temporal autocorrelation.
    
    This preprocessing step fits a first-order autoregressive model to each channel
    and retains the residuals, which have reduced autocorrelation structure.
    This can improve the independence assumption for many statistical analyses.
    
    Args:
        data (numpy.ndarray): Input signal data
            - Shape: (time_points, channels)
            
    Returns:
        numpy.ndarray: Whitened signal with reduced autocorrelation
            - Shape: (time_points-1, channels) [one sample shorter]
            
    Notes:
        - Fits AR(1) model: x[t] = a*x[t-1] + b + noise
        - Returns residuals: x[t] - (a*x[t-1] + b)
        - Reduces temporal correlations while preserving signal content
        - Useful before applying analyses that assume independence
        - Output is one sample shorter due to lag-1 operation
        
    Mathematical model:
        For each channel i:
        x[t] = w[0]*x[t-1] + w[1] + residual[t]
        where w is fitted via least squares
        
    Applications:
        - Seizure detection preprocessing
        - Connectivity analysis preparation  
        - Statistical analysis requiring independence
        
    Example:
        >>> # Remove temporal autocorrelation
        >>> whitened_data = ar_one(filtered_data)
        >>> # Apply detection algorithm to whitened data
    """
    # Retrieve data attributes
    n_samp, n_chan = data.shape
    # Apply AR(1)
    data_white = np.zeros((n_samp-1, n_chan))
    for i in range(n_chan):
        win_x = np.vstack((data[:-1, i], np.ones(n_samp-1)))
        w = np.linalg.lstsq(win_x.T, data[1:, i], rcond=None)[0]
        data_white[:, i] = data[1:, i] - (data[:-1, i]*w[0] + w[1])
    return data_white

def preprocess_for_detection(data,fs,montage='bipolar',target=256, wavenet=False, pre_mask = None):
    """
    Complete preprocessing pipeline for seizure detection algorithms.
    
    This function implements a standardized preprocessing workflow commonly used
    for automated seizure detection, including montage conversion, channel rejection,
    filtering, resampling, and autoregressive whitening.
    
    Args:
        data (pandas.DataFrame): Raw iEEG data with channel names as columns
        fs (float): Original sampling frequency in Hz
        montage (str, optional): Montage type ('bipolar' or 'car'). Defaults to 'bipolar'.
        target (int, optional): Target sampling frequency for resampling. Defaults to 256.
        wavenet (bool, optional): If True, uses WaveNet-specific parameters. Defaults to False.
        pre_mask (list, optional): Pre-specified channels to exclude. If None, auto-detects bad channels.
        
    Returns:
        tuple: (processed_data, new_fs, [bad_channels]) where:
            - processed_data: Preprocessed DataFrame ready for detection
            - new_fs: Final sampling frequency (= target)
            - bad_channels: List of rejected channels (only if pre_mask=None)
            
    Preprocessing pipeline:
        1. Channel type classification
        2. Montage conversion (bipolar or CAR)
        3. Bad channel detection and removal  
        4. Notch filtering (60/120Hz)
        5. Bandpass filtering
        6. Resampling to target frequency
        7. AR(1) whitening
        
    Montage options:
        - 'bipolar': Adjacent contact differences (reduces common artifacts)
        - 'car': Common average reference (subtracts mean across channels)
        
    Filter settings:
        - Standard: 3-100Hz bandpass, 256Hz target
        - WaveNet: 3-127Hz bandpass, 128Hz target
        
    Example:
        >>> # Standard seizure detection preprocessing
        >>> proc_data, fs_new = preprocess_for_detection(raw_data, 1000)
        >>> # Use proc_data with detection algorithm
    """
    # This function implements preprocessing steps for seizure detection
    chs = data.columns.to_list()
    ch_df = check_channel_types(chs)
    # Montage
    if montage == 'bipolar':
        data_bp_np,bp_ch_df = bipolar_montage(data.to_numpy().T,ch_df)
        bp_ch = bp_ch_df.name.to_numpy()
    elif montage == 'car':
        data_bp_np = (data.to_numpy().T - np.mean(data.to_numpy(),1))
        bp_ch = chs
    
    # Channel rejection
    if pre_mask is None:
        mask,_ = detect_bad_channels(data_bp_np.T*1e3,fs)
        data_bp_np = data_bp_np[mask,:]
        mask_list = [ch for ch in bp_ch[~mask]]
        bp_ch = bp_ch[mask]
    else:
        mask = np.atleast_1d([ch not in pre_mask for ch in bp_ch])
        data_bp_np = data_bp_np[mask,:]
        bp_ch = bp_ch[mask]
        
    # Filtering and autocorrelation
    if wavenet:
        target=128
        data_bp_notch = notch_filter(data_bp_np,fs)
        data_bp_filt = bandpass_filter(data_bp_notch,fs,lo=3,hi=127)
        signal_len = int(data_bp_filt.shape[1]/fs*target)
        data_bpd = sc.signal.resample(data_bp_filt,signal_len,axis=1).T
        fsd = int(target)
    else:
        # Bandpass filtering
        data_bp_notch = notch_filter(data_bp_np,fs)
        data_bp_filt = bandpass_filter(data_bp_notch,fs,lo=3,hi=100)
        # Down sampling
        signal_len = int(data_bp_filt.shape[1]/fs*target)
        data_bpd = sc.signal.resample(data_bp_filt,signal_len,axis=1).T
        fsd = int(target)
    data_white = ar_one(data_bpd)
    data_white_df = pd.DataFrame(data_white,columns = bp_ch)
    if pre_mask is None:
        return data_white_df,fsd,mask_list
    else:
        return data_white_df,fsd
    

def remove_scalp_electrodes(raw_labels):
    """
    Filter out scalp EEG electrodes from a list of channel labels.
    
    This function removes standard scalp EEG channels (10-20 system), ECG, EMG,
    and other non-intracranial channels from a channel list, leaving only 
    depth/grid electrodes for intracranial analysis.
    
    Args:
        raw_labels (list): List of all channel labels from the recording
        
    Returns:
        list: Filtered list containing only intracranial electrode channels
        
    Removed channel types:
        - Standard 10-20 EEG: CZ, FZ, PZ, C03/C04, F03/F04, etc.
        - Physiological monitoring: EKG01/02, EMG01/02, ROC, LOC
        - CHOP-specific scalp: C119-C128 range
        - Reference/ground: DC01, DC07
        
    Notes:
        - Case-insensitive matching (converts to uppercase)
        - Preserves intracranial depth and grid electrodes
        - Useful for focusing analysis on brain tissue recordings
        - CHOP hospital uses specific C1xx numbering for scalp channels
        
    Example:
        >>> all_channels = ['LH01', 'LH02', 'CZ', 'F03', 'RG01', 'EKG01']
        >>> brain_channels = remove_scalp_electrodes(all_channels)
        >>> # Returns ['LH01', 'LH02', 'RG01'] - only intracranial channels
    """
    scalp_list = ['CZ','FZ','PZ',
                  'A01','A02',
                  'C03','C04',
                  'F03','F04','F07','F08',
                  'Fp01','Fp02',
                  'O01','O02',
                  'P03','P04',
                  'T03','T04','T05','T06',
                  'EKG01','EKG02',
                  'ROC','LOC',
                  'EMG01','EMG02',
                  'DC01','DC07'
                  ]
    chop_scalp = ['C1'+str(x) for x in range(19,29)]
    scalp_list += chop_scalp
    return [l for l in raw_labels if l.upper() not in scalp_list]
################################################ Feature Extraction ################################################


######################## Univariate, Time Domain ########################
def _timeseries_to_wins(
    data: np.ndarray, fs: float, win_size=2, win_stride=1
) -> np.ndarray:
    """_summary_

    Args:
        data (np.ndarray)
        fs (float): _description_
        win_size (int, optional): _description_. Defaults to 2.
        win_stride (int, optional): _description_. Defaults to 1.

    Returns:
        np.ndarray: _description_
    """
    n_samples = data.shape[-1]

    idx = (
        np.arange(win_size * fs, dtype=int)[None, :]
        + np.arange(n_samples - win_size * fs + 1, dtype=int)[
            :: int(win_stride * fs), None
        ]
    )
    return data[:, idx]

def ft_extract(
    data: np.ndarray, fs: float, ft: str, win_size=2, win_stride=1, fn_kwargs={}
) -> np.ndarray:
    """_summary_

    Args:
        data (mne.io.edf.edf.RawEDF): _description_
        ft (str): _description_
        win_size (int, optional): _description_. Defaults to 2.
        win_stride (int, optional): _description_. Defaults to 1.

    Returns:
        np.ndarray: _description_
    """
    wins = _timeseries_to_wins(data, fs, win_size, win_stride)
    wins = np.transpose(wins, (1, 0, 2))

    # if ft is a list of features, then calculate both featurs and concatenate
    if isinstance(ft, list):
        assert len(ft) == len(fn_kwargs), "Incorrect number of feature arguments given"
        # ft_array = np.empty((n_ch, n_wins, len(ft)))
        ft_array = []
        for i, fn in enumerate(ft):
            # if f is not callable, then raise value error
            if not callable(fn):
                raise ValueError("Incorrect feature argument given")
            for win in wins:
                ft_array.append(fn(win, **(fn_kwargs[i])))
        ft_array = np.array(ft_array)
        # transpose to n_ch x n_wins x n_ft
        ft_array = np.transpose(ft_array, (1, 0, 2))
        return ft_array

    elif callable(ft):
        # ft_array = np.empty((n_ch, n_wins))

        ft_array = []

        for i, win in enumerate(wins):
            ft_array.append(ft(win, **fn_kwargs))
        
        ft_array = np.array(ft_array)
        
        # convert 2 dim to 3 dim
        if ft_array.ndim == 2:
            ft_array = ft_array[:, :, None]
        # transpose to n_ch x n_wins x n_ft
        ft_array = np.transpose(ft_array, (1, 0, 2))

    else:
        raise ValueError("Incorrect feature type given")

    return ft_array

def MovingWinClips(x,fs,winLen,winDisp):
    """
    Extract overlapping or non-overlapping time windows from a 1D signal.
    
    This function creates a matrix of time windows from a continuous signal,
    useful for feature extraction or sliding window analysis. Each row contains
    one time window of the signal.
    
    Args:
        x (array-like): Input 1D signal
        fs (float): Sampling frequency in Hz
        winLen (float): Window length in seconds
        winDisp (float): Window displacement/step size in seconds
        
    Returns:
        numpy.ndarray: Matrix of time windows
            - Shape: (n_windows, samples_per_window)
            - Each row is one time window
            
    Notes:
        - Window displacement determines overlap:
            * winDisp = winLen: Non-overlapping windows
            * winDisp < winLen: Overlapping windows
            * winDisp > winLen: Gaps between windows
        - Uses num_wins() to calculate number of windows
        - Windows are left-aligned (start at displacement intervals)
        - Incomplete final windows are excluded
        
    Example:
        >>> # Extract 2-second windows with 1-second overlap
        >>> signal = np.random.randn(5000)  # 10 seconds at 500Hz
        >>> windows = MovingWinClips(signal, fs=500, winLen=2.0, winDisp=1.0)
        >>> print(f"Shape: {windows.shape}")  # (9, 1000) - 9 windows of 1000 samples
    """
    # calculate number of windows and initialize receiver
    nWins = num_wins(len(x),fs,winLen,winDisp)
    samples = np.empty((nWins,winLen*fs))
    # create window indices - these windows are left aligned
    idxs = np.array([(winDisp*fs*i,(winLen+winDisp*i)*fs)\
                     for i in range(nWins)],dtype=int)
    # apply feature function to each channel
    for i in range(idxs.shape[0]):
        samples[i,:] = x[idxs[i,0]:idxs[i,1]]
    
    return samples



def dice_score(x,y):
    """
    Calculate Dice similarity coefficient between two sets or arrays.
    
    The Dice coefficient measures overlap between two sets, commonly used
    for comparing binary masks, electrode selections, or segmentation results.
    Returns a value between 0 (no overlap) and 1 (perfect overlap).
    
    Args:
        x (array-like): First set or array to compare
        y (array-like): Second set or array to compare
        
    Returns:
        float: Dice coefficient (0-1 scale)
            - 0: No overlap between sets
            - 1: Perfect overlap (identical sets)
            
    Formula:
        Dice = 2 * |intersection| / (|x| + |y|)
        
    Notes:
        - Handles scalar inputs by converting to arrays
        - Uses np.intersect1d for set intersection
        - More sensitive to size differences than Jaccard index
        - Commonly used in medical image analysis and electrode studies
        
    Example:
        >>> set1 = ['ch1', 'ch2', 'ch3', 'ch4']
        >>> set2 = ['ch2', 'ch3', 'ch4', 'ch5'] 
        >>> similarity = dice_score(set1, set2)
        >>> print(f"Dice coefficient: {similarity:.3f}")  # 0.75
    """
    num = 2*len(np.intersect1d(x,y))
    if len(x.shape) < 1:
        x = np.array([str(x)])
    if len(y.shape) < 1:
        y = np.array([str(y)])
    denom = len(x)+len(y)
    return num/denom


########################### Workspace Preparation ###########################
def set_seed(seed):
    """
    Set random seeds for reproducible results across multiple libraries.
    
    This function sets random seeds for NumPy, PyTorch, and Python's random
    module to ensure reproducible results in stochastic algorithms and analyses.
    
    Args:
        seed (int): Random seed value to use across all libraries
        
    Notes:
        - Sets seeds for: numpy, torch, and Python random
        - Critical for reproducible machine learning experiments
        - Should be called at the beginning of analysis scripts
        - Does not guarantee identical results across different hardware/software versions
        
    Example:
        >>> set_seed(42)  # Use consistent seed across experiments
        >>> # Now all random operations will be reproducible
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

########################### OS Utils ###########################

def in_parallel(func, data, verbose=False, n_jobs = -1):
    """
    Execute a function in parallel across multiple data items using joblib.
    
    This utility function parallelizes the execution of a function across
    a list of data items, utilizing multiple CPU cores for faster processing.
    Useful for embarrassingly parallel tasks like feature extraction.
    
    Args:
        func (callable): Function to apply to each data item
            - Should accept one argument (item from data list)
        data (list): List of data items to process
        verbose (bool, optional): If True, prints progress information. Defaults to False.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1 (all cores).
            - -1: Use all available CPU cores
            - 1: No parallelization (sequential)
            - >1: Use specified number of cores
            
    Returns:
        list: Results from applying func to each item in data
        
    Notes:
        - Uses joblib.Parallel with delayed execution
        - Automatically detects available CPU cores when n_jobs=-1
        - Best for CPU-bound tasks with independent data items
        - Overhead may not be worth it for very fast functions
        
    Example:
        >>> def process_patient(patient_id):
        >>>     # Some analysis function
        >>>     return analyze_data(patient_id)
        >>> 
        >>> patient_list = ['HUP001', 'HUP002', 'HUP003']
        >>> results = in_parallel(process_patient, patient_list, verbose=True)
    """
    if n_jobs < 1:
        threads = os.cpu_count()
    else:
        threads = n_jobs

    if verbose:
        print(f"Processing {len(data)} items in parallel using {threads} threads")

    return Parallel(n_jobs=threads)(delayed(func)(item) for item in data)

########################### Analysis Utils ###########################

def calculate_seizure_similarity(annots,first_annot = 'ueo_consensus', second_annot = 'ueo_consensus',paired=True):
    """
    Calculate similarity metrics between seizure annotations within and across patients.
    
    This function computes agreement metrics (kappa, F1, MCC) between seizure onset
    annotations, either comparing different annotators or comparing seizures within
    the same patient. Useful for studying annotation reliability and seizure consistency.
    
    Args:
        annots (pandas.DataFrame): DataFrame containing seizure annotations with columns:
            - patient: Patient identifier
            - stim: Binary indicator (1=stimulated, 0=spontaneous)
            - typical: Binary indicator for typical seizures
            - {first_annot}: First annotation array/mask
            - {second_annot}: Second annotation array/mask
        first_annot (str, optional): Column name for first annotation. Defaults to 'ueo_consensus'.
        second_annot (str, optional): Column name for second annotation. Defaults to 'ueo_consensus'.
        paired (bool, optional): If True, requires ≥2 spontaneous seizures per patient. Defaults to True.
        
    Returns:
        pandas.DataFrame: Similarity results with columns:
            - kappa: Cohen's kappa coefficient
            - F1: F1 score  
            - MCC: Matthews correlation coefficient
            - patient: Patient identifier
            - spont: Boolean (True=both seizures spontaneous)
            - typical: Boolean (True=at least one seizure typical)
            
    Agreement metrics:
        - Cohen's kappa: Inter-rater agreement correcting for chance
        - F1 score: Harmonic mean of precision and recall
        - MCC: Matthews correlation coefficient (balanced metric)
        
    Notes:
        - Compares all seizure pairs within each patient
        - Skips comparisons between two stimulated seizures
        - Skips patients without sufficient seizures (if paired=True)
        - Annotations should be binary arrays/masks of electrode involvement
        
    Example:
        >>> similarity_df = calculate_seizure_similarity(seizure_annots)
        >>> # Compare spontaneous vs stimulated seizure similarity
        >>> spont_sim = similarity_df[similarity_df.spont]['MCC'].median()
        >>> stim_sim = similarity_df[~similarity_df.spont]['MCC'].median()
    """
    annot_list = ["kappa","F1","MCC","patient","spont","typical"]
    annot_dict = {key:[] for key in annot_list}
    skip_pt = []
    for pt,group in annots.groupby("patient"):
        if (sum(group.stim == 0) < 2) and paired:
            skip_pt.append(pt)
            continue
        elif len(group) < 2:
            skip_pt.append(pt)
            continue
        # Iterate through each seizure
        for i in range(len(group)):
            group.reset_index(drop=True,inplace=True)
            ch_mask = group.loc[i,first_annot].reshape(-1)
            for j in range(i+1,len(group)):
                if (group.loc[i,'stim'] == 1)  and (group.loc[j,'stim'] == 1): # skip both stim
                    continue
                ch_mask2 = group.loc[j,second_annot].reshape(-1)
                annot_dict["kappa"].append(cohen_kappa_score(ch_mask,ch_mask2))
                annot_dict["F1"].append(f1_score(ch_mask,ch_mask2))
                annot_dict["MCC"].append(matthews_corrcoef(ch_mask,ch_mask2))
                annot_dict["spont"].append(not ((group.loc[i,'stim'] == 1)  or (group.loc[j,'stim'] == 1)))
                # want to append a boolean that will tell me if one sz is stim and one sz is typical
                annot_dict["typical"].append(((group.loc[i,'typical'] == 1)  
                                                        or (group.loc[j,'typical'] == 1)))
                annot_dict["patient"].append(pt)
    annot_df = pd.DataFrame(annot_dict)
    print(f"Skipped {skip_pt} due to insufficient spontaneous seizures")
    return annot_df

def calculate_spread_similarity(annots, first_annot='sz_chs', second_annot='sz_chs',
                                 sources='all_channels', spread_thresh=30, paired=True):
    """
    Calculate similarity metrics for seizure spread patterns within a time threshold.
    
    This function compares seizure propagation patterns by analyzing which channels/regions
    are recruited within a specified time window. It computes both binary overlap (MCC) and
    temporal ranking similarity (Spearman correlation) between seizure pairs.
    
    Args:
        annots (pandas.DataFrame): DataFrame containing seizure annotations with columns:
            - patient: Patient identifier
            - stim: Binary indicator (1=stimulated, 0=spontaneous)
            - typical: Binary indicator for typical seizures
            - sz_times: List of recruitment times for each channel/region
            - {first_annot}: List of recruited channels/regions (e.g., 'sz_chs')
            - {second_annot}: List of recruited channels/regions for comparison
            - {sources}: List of all possible channels/regions to consider
        first_annot (str, optional): Column name for first seizure's recruited units. 
            Defaults to 'sz_chs'.
        second_annot (str, optional): Column name for second seizure's recruited units. 
            Defaults to 'sz_chs'.
        sources (str, optional): Column name for all possible channels/regions. 
            Defaults to 'all_channels'.
        spread_thresh (float, optional): Time threshold in seconds for early recruitment. 
            Defaults to 30.
        paired (bool, optional): If True, requires ≥2 spontaneous seizures per patient. 
            Defaults to True.
            
    Returns:
        pandas.DataFrame: Spread similarity results with columns:
            - MCC: Matthews correlation coefficient for binary recruitment overlap
            - Rank: Spearman correlation for temporal recruitment ranking
            - patient: Patient identifier
            - spont: Boolean (True=both seizures spontaneous)
            - typical: Boolean (True=at least one seizure typical)
            
    Algorithm:
        1. For each seizure pair within a patient:
           a. Identify channels recruited before spread_thresh
           b. Create binary mask of early-recruited channels
           c. Calculate recruitment latencies for all channels
           d. Compare binary patterns (MCC) and ranking patterns (Spearman)
        
    Similarity metrics:
        - MCC: Binary overlap of channels recruited within time threshold
        - Rank: Spearman correlation of recruitment time rankings across all channels
        
    Notes:
        - Skips seizure pairs where both are stimulated
        - Handles missing recruitment data (empty sz_times lists)
        - Uses minimum recruitment time when channels appear multiple times
        - Assigns maximum latency + 1 to non-recruited channels for ranking
        - Filters patients with insufficient seizure counts (if paired=True)
        
    Example:
        >>> # Compare seizure spread patterns within 30 seconds
        >>> spread_sim = calculate_spread_similarity(seizure_data, spread_thresh=30)
        >>> print(f"Mean early spread similarity: {spread_sim['MCC'].mean():.3f}")
        >>> print(f"Mean temporal ranking similarity: {spread_sim['Rank'].mean():.3f}")
    """
    annot_list = ["MCC", "Rank", "patient", "spont", "typical"]
    annot_dict = {key: [] for key in annot_list}
    skip_pt = []

    for pt, group in annots.groupby("patient"):
        if (sum(group.stim == 0) < 2 and paired) or (len(group) < 2):
            skip_pt.append(pt)
            continue

        group = group.reset_index(drop=True)

        for i in range(len(group)):
            sz_i = group.loc[i]
            if len(sz_i['sz_times']) == 0:
                continue

            # Filter seizure annotations to exclude EmptyLabel, white matter, and WM regions
            filtered_indices = [idx for idx, unit in enumerate(sz_i[first_annot])
                              if ((unit != 'EmptyLabel') and ('white' not in unit.lower()) and ('wm' not in unit.lower()))]
            
            if len(filtered_indices) == 0:
                continue
                
            filtered_first_annot = [sz_i[first_annot][idx] for idx in filtered_indices]
            filtered_sz_times = [sz_i['sz_times'][idx] for idx in filtered_indices]

            # Create mask for channels/regions recruited before spread_thresh
            ch_time_mask = np.array(filtered_sz_times) < spread_thresh
            ch_mask = np.isin(sz_i[sources], np.array(filtered_first_annot)[ch_time_mask])

            # Use groupby one-liner to get minimum time per unit
            onset_dict = pd.DataFrame({
                "unit": filtered_first_annot,
                "time": filtered_sz_times
            }).groupby("unit")["time"].min().to_dict()

            max_latency = max(onset_dict.values()) + 1
            all_latencies = np.array([onset_dict.get(ch, max_latency) for ch in sz_i[sources]])

            for j in range(i + 1, len(group)):
                sz_j = group.loc[j]
                if sz_i['stim'] == 1 and sz_j['stim'] == 1:
                    continue
                if len(sz_j['sz_times']) == 0:
                    continue

                # Filter seizure annotations for second seizure
                filtered_indices2 = [idx for idx, unit in enumerate(sz_j[second_annot])
                                   if ((unit != 'EmptyLabel') and ('white' not in unit.lower()) and ('wm' not in unit.lower()))]
                
                if len(filtered_indices2) == 0:
                    continue
                    
                filtered_second_annot = [sz_j[second_annot][idx] for idx in filtered_indices2]
                filtered_sz_times2 = [sz_j['sz_times'][idx] for idx in filtered_indices2]

                ch_time_mask2 = np.array(filtered_sz_times2) < spread_thresh
                ch_mask2 = np.isin(sz_j[sources], np.array(filtered_second_annot)[ch_time_mask2])

                onset_dict2 = pd.DataFrame({
                    "unit": filtered_second_annot,
                    "time": filtered_sz_times2
                }).groupby("unit")["time"].min().to_dict()

                max_latency2 = max(onset_dict2.values()) + 1
                all_latencies2 = np.array([onset_dict2.get(ch, max_latency2) for ch in sz_j[sources]])

                annot_dict["MCC"].append(matthews_corrcoef(ch_mask, ch_mask2))
                annot_dict["Rank"].append(sc.stats.spearmanr(all_latencies, all_latencies2).statistic)
                annot_dict["spont"].append(not (sz_i['stim'] == 1 or sz_j['stim'] == 1))
                annot_dict["typical"].append(sz_i['typical'] == 1 or sz_j['typical'] == 1)
                annot_dict["patient"].append(pt)

    annot_df = pd.DataFrame(annot_dict)
    print(f"Skipped {skip_pt} due to insufficient spontaneous seizures")
    return annot_df
    
def plot_seizure_similarity(dat,agreement='MCC',palette=['red','blue','purple'],annot_type='',
                            sz_level=True, binary = False, typical = None, combiner = 75,
                            annot_stats=True,figpath=''):
    """
    Create publication-quality plots comparing seizure similarity between conditions.
    
    This function generates statistical comparison plots for seizure similarity metrics,
    typically comparing spontaneous-spontaneous vs stimulated-spontaneous seizure pairs.
    Includes statistical testing and effect size reporting.
    
    Args:
        dat (pandas.DataFrame): Seizure similarity data from calculate_seizure_similarity()
        agreement (str, optional): Similarity metric to plot ('MCC', 'kappa', 'F1'). 
            Defaults to 'MCC'.
        palette (list, optional): Colors for plot elements. Defaults to ['red','blue','purple'].
        annot_type (str, optional): Annotation type label for titles. Defaults to ''.
        sz_level (bool, optional): If True, uses seizure-level analysis (mixed effects).
            If False, aggregates to patient level. Defaults to True.
        binary (bool, optional): If True, treats agreement as binary outcome. Defaults to False.
        typical (bool, optional): Filter for typical seizures only. Defaults to None.
        combiner (int, optional): Percentile for patient-level aggregation. Defaults to 75.
        annot_stats (bool, optional): If True, adds statistical annotations to plot. 
            Defaults to True.
            
    Returns:
        tuple: (fig, ax) - matplotlib figure and axis objects
        
    Statistical methods:
        - Seizure-level: Mixed-effects linear model (accounts for patient clustering)
        - Patient-level: Wilcoxon signed-rank test (paired comparison)
        - Binary outcomes: Chi-square test of independence
        
    Plot elements:
        - Point plot showing medians with error bars
        - Swarm plot showing individual data points
        - Statistical annotations (p-values, effect sizes)
        - Cohen's d effect size calculation and reporting
        
    Notes:
        - Automatically handles patient-level aggregation using specified percentile
        - Supports both continuous similarity metrics and binary agreement outcomes
        - Prints summary statistics (median, IQR) for each condition
        - Uses statannotations package for clean statistical annotations
        
    Example:
        >>> fig, ax = plot_seizure_similarity(similarity_data, agreement='MCC')
        >>> plt.savefig('seizure_similarity.pdf', bbox_inches='tight')
        >>> plt.show()
    """
    if typical is not None:
        all_groups = []
        for _, group in dat.groupby(['patient']):
            stims = group.loc[(group.spont==False) & (group.typical == typical),:]

            if len(stims) == 0:
                continue
            
            all_groups.append(pd.concat([stims,group.loc[group.spont,:]],axis=0))

        dat = pd.concat(all_groups, axis=0)


    def percentile(x,combiner=combiner):
        return np.percentile(x,combiner,method='nearest')
        # return np.mean(x)


    # Define the aggregation functions
    numeric_cols = dat.select_dtypes(include='number').columns
    non_numeric_cols = dat.select_dtypes(exclude='number').columns.difference(['patient', 'spont'])
    
    if (agreement == 'Rank'):
        dat['Rank'] = dat['Rank'].fillna(0)
        # dat = dat[~dat.isna().any(axis=1)]

    pt_data = dat.groupby(['patient', 'spont']).agg(
        {col: percentile for col in numeric_cols} |
        {col: 'max' for col in non_numeric_cols}
    ).reset_index()

    fig,ax = plt.subplots(figsize=(4,5))
    if sz_level:
        model = smf.mixedlm(f"{agreement} ~ C(spont)", dat, groups="patient")
        result = model.fit()
        print(result.summary())
        print(result.pvalues)
        plot_data = dat

    else:
        plot_data = pt_data
        _,p = sc.stats.wilcoxon(pt_data[~pt_data.spont].sort_values('patient')[agreement],pt_data[pt_data.spont].sort_values('patient')[agreement])
        if binary:
            cont = pd.crosstab(pt_data.spont,pt_data[agreement])
            res = sc.stats.chi2_contingency(cont)
            p = res.pvalue
            print(res)
            fig1,ax1 = plt.subplots()
            sns.heatmap(cont,annot=True,robust=True,
                        xticklabels=True,yticklabels=True,
                        cmap=sns.light_palette("seagreen", as_cmap=True),
                        ax=ax1,
                        cbar=False)
            ax1.set_yticks([0.5,1.5],["Stim-Spont","Spont-Spont"])
            ax1.set_xlabel("Region Agreement?")
            ax1.set_ylabel("")
            fig1.savefig(ospj(figpath,"one_onset_region_boxes.pdf"))
        d = cohens_d(pt_data[~pt_data.spont][agreement],pt_data[pt_data.spont][agreement])
        # print(f"Paired t-test - p: {p}, d: {d}")
        # ax.set_title(f"Patient-Level Seizure{annot_type} Similarity")

    ax = sns.pointplot(data=plot_data,x="spont",y=agreement,
                errorbar=None,
                markers="_",
                linestyles="none",
                palette=palette[:2],
                estimator=np.median,
                linewidth=4,
                markersize=45,
                ax=ax)
    # plt.setp(ax.lines, linewidth=20)
    sns.swarmplot(data=plot_data,x="spont",y=agreement,
                alpha=.7,
                palette = palette[:2],
                ax=ax
                # hue='patient'
                )
    if annot_stats:
        annotator = Annotator(ax,[(True,False)],data=plot_data,x='spont',y=agreement)
        annotator.configure(test='Wilcoxon',
        loc='outside',
        text_format='star',
        fontsize=14,
        pvalue_thresholds=[[1e-4, "****"], [1e-3, "***"],
                        [1e-2, "**"], [0.05, "*"],[1, "ns"]])
        annotator.apply_and_annotate()

    plt.ylim([-0.25,1.15])
    sns.despine()
    ax.set_xticks([0,1],["Stim Induced-\nSpontaneous","Spontaneous-\nSpontaneous"])
    ax.set_xlabel('')
    # ax.set_ylabel(f"Electrographic Similarity ({agreement})")
    ax.set_ylabel('Onset Similarity ($\phi$)')
    x_spont = plot_data[plot_data.spont][agreement]
    x_stim = plot_data[~plot_data.spont][agreement]

    print(f"Spontaneous: N = {len(x_spont)} {x_spont.median():.2f} [{np.percentile(x_spont,25,method='nearest'):.2f}, {np.percentile(x_spont,75,method='nearest'):.2f}]")
    print(f"Stim: N = {len(x_stim)} {x_stim.median():.2f} [{np.percentile(x_stim,25,method='nearest'):.2f}, {np.percentile(x_stim,75,method='nearest'):.2f}]")


    return fig,ax