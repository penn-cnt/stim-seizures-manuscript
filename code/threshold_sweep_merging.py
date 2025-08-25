"""
Script for merging threshold-swept model predictions with human annotations.

This script processes machine learning model predictions for seizure channel detection
and merges them with human annotations to compute evaluation metrics (MCC scores).
It runs threshold tuning analysis in parallel across different model parameters.
"""

# Scientific computing imports
import pandas as pd

# Only import the sklearn metric that's actually used
from sklearn.metrics import matthews_corrcoef

# OS imports
from os.path import join as ospj
from utils import in_parallel

# Custom imports
from config import Config

def wideform_preds(element, all_labels):
    """
    Convert channel predictions to wide-form boolean array.
    
    Args:
        element: List of predicted channels
        all_labels: Complete list of all possible channel labels
    
    Returns:
        List of booleans indicating which channels were predicted
    """
    return [label in element for label in all_labels]

def apply_mcc(row):
    """
    Apply Matthews Correlation Coefficient calculation to a data row.
    
    Computes MCC scores for different prediction types (UEO/SEC) and annotation
    methods (consensus/any) by comparing model predictions to ground truth.
    
    Args:
        row: DataFrame row containing predictions and annotations
    
    Returns:
        Modified row with added MCC scores
    """
    all_chs = row.all_chs
    
    # Process each prediction type (ueo/sec with strict/loose thresholds)
    for col in ['ueo_chs_strict', 'ueo_chs_loose', 'sec_chs_strict', 'sec_chs_loose']:
        # Convert model predictions to wide-form boolean array
        row[col + '_bool'] = wideform_preds(row[col], all_chs)
        
        # Calculate MCC for both consensus and any annotation methods
        for annot in ['consensus', 'any']:
            ch_preds = row[f'{col[:3]}_{annot}']  # Get corresponding annotation
            row[f'{col}_{annot}_MCC'] = matthews_corrcoef(row[col + '_bool'], ch_preds)
    
    return row

# Load configuration and data paths
prodatapath = Config.deal('prodatapath')

# Load human annotations with consensus annotation already created
annotations_df = pd.read_pickle(ospj(prodatapath, "threshold_tuning_consensus_v2.pkl"))
annotations_df.columns = ['Patient' if c == 'patient' else c for c in annotations_df.columns]
annotations_df.sort_values('approximate_onset', inplace=True)

# Define model parameter combinations for processing
params = []
for epochs in [10]:
    for demin in [False]:
        for movtype in ['mean']:
            for movwin in [20]:
                for movdata in ['prob']:
                    params.append((epochs, demin, movtype, movwin, movdata))

def threshold_merge(params):
    """
    Merge model predictions with human annotations for a specific parameter set.
    
    Loads predicted channels, merges with annotations based on seizure onset timing,
    and computes MCC scores for evaluation.
    
    Args:
        params: Tuple of (epochs, demin, movtype, movwin, movdata) parameters
    """
    print(f'Starting: {params}')
    epochs, demin, movtype, movwin, movdata = params
    
    # Load model predictions for this parameter combination
    predicted_channels = pd.read_pickle(
        ospj(prodatapath, f"pretrain_predicted_channels_epoch-{epochs}_min-{str(demin)}_mov-{movtype}-{str(movwin)}-{movdata}_newptsv2.pkl")
    )
    predicted_channels = predicted_channels[predicted_channels.to_annotate == 1]
    predicted_channels.sort_values('approximate_onset', inplace=True)

    # Merge predictions with human annotations using approximate seizure onset time
    # Uses time-based merge with 240-second tolerance to match seizures
    pred_channels_wannots = pd.merge_asof(
        predicted_channels,
        annotations_df[['approximate_onset', 'Patient', 'all_chs', 'ueo_consensus', 'ueo_any', 'sec_consensus', 'sec_any']],
        on='approximate_onset', 
        by='Patient',
        tolerance=240,  # 4-minute tolerance for matching seizures
        direction='nearest'
    )
        
    # Remove rows without human annotations
    pred_channels_wannots.dropna(axis=0, subset='ueo_consensus', inplace=True)
    pred_channels_wannots.sort_values(['Patient', 'iEEG_ID', 'approximate_onset'], inplace=True)
    
    # Calculate MCC scores for all prediction/annotation combinations
    pred_channels_wmcc = pred_channels_wannots.apply(apply_mcc, axis=1)

    # Save results with MCC scores
    pred_channels_wmcc.to_pickle(
        ospj(prodatapath, f"pretrain_predicted_channels_wmcc_epoch-{epochs}_min-{str(demin)}_mov-{movtype}-{str(movwin)}-{movdata}_v3.pkl")
    )
    print('Iter Done')

# Run threshold merging in parallel across all parameter combinations
_ = in_parallel(threshold_merge, params)