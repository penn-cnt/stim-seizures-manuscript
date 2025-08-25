# Standard imports
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from os.path import join as ospj

def stim_detect(data, threshold, fs):
    """
    Detect stimulation artifacts in neural data by finding peaks in the derivative.
    
    Args:
        data: DataFrame with neural channels as columns
        threshold: Array of threshold values for each channel
        fs: Sampling frequency in Hz
        
    Returns:
        pk_idxs: Array of indices where stimulation artifacts occur
        stim_chs: Boolean array indicating which channels have stimulation
    """
    # Initialize array to store peak locations for each channel
    all_pks = np.zeros_like(data)
    
    # Find peaks in each channel's derivative (stimulation causes sharp changes)
    for i, (_, ch) in enumerate(data.items()):
        pks, _ = sc.signal.find_peaks(np.abs(np.diff(ch.to_numpy())),
                                    height=threshold[i],
                                    distance=fs/4*3,  # Minimum 750ms between peaks
                                    )
        all_pks[pks, i] = 1
    
    # Find time points where multiple channels show peaks (true stimulation events)
    pk_idxs, _ = sc.signal.find_peaks(all_pks.sum(axis=1),
                            distance=fs/4*3,  # Minimum 750ms between stimulation events
                            )
    
    # Identify which channels had any stimulation artifacts
    stim_chs = all_pks.any(0)
    return pk_idxs, stim_chs

def barndoor(sz, pk_idxs, fs, pre=50e-3, post=100e-3, plot=False, figpath=''):
    """
    Remove stimulation artifacts using the "barn door" interpolation method.
    Replaces artifact periods with blended data from before and after the artifact.
    
    Args:
        sz: DataFrame containing seizure/neural data
        pk_idxs: Array of stimulation artifact indices from stim_detect()
        fs: Sampling frequency in Hz
        pre: Pre-artifact window duration in seconds (default 50ms)
        post: Post-artifact window duration in seconds (default 100ms)
        plot: Whether to generate diagnostic plots
        figpath: Path to save plots if plot=True
        
    Returns:
        data: DataFrame with stimulation artifacts removed
    """
    data = sz.copy()
    
    # Convert time windows to sample indices
    pre_idx = np.floor(pre * fs).astype(int)
    post_idx = np.floor(post * fs).astype(int)
    win_idx = pre_idx + post_idx
    
    # Process each stimulation artifact
    for idx in pk_idxs:
        # Define artifact window boundaries
        sidx = int(idx - pre_idx)  # Start of artifact window
        eidx = int(idx + post_idx)  # End of artifact window
        
        # Skip if window extends beyond data boundaries
        if ((eidx + win_idx) > data.shape[0]) | ((sidx - win_idx) < 0):
            continue
            
        win_idx = eidx - sidx
        
        # Create linear taper for blending pre- and post-artifact data
        taper = np.linspace(0, 1, win_idx)
        
        # Extract data segments before and after the artifact
        pre_data = data.iloc[sidx - win_idx:sidx, :].to_numpy()
        post_data = data.iloc[eidx:eidx + win_idx, :].to_numpy()
        
        # Replace artifact with weighted blend of flipped pre/post data
        data.iloc[sidx:eidx, :] = (np.flip(pre_data, 0) * np.flip(taper).reshape(-1, 1) + 
                                  np.flip(post_data, 0) * taper.reshape(-1, 1))
        
        # Generate diagnostic plots if requested
        if plot:
            _, axs = plt.subplots(4, 1)
            axs[0].plot(sz.iloc[sidx - win_idx:eidx + win_idx, 8])  # Original data
            axs[1].plot(np.flip(pre_data[:, 8], 0))  # Pre-artifact data (flipped)
            axs[1].plot(np.flip(post_data[:, 8], 0))  # Post-artifact data (flipped)
            axs[2].plot(np.flip(taper))  # Tapering weights
            axs[2].plot(taper)
            axs[3].plot(data.iloc[sidx - win_idx:eidx + win_idx, 8])  # Reconstructed data
            axs[3].axvline(sidx)  # Mark artifact boundaries
            axs[3].axvline(eidx)
            plt.show()
            
            # Comparison plot
            fig = plt.figure()
            plt.plot(sz.iloc[sidx - win_idx:eidx + win_idx, 8])  # Original
            plt.plot(data.iloc[sidx - win_idx:eidx + win_idx, 8])  # Reconstructed
            plt.axvline(sidx, color='k')  # Artifact boundaries
            plt.axvline(eidx, color='k')
            plt.show()
            fig.savefig(ospj(figpath, 'reconstruction_error.pdf'))
            plot = False  # Only plot first artifact

    return data
        