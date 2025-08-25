### SEIZURE DETECTION PIPELINE - PRE-TRAINED MODELS
"""
Script for seizure detection using pre-trained models (LSTM, AbsSlope, WaveNet).
Trains and applies multiple seizure detection algorithms to iEEG data in BIDS format.
Each model is trained on interictal data and tested on seizure recordings to generate
probability matrices for downstream analysis and visualization.

The pipeline supports three detection methods:
1. LSTM - Long Short-Term Memory autoregressive model
2. AbsSlope - Absolute slope feature-based detector  
3. WaveNet - Pre-trained convolutional neural network

Output: Probability matrices saved as pickled DataFrames with seizure onset visualizations.
"""

# Scientific computing imports
import os as _os
_os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO/WARN messages
_os.environ['TF_TRT_DISABLED'] = '1'       # Silence TF-TRT warnings if TensorRT not installed

import numpy as np
import pandas as pd
from scipy.linalg import hankel
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler

# Plotting imports
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Deep learning imports  
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.models import load_model
from tensorflow.config.experimental import set_memory_growth, list_physical_devices
import tensorflow as tf
from absl import logging as absl_logging

# Suppress TensorFlow logging
absl_logging.set_verbosity(absl_logging.ERROR)
tf.get_logger().setLevel('ERROR')

# Configure GPU memory growth to prevent allocation issues
try:
    for _gpu in list_physical_devices('GPU'):
        set_memory_growth(_gpu, True)
except Exception:
    pass

# File system and utility imports
import os
from os.path import join as ospj
from os.path import exists as ospe
from utils import *
from stim_seizure_preprocessing_utils import *
import sys
sys.path.append('/users/wojemann/iEEG_processing')

# Set default colormap for visualizations
plt.rcParams['image.cmap'] = 'magma'

# Global configuration
OVERWRITE = False  # Whether to overwrite existing probability matrix files

def prepare_segment(data, fs = 256,train_win = 12, pred_win = 1, w_size = 1, w_stride=0.5,ret_time=False):
    """
    Convert seizure/interictal data into input-target pairs for autoregressive LSTM training.
    
    Creates sliding windows with Hankel matrix structure for time series prediction.
    Each window contains a training sequence (12 time points) and target (1 time point).
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Multi-channel iEEG data (samples x channels)
    fs : int, default=256
        Sampling frequency in Hz
    train_win : int, default=12  
        Number of time points for training sequence
    pred_win : int, default=1
        Number of time points for prediction target
    w_size : float, default=1
        Window size in seconds
    w_stride : float, default=0.5
        Window stride in seconds
    ret_time : bool, default=False
        Whether to return time stamps for each window
        
    Returns:
    --------
    input_data : torch.Tensor
        Training sequences (n_samples, train_win, n_channels)
    target_data : torch.Tensor  
        Target sequences (n_samples, n_channels)
    win_times : numpy.ndarray, optional
        Window start times if ret_time=True
    """
    data_ch = data.columns.to_list()
    data_np = data.to_numpy()
    train_win = 12
    pred_win = 1
    j = int(fs-(train_win+pred_win)+1)
    nwins = num_wins(len(data_np[:,0]),fs,w_size,w_stride)
    data_mat = torch.zeros((nwins,j,(train_win+pred_win),len(data_ch)))
    for k in range(len(data_ch)):
        samples = MovingWinClips(data_np[:,k],fs,1,0.5)
        for i in range(samples.shape[0]):
            clip = samples[i,:]
            mat = torch.tensor(hankel(clip[:j],clip[-(train_win+pred_win):]))
            data_mat[i,:,:,k] = mat
    time_mat = MovingWinClips(np.arange(len(data))/fs,fs,1,0.5)
    win_times = time_mat[:,0]
    data_flat = data_mat.reshape((-1,train_win + pred_win,len(data_ch)))
    input_data = data_flat[:,:-1,:].float()
    target_data = data_flat[:,-1,:].float()
    if ret_time:
        return input_data, target_data, win_times
    else:
        return input_data, target_data

def prepare_wavenet_segment(data, fs = 128, w_size = 1, w_stride=0.5,ret_time=False):
    """
    Prepare data segments for WaveNet seizure detection model.
    
    Formats multi-channel iEEG data into non-overlapping windows suitable for 
    convolutional neural network processing. Reshapes data for channel-wise analysis.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Multi-channel iEEG data (samples x channels)
    fs : int, default=128
        Sampling frequency in Hz
    w_size : float, default=1
        Window size in seconds  
    w_stride : float, default=0.5
        Window stride in seconds
    ret_time : bool, default=False
        Whether to return time stamps for each window
        
    Returns:
    --------
    data_flat : numpy.ndarray
        Flattened data array (n_windows*n_channels, window_length)
    win_times : numpy.ndarray, optional
        Window start times if ret_time=True
    """
    data_ch = data.columns.to_list()
    n_ch = len(data_ch)
    data_np = data.to_numpy()
    win_len_idx = w_size*fs
    nwins = num_wins(len(data_np[:,0]),fs,w_size,w_stride)
    data_mat = np.zeros((nwins,win_len_idx,len(data_ch)))
    for k in range(n_ch):
        samples = MovingWinClips(data_np[:,k],fs,w_size,w_stride)
        data_mat[:,:,k] = samples
    time_mat = MovingWinClips(np.arange(len(data))/fs,fs,w_size,w_stride)
    win_times = time_mat[:,0]
    data_flat = data_mat.transpose(0,2,1).reshape(-1,win_len_idx)
    if ret_time:
        return data_flat, win_times
    else:
        return data_flat
    
def predict_sz(model, input_data, target_data,batch_size=1,ccheck=False):
    """
    Generate seizure detection predictions using trained autoregressive model.
    
    Computes mean squared error (MSE) between model predictions and targets
    for each data window. Higher MSE indicates potential seizure activity.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained PyTorch model (e.g., LSTM)
    input_data : torch.Tensor
        Input sequences (n_samples, sequence_length, n_channels)
    target_data : torch.Tensor
        Target sequences (n_samples, n_channels)
    batch_size : int, default=1
        Batch size for inference
    ccheck : bool, default=False
        Whether to use CUDA GPU acceleration
        
    Returns:
    --------
    numpy.ndarray
        MSE loss distribution (n_samples, n_channels)
    """
    dataset = TensorDataset(input_data,target_data)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
    if ccheck:
        model.cuda()
    with torch.no_grad():
        model.eval()
        mse_distribution = []
        for inputs, targets in dataloader:
            if ccheck:
                inputs = inputs.cuda()
                targets = targets.cuda()
            outputs = model(inputs)
            mse = (outputs-targets)**2
            mse_distribution.append(mse)
            del inputs, targets, outputs, mse
    return torch.cat(mse_distribution).cpu().numpy()

def repair_data(outputs,data,fs=256,train_win=12,pred_win=1,w_size=1,w_stride=.5):
    """
    Reshape prediction outputs back to original data dimensions.
    
    Converts flattened prediction outputs from predict_sz back into 
    multi-channel time series format for visualization and analysis.
    
    Parameters:
    -----------
    outputs : numpy.ndarray
        Flattened prediction outputs from predict_sz
    data : pandas.DataFrame
        Original data used to determine reshape dimensions
    fs : int, default=256
        Sampling frequency in Hz
    train_win : int, default=12
        Training window length used in prepare_segment
    pred_win : int, default=1  
        Prediction window length used in prepare_segment
    w_size : float, default=1
        Window size in seconds
    w_stride : float, default=0.5
        Window stride in seconds
        
    Returns:
    --------
    numpy.ndarray
        Reshaped outputs (n_windows, n_samples_per_window, n_channels)
    """
    nwins = num_wins(len(data.to_numpy()[:,0]),fs,w_size,w_stride)
    nchannels = data.shape[1]
    repaired = outputs.reshape((nwins,fs-(train_win + pred_win)+1,nchannels))
    return repaired

def train_model(model,dataloader,criterion,optimizer,num_epochs=10,ccheck=False):
    """
    Train PyTorch model using provided data and optimization parameters.
    
    Performs standard supervised learning with backpropagation. Includes
    progress tracking and memory management for efficient training.
    
    Parameters:
    -----------
    model : torch.nn.Module
        PyTorch model to train
    dataloader : torch.utils.data.DataLoader
        Data loader with training batches
    criterion : torch.nn.Module
        Loss function (e.g., MSELoss)
    optimizer : torch.optim.Optimizer
        Optimization algorithm (e.g., Adam)
    num_epochs : int, default=10
        Number of training epochs
    ccheck : bool, default=False
        Whether to use CUDA GPU acceleration
        
    Returns:
    --------
    None
        Model is modified in-place
    """
    # Training loop
    tbar = tqdm(range(num_epochs),leave=False)
    for e in tbar:
        for inputs, targets in dataloader:
            if ccheck:
                inputs = inputs.cuda()
                targets = targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            del inputs, targets, outputs
        if e % 10 == 9:
            tbar.set_description(f"{loss.item():.4f}")
            del loss

class LSTMModel(nn.Module):
    """
    Long Short-Term Memory model for seizure detection via autoregression.
    
    Implements a single-layer LSTM followed by a fully connected layer for 
    time series prediction. Trained to predict the next time point given
    a sequence of previous time points. Seizures are detected as periods
    of high prediction error (increased MSE loss).
    
    Architecture:
    - LSTM layer with configurable hidden size
    - Linear output layer mapping to all channels
    - Built-in RobustScaler for data normalization
    """
    def __init__(self, input_size, hidden_size):
        """Initialize LSTM model with specified dimensions."""
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def fit_scaler(self, x):
        """Fit RobustScaler to training data for normalization."""
        self.scaler = RobustScaler().fit(x)

    def scaler_transform(self, x):
        """Apply fitted scaler transformation to input data."""
        return self.scaler.transform(x)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1,:])
        return out
    def __str__(self):
         return "LSTM"

class AbsSlope():
    """
    Absolute slope-based seizure detection algorithm.
    
    Computes seizure probability based on the absolute value of signal derivatives.
    Uses sliding windows to calculate the mean absolute slope for each channel,
    normalized by baseline standard deviations. Higher slopes typically indicate
    seizure activity with rapid voltage changes.
    
    Features:
    - RobustScaler normalization for artifact resistance
    - Configurable window size and stride
    - Baseline normalization using interictal data statistics
    """
    def __init__(self, win_size = 1, stride = 0.5, fs = 256):
        self.function = lambda x: np.mean(np.abs(np.diff(x,axis=-1)),axis=-1)
        self.win_size = win_size
        self.stride = stride
        self.fs = fs
    
    def __str__(self) -> str:
        return "AbsSlp"
        
    def fit(self, x):
        # x should be samples x channels df
        self.scaler = RobustScaler().fit(x)
        nx = self.scaler.transform(x)
        self.inter = pd.DataFrame(nx,columns = x.columns)
        self.nstds = np.std(nx,axis=0)

    def get_times(self, x):
        # x should be samples x channels df
        time_mat = MovingWinClips(np.arange(len(x))/self.fs,self.fs,self.win_size,self.stride)
        return np.ceil(time_mat[:,-1])

    def forward(self, x):
        # x is samples x channels df
        self.data = x
        x = self.scaler.transform(x)
        x = x.T
        slopes = ft_extract(x, self.fs, self.function, self.win_size, self.stride)
        scaled_slopes = slopes.squeeze()/self.nstds.reshape(-1,1)*self.fs
        scaled_slopes = scaled_slopes.squeeze()
        return scaled_slopes/1000
    
    def __call__(self, *args):
        return self.forward(*args)


class WVNT():
    """
    WaveNet-based seizure detection wrapper class.
    
    Utilizes a pre-trained convolutional neural network (WaveNet) for seizure detection.
    The model was previously trained on iEEG data to classify seizure vs non-seizure epochs.
    This wrapper handles data preprocessing, windowing, and probability extraction for
    real-time seizure detection applications.
    
    The WaveNet architecture is particularly effective at capturing temporal patterns
    in multi-channel neural data through dilated convolutions and residual connections.
    
    Parameters:
    -----------
    mdl : tensorflow.keras.Model
        Pre-trained WaveNet model loaded from disk
    win_size : float, default=1
        Window size in seconds for analysis
    stride : float, default=0.5  
        Window stride in seconds (overlap control)
    fs : int, default=128
        Sampling frequency for the model input
    """
    def __init__(self, mdl, win_size = 1, stride = 0.5, fs = 128):
        """Initialize WaveNet wrapper with model and windowing parameters."""
        self.win_size = win_size
        self.stride = stride
        self.fs = fs
        self.mdl = mdl
    
    def __str__(self) -> str:
        return "WVNT"
        
    def fit(self, x):
        """
        Fit RobustScaler to training data for normalization.
        
        Parameters:
        -----------
        x : pandas.DataFrame
            Training data (samples x channels)
        """
        self.scaler = RobustScaler().fit(x)

    def get_times(self, x):
        """
        Calculate time stamps for analysis windows.
        
        Parameters:
        -----------
        x : pandas.DataFrame
            Input data (samples x channels)
            
        Returns:
        --------
        numpy.ndarray
            Window end times in seconds
        """
        time_mat = MovingWinClips(np.arange(len(x))/self.fs,self.fs,self.win_size,self.stride)
        return np.ceil(time_mat[:,-1])

    def forward(self, x):
        """
        Generate seizure detection predictions using WaveNet model.
        
        Processes multi-channel iEEG data through sliding windows, applies
        normalization, and generates seizure probability for each window-channel pair.
        
        Parameters:
        -----------
        x : pandas.DataFrame
            Input iEEG data (samples x channels)
            
        Returns:
        --------
        numpy.ndarray
            Seizure probabilities (channels x windows)
        """
        # Store channel names and calculate dimensions
        chs = x.columns
        nwins = num_wins(len(x),self.fs,1,0.5)
        nch = len(chs)
        
        # Apply normalization and prepare data for WaveNet
        x = pd.DataFrame(self.scaler.transform(x),columns=chs)
        x = prepare_wavenet_segment(x)
        
        # Generate predictions (get seizure probability from class 1)
        y = self.mdl.predict(x)[:,1]
        
        # Reshape to channels x windows format
        return y.reshape(nwins,nch).T
        
    def __call__(self, *args):
        """Allow direct calling of the forward method."""
        return self.forward(*args)

# Train the model instance using provided data

def plot_and_save_detection(mat,win_times,yticks,fig_save_path,xlim = None):
    """
    Create and save seizure detection heatmap with time axis and channel labels.
    
    Generates a comprehensive visualization showing seizure probability over time
    for each channel, with seizure onset marked and proper axis labeling.
    
    Parameters:
    -----------
    mat : numpy.ndarray
        Seizure probability matrix (channels x time_windows)
    win_times : numpy.ndarray
        Time stamps for each window
    yticks : list
        Channel labels for y-axis
    fig_save_path : str
        Path to save the generated figure
    xlim : tuple, optional
        X-axis limits for zooming
    """
    # plt.subplots(figsize=(48,24))
    plt.imshow(mat)
    plt.axvline(np.argwhere(np.ceil(win_times)==120)[0])

    plt.xlabel('Time (s)')
    plt.yticks(np.arange(len(yticks)),yticks,rotation=0,fontsize=10)
    plt.xticks(np.arange(0,len(win_times),10),win_times.round(1)[np.arange(0,len(win_times),10)]-120)
    if xlim is not None:
        plt.xlim(xlim)
    plt.clim([0,4])
    plt.savefig(fig_save_path)

def plot_and_save_detection_figure(mat,win_times,yticks,fig_save_path,xlim = None,cmap=False):
    # plt.subplots(figsize=(48,24))
    plot_onset_lower = np.argwhere(np.ceil(win_times)==120)[0]
    plot_onset_upper = np.argwhere(np.ceil(win_times)==210)[0] if max(win_times) > 210 else mat.shape[1]
    plt.imshow(mat[:,int(plot_onset_lower):int(plot_onset_upper)],cmap=cmap)

    plt.xticks([])
    plt.yticks([])
    if xlim is not None:
        plt.xlim(xlim)
    plt.clim([0,1])
    plt.savefig(fig_save_path,bbox_inches='tight')

def main():
    """
    Main seizure detection pipeline using pre-trained models.
    
    Workflow:
    1. Load configuration and seizure metadata from BIDS format
    2. Configure GPU settings for optimal performance  
    3. For each patient with available data:
       - Load interictal training data from BIDS
       - Clean electrode labels and localize neural channels
       - For each seizure recording:
         - Train model on interictal/early seizure data
         - Generate predictions across full seizure recording
         - Save probability matrices and visualizations
    4. Support three detection algorithms: LSTM, AbsSlope, WaveNet
    
    Models are trained patient-specifically on interictal data and applied
    to detect seizure onset patterns in ictal recordings.
    """
    # Configure GPU memory growth for TensorFlow/PyTorch compatibility
    gpus = list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # Load configuration and paths
    _,_,datapath,prodatapath,metapath,figpath,patient_table,rid_hup,_ = load_config(ospj('/mnt/leif/littlab/users/wojemann/stim-seizures/code','config.json'),None)

    # Load seizure metadata from BIDS processing
    seizures_df = pd.read_csv(ospj(metapath,"stim_seizure_information_BIDS.csv"))

    # Detection parameters
    onset_time = 120          # Seizure onset time in recording (seconds)
    montage = 'bipolar'       # Electrode montage for preprocessing  
    train_win = 12           # Training window length (time points)
    pred_win = 1             # Prediction window length (time points)
    num_epochs = 10          # LSTM training epochs
    all_mdl_strs = ['AbsSlp','WVNT','LSTM']  # Models to run

    if 'WVNT' in all_mdl_strs:
        wave_model = load_model(ospj(prodatapath,'WaveNet','v111.hdf5'))

    # Iterating through each patient that we have annotations for
    pbar = tqdm(patient_table.iterrows(),total=len(patient_table))
    for _,row in pbar:
        pt = row.ptID
        pbar.set_description(desc=f"Patient: {pt}",refresh=True)
        
        # Skipping if no training data has been identified
        if len(row.interictal_training) == 0:
            continue

        # Loading data from bids
        inter_raw,fs_raw = get_data_from_bids(ospj(datapath,"BIDS"),pt,'interictal')

        # Pruning channels
        chn_labels = remove_scalp_electrodes(inter_raw.columns)
        inter_raw = inter_raw[chn_labels]

        try: # channel localization exception catch
            electrode_localizations,electrode_regions = electrode_wrapper(pt,rid_hup,datapath)
            if pt[:3] == 'CHO':
                suffix = ['CHOPR','CHOPM']
            else:
                suffix = ['dkt','atropos']

            electrode_localizations.name = clean_labels(electrode_localizations.name,pt) #don't end up using grey/white matter
            electrode_regions.name = clean_labels(electrode_regions.name,pt)
            electrode_localizations.to_pickle(ospj(prodatapath,pt,f'electrode_localizations_{suffix[1]}.pkl')) #don't end up using grey/white matter
            electrode_regions.to_pickle(ospj(prodatapath,pt,f'electrode_localizations_{suffix[0]}.pkl'))
            neural_channels = electrode_localizations.name[(electrode_localizations.name.isin(inter_raw.columns)) & ((electrode_localizations.label == 'white matter') | (electrode_localizations.label == 'gray matter'))]
        
        except:
            print(f"electrode localization failed for {pt}")
            neural_channels = chn_labels
        inter_neural = inter_raw.loc[:,neural_channels]
        
        # get baseline stds for stimulation artifact interpolation
        baseline_stds = inter_neural.std().to_numpy()

        for i_mdl,mdl_str in  enumerate(all_mdl_strs):
            wvcheck = mdl_str=='WVNT'
            # Preprocess the signal
            target=128
            inter_pre, fs, mask = preprocess_for_detection(inter_neural,fs_raw,montage,target=target,wavenet=wvcheck,pre_mask = None)

            seizure_times = seizures_df[seizures_df.Patient == pt]

            # Iterating through each seizure for that patient
            qbar = tqdm(seizure_times.iterrows(),total=len(seizure_times),leave=False)
            for i,(_,sz_row) in enumerate(qbar):
                if (pt == 'CHOP037') & (sz_row.approximate_onset == 962082.12):
                    continue
                set_seed(1071999)
                qbar.set_description(f"{mdl_str} processing seizure {i}")
                # Load in seizure and metadata for BIDS path
                seizure,fs_raw, _, _, task, run = get_data_from_bids(ospj(datapath,"BIDS"),pt,str(int(sz_row.approximate_onset)),return_path=True, verbose=0)

                # Filter out bad channels from interictal clip
                seizure = seizure[neural_channels]

                # Interpolating stimulation artifact
                if sz_row.stim == 1:
                    stim_chs = np.zeros((len(seizure.columns),),dtype=bool)
                    for ch in sz_row.stim_channels.split('-'):
                        ch = clean_labels([ch],pt)[0]
                        stim_chs += np.array([ch == c for c in seizure.columns])
                    pk_idxs,_ = stim_detect(seizure,threshold=baseline_stds*100,fs=fs_raw)
                    seizure = barndoor(seizure,pk_idxs,fs_raw,plot=False)
                    seizure = seizure.iloc[:,~stim_chs]
                
                # Preprocess seizure for seizure detection task
                seizure_pre, fs = preprocess_for_detection(seizure,fs_raw,montage,target=target,wavenet=wvcheck,pre_mask=mask)
                
                noisy_channel_mask = seizure_pre.loc[onset_time*fs:,:].abs().max() <= (np.median(seizure_pre.loc[onset_time*fs:,:].abs().max())*50)
                # noisy_channel_list = seizure_pre.columns[noisy_channel_mask].to_list()
                seizure_pre = seizure_pre.loc[:,noisy_channel_mask]

                # Perform overwrite check
                prob_path = f"pretrain_probability_matrix_nosmooth_mdl-{mdl_str}_fs-{int(fs)}_montage-{montage}_task-{task}_run-{run}.pkl"
                if (not OVERWRITE) and ospe(ospj(prodatapath,pt,prob_path)):
                    continue

                if sz_row.stim == 1:    
                    sz_train = inter_pre.loc[:,seizure_pre.columns]
                else:
                    sz_train = seizure_pre.loc[:fs*60,:]

                if mdl_str in ['LSTM']:
                    ##############################
                    input_size = sz_train.shape[1]
                    hidden_size = 10
                    # Check for cuda
                    ccheck = torch.cuda.is_available()
                    # Initialize the model
                    if mdl_str == 'LSTM':
                        model = LSTMModel(input_size, hidden_size)
                    if ccheck:
                        model.cuda()

                    # Scale the training data
                    model.fit_scaler(sz_train)
                    sz_train_z = model.scaler_transform(sz_train)
                    sz_train_z = pd.DataFrame(sz_train_z,columns=sz_train.columns)

                    # Prepare input and target data for the LSTM
                    input_data,target_data = prepare_segment(sz_train_z,fs=fs)

                    dataset = TensorDataset(input_data, target_data)
                    full_batch = len(dataset)
                    dataloader = DataLoader(dataset, batch_size=full_batch, shuffle=False)
                    
                    # Define loss function and optimizer
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=0.01)

                    # Train the model, this will just modify the model object, no returns
                    train_model(model,dataloader,criterion,optimizer,ccheck=ccheck,num_epochs=num_epochs)
                    
                    ################################################
                    seizure_z = model.scaler_transform(seizure_pre)
                    seizure_z = pd.DataFrame(seizure_z,columns=seizure_pre.columns)
                    input_data, target_data,time_wins = prepare_segment(seizure_z,fs,train_win,pred_win,ret_time=True)
                    # Generate seizure detection predictions for each window
                    outputs = predict_sz(model,input_data,target_data,batch_size=len(input_data)//2,ccheck=ccheck)
                    seizure_mat = repair_data(outputs,seizure_z,fs=fs)
                    # Getting raw predicted loss values for each window
                    raw_sz_vals = np.sqrt(np.mean(seizure_mat,axis=1)).T
                    # Creating classifications
                    mdl_outs = raw_sz_vals
                    ###
                
                elif mdl_str in ['AbsSlp','WVNT']:
                    if mdl_str == 'AbsSlp':
                        model = AbsSlope(1,.5, fs)
                        model.fit(sz_train)
                    elif mdl_str == 'WVNT':
                        model = WVNT(wave_model,1,.5,fs)
                        model.fit(sz_train)
                    mdl_outs = model(seizure_pre)
                    time_wins = model.get_times(seizure_pre)

                sz_prob = mdl_outs.copy()
                sz_prob_df = pd.DataFrame(sz_prob.T,columns = seizure_pre.columns)
                time_df = pd.Series(time_wins,name='time')
                sz_prob_df = pd.concat((sz_prob_df,time_df),axis=1)
                os.makedirs(ospj(prodatapath,pt),exist_ok=True)
                sz_prob_df.to_pickle(ospj(prodatapath,pt,prob_path))
                
                ### Visualization
                detect_idx = np.argwhere(np.ceil(time_wins)==120)[0]
                first_detect = np.argmax(sz_prob[:,int(detect_idx):]>.5,axis=1)
                first_detect[first_detect == 0] = sz_prob.shape[1]
                ch_sorting = np.argsort(first_detect)
                colors = sns.color_palette("deep", len(all_mdl_strs))
                # Plot heatmaps for the first 4 colors
                cmap = LinearSegmentedColormap.from_list('custom_cmap', [(1, 1, 1), colors[i_mdl]])
                os.makedirs(ospj(figpath,pt,"annotations",str(int(sz_row.approximate_onset)),mdl_str),exist_ok=True)
                plot_and_save_detection_figure(sz_prob,
                                        time_wins,
                                        seizure.columns[ch_sorting],
                                        ospj(figpath,pt,"annotations",str(int(sz_row.approximate_onset)),mdl_str,f"{montage}_sz_prob_colored.png"),
                                        cmap = cmap)
                plot_and_save_detection(sz_prob,
                                        time_wins,
                                        seizure.columns[ch_sorting],
                                        ospj(figpath,pt,"annotations",str(int(sz_row.approximate_onset)),mdl_str,f"{montage}_sz_prob.png"),
                                        )
                del model
if __name__ == "__main__":
    main()