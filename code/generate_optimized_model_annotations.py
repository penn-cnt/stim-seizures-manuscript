# iEEG imports
from ieeg.auth import Session

# Scientific computing imports
import numpy as np
import scipy as sc
import pandas as pd
from tqdm import tqdm


# Plotting
import matplotlib.pyplot as plt

# OS imports
from os.path import join as ospj
from utils import *
import sys
sys.path.append('/users/wojemann/iEEG_processing')

plt.rcParams['image.cmap'] = 'magma'

def main():
    _,_,datapath,prodatapath,metapath,_,patient_table,_,_ = load_config(ospj('/mnt/leif/littlab/users/wojemann/stim-seizures/code','config.json'),None)

    seizures_df = pd.read_csv(ospj(metapath,"stim_seizure_information_BIDS.csv"))
    montage = 'bipolar'
    mdl_str = 'LSTM'
    clf_fs = 128
    onset_time = 120
    funs = ['mean','med']
    param_list = []
    # OPTIMAL: mean, mean, med
    for pt_comb in ['mean']:
        for sz_comb in ['mean']:
            for smooth in ['med']:
                for v in [4]:#[2,3]:
                    param_list.append(dict(
                        smooth=smooth,
                        pt_comb=pt_comb,
                        sz_comb=sz_comb,
                        v=v
                    ))
                    
    def par_fun(params):
        # Iterating through each patient that we have annotations for
        predicted_channels = {'Patient': [],
                            'iEEG_ID': [],
                            'model':[],
                            'stim':[],
                            'approximate_onset': [],
                            'offset': [],
                            'threshold':[],
                            'ueo_chs_strict':[],
                            'ueo_chs_loose': [],
                            'sec_chs_strict': [],
                            'sec_chs_loose':[],
                            'sz_chs': [],
                            'sz_times': [],
                            'all_channels': []
                            }
        smooth=params['smooth']
        pt_comb=params['pt_comb']
        sz_comb=params['sz_comb']
        v=params['v']

        tuned_thresholds = pd.read_pickle(ospj(prodatapath,f"patient_tuned_classification_thresholds_stim_sz-{sz_comb}.pkl"))

        pbar = tqdm(patient_table.iterrows(),total=len(patient_table))
        
        for _,row in pbar:
            pt = row.ptID
            # if pt != 'CHOP041':
            #     continue
            pbar.set_description(desc=f"Patient -- {pt}",refresh=True)
            if (len(row.interictal_training) == 0) or (pt not in tuned_thresholds.Patient.to_numpy()):
                continue

            pt_thresh = tuned_thresholds[(tuned_thresholds.Patient == pt) & (tuned_thresholds.model == mdl_str)]
            if 0 not in pt_thresh.stim.values:
                if sz_comb == 'mean':
                    thresholds = [1.6060201480762224, pt_thresh.threshold.item()]
                else:
                    thresholds = [1.6900109236557836, pt_thresh.threshold.item()]
            else:
                thresholds = pt_thresh.sort_values('stim').threshold.to_list()
            if v>2:
                if sz_comb == 'mean':
                    thresholds[0] = 1.6060201480762224
                else:
                    thresholds[0] = 1.6900109236557836
            seizure_times = seizures_df[seizures_df.Patient == pt]
            qbar = tqdm(seizure_times.iterrows(),total=len(seizure_times),desc = 'Seizures',leave=False)

            for _,sz_row in qbar:
                if (pt == 'CHOP037') & (sz_row.approximate_onset == 962082.12):
                        continue
                _,_, _, _, task, run = get_data_from_bids(ospj(datapath,"BIDS"),pt,str(int(sz_row.approximate_onset)),return_path=True, verbose=0)
                prob_path = f"pretrain_probability_matrix_nosmooth_mdl-{mdl_str}_fs-{clf_fs}_montage-{montage}_task-{task}_run-{run}.pkl"
                sz_prob = pd.read_pickle(ospj(prodatapath,pt,prob_path))
                time_wins = sz_prob.time.to_numpy()
                sz_prob.drop('time',axis=1,inplace=True)
                prob_chs = sz_prob.columns.to_numpy()
                sz_prob = sz_prob.to_numpy().T
                if smooth == 'med':
                    sz_prob = sc.ndimage.median_filter(sz_prob,size=20,mode='nearest',axes=1,origin=0)
                else:
                    sz_prob = sc.ndimage.uniform_filter1d(sz_prob,size=20,mode='nearest',axis=1,origin=0)

                threshold = thresholds[int(sz_row.stim)]

                # Generate predicitons
                predicted_channels['Patient'].append(sz_row.Patient)
                predicted_channels['iEEG_ID'].append(sz_row.IEEGname)
                predicted_channels['model'].append(mdl_str)
                predicted_channels['stim'].append(sz_row.stim)
                predicted_channels['approximate_onset'].append(sz_row.approximate_onset)
                predicted_channels['offset'].append(sz_row.end)
                predicted_channels['threshold'].append(threshold)
                predicted_channels['all_channels'].append(np.array([s.split("-")[0] for s in prob_chs]).flatten())
                
                sz_clf_final = sz_prob > threshold

                first_sz_idx_offset = np.argmin(np.abs(time_wins-onset_time))
                
                # Get channels
                sliced_data = sz_clf_final[:,first_sz_idx_offset:]
                df = pd.DataFrame(sliced_data).T
                seizing = df.rolling(window=10,closed='right').apply(lambda x: sum(x == 1)>8)

                first_sz_idxs = seizing.idxmax().to_numpy() - 9
                seized_idxs = np.any(sliced_data,axis=1)
                first_sz_idxs += first_sz_idx_offset
                if sum(seized_idxs) > 0:
                    sz_times_arr = time_wins[first_sz_idxs[seized_idxs]]
                    sz_times_arr -= np.min(sz_times_arr)
                    sz_ch_arr = prob_chs[seized_idxs]
                    sz_ch_arr = np.array([s.split("-")[0] for s in sz_ch_arr]).flatten()
                    onset_index = np.min(first_sz_idxs[seized_idxs])
                else:
                    sz_ch_arr = []
                    sz_times_arr = []
                    onset_index = np.min(first_sz_idxs)
                predicted_channels['sz_chs'].append(sz_ch_arr)
                predicted_channels['sz_times'].append(sz_times_arr)

                new_onset_time = time_wins[onset_index]
                spread_index = np.argmin(np.abs(time_wins-(new_onset_time + 10)))

                mdl_ueo_idx = np.sum(sz_clf_final[:,onset_index:onset_index+10],axis=1) > 8
                mdl_ueo_ch_bp = prob_chs[mdl_ueo_idx]
                mdl_ueo_ch_strict = np.array([s.split("-")[0] for s in mdl_ueo_ch_bp]).flatten()
                mdl_ueo_ch_loose = np.unique(np.array([s.split("-") for s in mdl_ueo_ch_bp]).flatten())
                predicted_channels['ueo_chs_strict'].append(mdl_ueo_ch_strict)
                predicted_channels['ueo_chs_loose'].append(mdl_ueo_ch_loose)

                mdl_sec_idx = np.sum(sz_clf_final[:,spread_index:spread_index+10],axis=1) > 8
                mdl_sec_ch_bp = prob_chs[mdl_sec_idx]
                mdl_sec_ch_strict = np.array([s.split("-")[0] for s in mdl_sec_ch_bp]).flatten()
                mdl_sec_ch_loose = np.unique(np.array([s.split("-") for s in mdl_sec_ch_bp]).flatten())
                predicted_channels['sec_chs_strict'].append(mdl_sec_ch_strict)
                predicted_channels['sec_chs_loose'].append(mdl_sec_ch_loose)

        predicted_channels = pd.DataFrame(predicted_channels)
        predicted_channels.to_pickle(ospj(prodatapath,f"optimized_predicted_channels_{mdl_str}_tuned_thresholds_v{v}_sz-{sz_comb}_pt-{pt_comb}_smooth-{smooth}.pkl"))
    _ = in_parallel(par_fun,param_list)
    # predicted_channels.to_csv(ospj(prodatapath,"optimized_predicted_channels.csv"))
if __name__ == "__main__":
    main()