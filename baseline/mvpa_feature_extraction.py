import os
import numpy as np
import pandas as pd
from mvpa2.datasets import Dataset
from mvpa2.datasets.eventrelated import fit_event_hrf_model
from mvpa2.featsel.base import SensitivityBasedFeatureSelection

from mvpa2.measures.anova import OneWayAnova
from mvpa2.featsel.base import SensitivityBasedFeatureSelection
from mvpa2.featsel.helpers import FixedNElementTailSelector

from config import info_dir, data_dir
info = pd.read_csv(info_dir)

# Feature Extraction
def load_data(data_path, sub_id):
    labels = []
    data = []

    for run_id in range(1, 9):
        tmp_label = info.loc[(info['sub'] == sub_id) & (info['run'] == run_id), 'label']
        if not len(tmp_label):
            continue
        labels.append(tmp_label.to_list())
        tmp_data = np.load(os.path.join(data_path, 'sub{:03d}_run{:03d}.npy'.format(sub_id, run_id)))
        data.append(tmp_data)
    
    data = np.concatenate(data)
    labels = np.concatenate(labels)
    
    return data, labels.astype(np.int)

rois = ['aSTG', 'HG', 'pSTG']

for sub_id in range(1, 21):
    data = []
    for roi in rois:
        data_path = os.path.join(data_dir, roi)
        tmp_data, label = load_data(data_path, sub_id)
        data.append(tmp_data)
    data = np.concatenate(data, axis=1)
    data = np.concatenate([data[i,:,:].T for i in range(len(data))])

    ds = Dataset(data)
    ds.sa['time_coords'] = np.linspace(0, len(ds)-1, len(ds))
    events = [{'onset': i*5, 'duration': 5, 'targets':label[i], 'chunks':i+1} for i in range(int(len(ds)/5))]

    hrf_estimates = fit_event_hrf_model(ds, events, time_attr='time_coords', condition_attr=('targets', 'chunks'), 
                                    design_kwargs=dict(drift_model='blank'), glmfit_kwargs=dict(model='ols'),
                                    return_model=True)

    fsel = SensitivityBasedFeatureSelection(OneWayAnova(), FixedNElementTailSelector(5000, mode='select', tail='upper'))

    fsel.train(hrf_estimates)
    ds_p = fsel(hrf_estimates)

    np.save('feat_sub{:03d}'.format(sub_id), ds_p.samples)