import os
import pandas as pd
import sys
import numpy as np
import nibabel as nib
from config import root_dir

log_pattern = os.path.join(root_dir, '{}', 'behav', 'task002_{}', 'behavdata.txt')

def main(fname):
    fmri_img = nib.load(fname)
    data = fmri_img.get_fdata()
    data = data[:132, :175, :, :]
    
    sub, run = os.path.basename(fname).split('_')
    log_fname = log_pattern.format(sub, run.split('.')[0])
    nv = pd.read_csv(log_fname)['run_volume'].iloc[-1] + 6
    
    _, _, _, data_nv = data.shape
    data = data[:,:,:,:min(data_nv, nv)]
    
    output_fname = fname.split('.')[0]
    np.savez_compressed(output_fname, a=data)
    return 

if __name__ == '__main__':
    fname = sys.argv[1]
    main(fname)
