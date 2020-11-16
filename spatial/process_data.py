import os
import pandas as pd, numpy as np

import nibabel as nib
import torch
root_dir = './'

group_root_dir = '/isi/music/OpenFMRI/forrest_gump'

# load label and split intervals
feature = "temporal"
mod = "_subjects_warp" # or "run": "_runs_out_warp"

data_path = os.path.join(root_dir, "data", feature)
data_key = {'temporal':'data', 'spatial':'a'}

def main():
    log_fname_pattern = os.path.join(group_root_dir,'sub{:03d}', 'behav', 'task002_run{:03d}', 'behavdata.txt')
    interval_list = []
    label_list = []
    for sub_id in range(1, 21):
        for run_id in range(1, 9):
            log_fname = log_fname_pattern.format(sub_id, run_id)
            try:
                log_data = pd.read_csv(log_fname)
            except:
                print("Missing Data: {}".format(log_fname, 2))
                continue
            volume_start = log_data['run_volume'].to_numpy()
            interval_list.append(volume_start)
            label_list.append(log_data['genre'].to_numpy())

    # split the runs data into train, val, test according to the ratio 7:1:2, len(interval_list) = 156
    print(len(interval_list), len(interval_list[0]), interval_list[0])

    # store the img and split them according to the time stamp, set the z dim as channel so we need transpose here.
    # save spatial data

    os.makedirs(data_path, exist_ok='True')
    os.makedirs(os.path.join(data_path, f'train{mod}'))
    os.makedirs(os.path.join(data_path, f'val{mod}'))
    os.makedirs(os.path.join(data_path, f'test{mod}'))

    fmri_fname = os.path.join(root_dir, 'data', 'sub{:03d}_run{:03d}.npz')
    
    interval_ind = 0
    t_ind = 0  # num of training samples
    
    for sub_id in range(1, 21):
        for run_id in range(1, 9):
            fmri_fname = fmri_fname.format(sub_id, run_id)
            try:
                # fmri_img = nib.load(fmri_fname)
                # whole_time_brain = np.array(fmri_img.get_fdata(), dtype=np.float32).transpose(2,3,0,1) # split in time
                whole_time_brain = np.load(fmri_fname)[data_key[feature]]
                if data_key == 'spatial':
                    whole_time_brain = fmri_img.astype(np.float32).transpose(2,3,0,1)
            except:
                print("Missing Data: {}".format(fmri_fname, 2))
                continue

            if interval_ind >= len(interval_list):
                break

            # Split train test data
            if 'subject' in mod:
                # split according the subjects
                if len(interval_list) * 0.75 > interval_ind:
                    file_type = f'train{mod}'
                elif len(interval_list) * 0.85 > interval_ind:
                    file_type = f'val{mod}'
                else:
                    file_type = f'test{mod}'
            else:
                # split according to the runs_out
                run_size = 8 if sub_id < 20 else 4

                if run_size * 0.8 > run_id:
                    file_type = f'train{mod}'
                elif run_size * 0.9 > run_id:
                    file_type = f'val{mod}'
                else:
                    file_type = f'test{mod}'

            # get data epoch
            for i, start in enumerate(interval_list[interval_ind]):
                end = min(len(interval_list[interval_ind]), interval_list[interval_ind][i+1])
                label = label_list[interval_ind][i]
                if feature == 'spatial':
                    data_label_dict = {'img':whole_time_brain[:, start:start, :, :], 'label':label}
                else:
                    data_label_dict = {'img':whole_time_brain[:, start:end],'label': label}

                torch.save(data_label_dict, os.path.join(data_path, file_type, f"{feature}_img_labels_{t_ind}.pt"))
                t_ind += 1
            interval_ind += 1  # to next run_id
        print(sub_id)
    print(interval_ind, t_ind)
    
    return


if __name__ == '__main__':
    main()