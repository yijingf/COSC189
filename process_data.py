import os
import pandas as pd, numpy as np

import nibabel as nib
import torch
root_dir = './'


def split_data_with_label():
    # load label and split intervals
    log_fname_pattern = os.path.join(root_dir, 'data/ds113b','sub{:03d}', 'behav', 'task002_run{:03d}', 'behavdata.txt')
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
    spatial_data_path = "./data/spatial"
    if not os.path.exists(spatial_data_path):
        os.mkdir(spatial_data_path)

    if not os.path.exists(os.path.join(spatial_data_path, 'train')):
        os.mkdir(os.path.join(spatial_data_path, 'train'))
        os.mkdir(os.path.join(spatial_data_path, 'val'))
        os.mkdir(os.path.join(spatial_data_path, 'test'))

    fmri_fname = os.path.join(root_dir, 'data/ds113b', 'sub{:03d}', 'BOLD','task002_run{:03d}', 'bold.nii.gz')
    interval_ind = 0
    t_ind = 0  # num of training samples

    for sub_id in range(1, 21):
        for run_id in range(1, 9):
            fmri_fname = fmri_fname.format(sub_id, run_id)
            try:
                fmri_img = nib.load(fmri_fname)
            except:
                print("Missing Data: {}".format(fmri_fname, 2))
                continue
            if interval_ind >= len(interval_list): break
            # split in time
            whole_time_brain = np.array(fmri_img.get_fdata(), dtype=np.float32).transpose(2, 3, 0, 1)  # z, time, x, y

            if len(interval_list) * 0.75 > interval_ind:
                file_type = 'train'
            elif len(interval_list) * 0.85 > interval_ind:
                file_type = 'val'
            else:
                file_type = 'test'

            for i, start in enumerate(interval_list[interval_ind]):
                if i != len(interval_list[interval_ind]) - 1:
                    img_label_dict = {'img':whole_time_brain[:, :, :, start:interval_list[interval_ind][i+1]],
                                      'label':label_list[interval_ind][i]}
                else:
                    img_label_dict = {'img':whole_time_brain[:, :, :, start:],
                                      'label':label_list[interval_ind][i]}
                torch.save(img_label_dict, os.path.join(spatial_data_path, file_type, f"spatial_img_labels_{t_ind}.pt"))
                t_ind += 1
            interval_ind += 1  # to next run_id
    print(interval_ind, t_ind)


if __name__ == '__main__':
    split_data_with_label()