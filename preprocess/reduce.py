import numpy as np
from glob import glob
from tqdm import tqdm

f_list = sorted(glob('./data/*.npz'))
template = np.load('./roi_template.npy')
index = np.where(template)

for f_name in tqdm(f_list[41:]):
    try:
        data = np.load(f_name)['a']
    except:
        continue
    _, _, _, t = data.shape
    data = np.array([data[:,:,:,i][np.where(template)] for i in range(t)]).T
    np.savez(f_name, data=data)
