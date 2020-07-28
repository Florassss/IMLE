import sys
import os.path
import glob
import pickle
import lmdb
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.progress_bar import ProgressBar

# configurations
# img_folder = '/home/nio/SRIM/results/RRDB_MS_RRDB_16_x8_Butterfly/n02279972_test/*'  # glob matching pattern
# lmdb_save_path = '/home/nio/data/ms_16_gen_test_256.lmdb'  # must end with .lmdb
# img_folder = '/home/nio/SRIM/results/RRDB_IM_RRDB_5_x8_Lamp/n03637318/*'  # glob matching pattern
# lmdb_save_path = '/home/nio/data/im_5_gen_256.lmdb'  # must end with .lmdb

# img_folder = '/mnt/disks/store/SRIM/results/RRDB_IM_RRDB_5_x8_Lamp/n03637318/*'  # glob matching pattern
# lmdb_save_path = '/home/s5peng/data/im_4_gen_256.lmdb'  # must end with .lmdb

img_folder = '/mnt/disks/store/SRIM/results/RRDB_MS_RRDB_25_x8_Bird/n01531178/*'  # glob matching pattern
lmdb_save_path = '/home/s5peng/data/ms_25_gen_256.lmdb'  # must end with .lmdb

# img_folder = '/mnt/disks/store/SRIM/results/RRDB_MS_RRDB_26_x8_Lamp/n03637318/*'  # glob matching pattern
# lmdb_save_path = '/home/s5peng/data/ms_26_gen_256.lmdb'  # must end with .lmdb

img_list = sorted(glob.glob(img_folder))
dataset = []
data_size = 0

print('Read images...')
pbar = ProgressBar(len(img_list))
for i, v in enumerate(img_list):
    pbar.update('Read {}'.format(v))
    img = cv2.imread(v, cv2.IMREAD_UNCHANGED)
    dataset.append(img)
    data_size += img.nbytes
env = lmdb.open(lmdb_save_path, map_size=data_size * 10)
print('Finish reading {} images.\nWrite lmdb...'.format(len(img_list)))

pbar = ProgressBar(len(img_list))
with env.begin(write=True) as txn:  # txn is a Transaction object
    for i, v in enumerate(img_list):
        pbar.update('Write {}'.format(v))
        base_name = os.path.splitext(os.path.basename(v))[0]
        key = base_name.encode('ascii')
        data = dataset[i]
        if dataset[i].ndim == 2:
            H, W = dataset[i].shape
            C = 1
        else:
            H, W, C = dataset[i].shape
        meta_key = (base_name + '.meta').encode('ascii')
        meta = '{:d}, {:d}, {:d}'.format(H, W, C)
        # The encode is only essential in Python 3
        txn.put(key, data)
        txn.put(meta_key, meta.encode('ascii'))
print('Finish writing lmdb.')

# create keys cache
keys_cache_file = os.path.join(lmdb_save_path, '_keys_cache.p')
env = lmdb.open(lmdb_save_path, readonly=True, lock=False, readahead=False, meminit=False)
with env.begin(write=False) as txn:
    print('Create lmdb keys cache: {}'.format(keys_cache_file))
    keys = [key.decode('ascii') for key, _ in txn.cursor()]
    pickle.dump(keys, open(keys_cache_file, "wb"))
print('Finish creating lmdb keys cache.')