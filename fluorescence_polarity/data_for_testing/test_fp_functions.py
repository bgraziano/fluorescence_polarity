import fluorescence_polarity as fp
import numpy as np
import skimage as sk
import pandas as pd
from pathlib import Path
pd.options.mode.chained_assignment = None # suppress waring messages for in-place dataframe edits

ipath = Path('data_for_testing/test_images/')
dpath = Path('data_for_testing/')

print(ipath)
print(dpath)

print(file1.stem)

cell_tr = pd.read_csv((dpath / 'test_DataFrame.csv'), index_col=0)
txy_test = pd.read_csv((dpath / 'fluor_polarity_txy_test.csv'), index_col=0)
xy_test = pd.read_csv((dpath / 'fluor_polarity_xy_test.csv'), index_col=0)

fluor = sk.io.MultiImage(str(ipath) + str(Path('/')) + 'fluor*')
fluor = sk.io.concatenate_images(fluor)
mask = sk.io.MultiImage(str(ipath) + str(Path('/')) + 'mask*')
mask = sk.io.concatenate_images(mask)
fluor1 = fluor[0,:,:]
mask1 = mask[0,:,:]

polarityscores_all = fp.fluor_polarity_txy(fluor, mask, cell_tr)
polarityscores_single = fp.fluor_polarity_xy(fluor1, mask1)

calc_txy = polarityscores_all.round(decimals=10)
truth_txy = txy_test.round(decimals=10)
assert np.all(calc_txy == truth_txy)

calc_xy = polarityscores_single.round(decimals=10)
truth_xy = xy_test.round(decimals=10)
assert np.all(calc_xy == truth_xy)

print('No errors found.')