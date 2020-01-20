import glob
import d6tstack.combine_csv

GRANULARITY = 0

if GRANULARITY == 1:
    rootdir = '/home/zach/Files/Nas/ReplayModels/ReplayDataProcessing/RANKED_STANDARD/Replays/1400-1600/CSVs_8x_10y_7z/'
elif GRANULARITY == .5:
    rootdir = '/home/zach/Files/Nas/ReplayModels/ReplayDataProcessing/RANKED_STANDARD/Replays/1400-1600/CSVs_16x_20y_14z/'
elif GRANULARITY == 0:
    rootdir = '/home/zach/Files/Nas/ReplayModels/ReplayDataProcessing/RANKED_STANDARD/Replays/1400-1600/CSVs/'

cfg_fnames = list(glob.glob(rootdir+"/*.csv"))
c = d6tstack.combine_csv.CombinerCSV(cfg_fnames)

# check columns
print('all equal',c.is_all_equal())
print('')
c.is_column_present()

if not c.is_all_equal():
    d6tstack.combine_csv.CombinerCSV(cfg_fnames).to_csv_align(output_dir=rootdir+"dask_CSVs")
