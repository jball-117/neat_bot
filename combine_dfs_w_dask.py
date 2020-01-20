## COMBINES DATAFRAMES OF PARSED REPLAY FILES ##
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

ProgressBar().register()

GRANULARITY = 0 # edit this. 1 -> 8x_10y_7z, .5 -> 16x_20y_14z, 0 -> exact
FRAMES_AHEAD = 15

## NN shouldn't care about game time in OT
def fix_ot_secs(df):
    if len(df[df['seconds_remaining'] == -1.0]) > 0:
        print('already fixed')
        return df
    i=0
    try:
        while df.at[i, 'seconds_remaining'] >= df.at[i+1, 'seconds_remaining']:
            i += 1
        i += 1
        for j in range(i, len(df)):
            df.at[j, 'seconds_remaining'] = -1
        return df
    except:
        return df

def manip(df):
    df = df.drop(columns=['Unnamed: 0'])
    df = fix_ot_secs(df)
    df['0_pos_x_nf'] = df['0_pos_x'].shift(-1*FRAMES_AHEAD)
    df['0_pos_y_nf'] = df['0_pos_y'].shift(-1*FRAMES_AHEAD)
    df['0_pos_z_nf'] = df['0_pos_z'].shift(-1*FRAMES_AHEAD)
    df = df.head(len(df.index)-FRAMES_AHEAD)
    return df

## combining replays ##
if GRANULARITY == 1:
    rootdir = '/home/zach/Files/Nas/ReplayModels/ReplayDataProcessing/RANKED_STANDARD/Replays/1400-1600/CSVs_8x_10y_7z/'
elif GRANULARITY == .5:
    rootdir = '/home/zach/Files/Nas/ReplayModels/ReplayDataProcessing/RANKED_STANDARD/Replays/1400-1600/CSVs_16x_20y_14z/'
elif GRANULARITY == 0:
    rootdir = '/home/zach/Files/Nas/ReplayModels/ReplayDataProcessing/RANKED_STANDARD/Replays/1400-1600/CSVs/'

all_dfs = dd.read_csv(rootdir+"/*.csv")
print(all_dfs.columns)
all_dfs = all_dfs.map_partitions(manip)

print(all_dfs.shape[0].compute())
print("WRITING...")
if GRANULARITY == 0:
    all_dfs.to_csv("exact_train_"+str(FRAMES_AHEAD)+"_frames"+".csv")
if GRANULARITY == .5:
    all_dfs.to_csv("train_16x_20y_14z_"+str(FRAMES_AHEAD)+"_frames"+".csv")
if GRANULARITY == 1:
    all_dfs.to_csv("train_8x_10y_7z_"+str(FRAMES_AHEAD)+"_frames"+".csv")
