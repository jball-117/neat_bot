## COMBINES DATAFRAMES OF PARSED REPLAY FILES ##
import pandas as pd
from tqdm import tqdm
import os

GRANULARITY = 0 # edit this. 1 -> 8x_10y_7z, .5 -> 16x_20y_14z, 0 -> exact

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

## combining replays ##
#rootdir = '/home/zach/Files/Nas/Replays'
if GRANULARITY == 1:
    rootdir = '/home/zach/Files/ReplayModels/ReplayDataProcessing/RANKED_STANDARD/Replays/1400-1600/CSVs_8x_10y_7z/'
elif GRANULARITY == .5:
    rootdir = '/home/zach/Files/ReplayModels/ReplayDataProcessing/RANKED_STANDARD/Replays/1400-1600/CSVs_16x_20y_14z/'
elif GRANULARITY == 0:
    rootdir = '/home/zach/Files/ReplayModels/ReplayDataProcessing/RANKED_STANDARD/Replays/1400-1600/CSVs/'

for root, dirs, files in os.walk(rootdir):
    all_dfs = pd.read_csv(rootdir+"/"+files[0], low_memory=False)
    all_dfs.drop(columns=['Unnamed: 0'], inplace=True)
    all_dfs = fix_ot_secs(all_dfs)
    all_dfs['0_pos_x_nf'] = all_dfs['0_pos_x'].shift(-1)
    all_dfs['0_pos_y_nf'] = all_dfs['0_pos_y'].shift(-1)
    all_dfs['0_pos_z_nf'] = all_dfs['0_pos_z'].shift(-1)
    all_dfs = all_dfs[:-1]
    for filename in tqdm(files[1:]):
        if not filename.endswith('.csv'):
            print("\n", filename, "not a csv\n")
            continue
        piece = pd.read_csv(rootdir+"/"+filename, low_memory=False)
        piece.drop(columns=['Unnamed: 0'], inplace=True)
        piece = fix_ot_secs(piece)
        piece['0_pos_x_nf'] = piece['0_pos_x'].shift(-1)
        piece['0_pos_y_nf'] = piece['0_pos_y'].shift(-1)
        piece['0_pos_z_nf'] = piece['0_pos_z'].shift(-1)
        piece = piece[:-1]
        all_dfs = all_dfs.append(piece, ignore_index=True)

    print(len(all_dfs))
    print("WRITING...")
    all_dfs.to_csv("exact_train.csv")
