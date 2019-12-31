## COMBINES DATAFRAMES OF PARSED REPLAY FILES ##
import pandas as pd
from tqdm import tqdm
import os

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
rootdir = '/home/zach/Files/ReplayModels/ReplayDataProcessing/RANKED_STANDARD/Replays/1400-1600/CSVs/'
for root, dirs, files in os.walk(rootdir):
    all_dfs = pd.read_csv(rootdir+"/"+files[0])
    all_dfs.drop(columns=['Unnamed: 0'], inplace=True)
    all_dfs = fix_ot_secs(all_dfs)
    all_dfs['0_pos_x'] = all_dfs['0_pos_x'].shift(-1)
    all_dfs['0_pos_y'] = all_dfs['0_pos_y'].shift(-1)
    all_dfs['0_pos_z'] = all_dfs['0_pos_z'].shift(-1)
    all_dfs = all_dfs[:-1]
    for filename in tqdm(files[1:]):
        if not filename.endswith('.csv'):
            print("\n", filename, "not a csv\n")
            continue
        piece = pd.read_csv(rootdir+"/"+filename)
        piece.drop(columns=['Unnamed: 0'], inplace=True)
        piece = fix_ot_secs(piece)
        piece['0_pos_x'] = piece['0_pos_x'].shift(-1)
        piece['0_pos_y'] = piece['0_pos_y'].shift(-1)
        piece['0_pos_z'] = piece['0_pos_z'].shift(-1)
        piece = piece[:-1]
        all_dfs = all_dfs.append(piece, ignore_index=True)

    print(len(all_dfs))
    print("WRITING...")
    all_dfs.to_csv("exact_train.csv")
