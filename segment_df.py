## TAKES IN DF FROM AND SEGMENTS POSITIONS ##
## WITH GRANULARITY OF 1 THERE WILL BE 8 X SEGMENTS, 10 Y SEGMENTS, 
## AND 7 Z SEGMENTS. WITH GRANULARITY OF .5 THERE WILL BE 16 X SEGMENTS, 20 Y
## SEGMENTS, AND 14 Z SEGMENTS ETC.
import pandas as pd
from numpy import isnan
from tqdm import tqdm
import os

GRANULARITY = .5 # edit this

OBx = []
OBy = []
OBz = []

class OutofBounds(Exception):
    """Raised when the segment is OB"""
    pass

def x_segment(x):
    if isnan(x):
        return x
    if x > 4096 or x < -4096:
        print("x:", x)
        OBx.append(x)
        raise OutofBounds
    seg = -4096
    while(True):
        if x >= seg and x < seg+(1024*GRANULARITY):
            return (seg+(seg+(1024*GRANULARITY)))/2
        seg += (1024*GRANULARITY)
        
def y_segment(y):
    if isnan(y):
        return y
    seg = -5120
    if y < -5120:
        return -5121 # IN BLUE TEAM GOAL
    elif y > 5120:
        return 5121 # IN ORANGE TEAM GOAL
    while(True):
        if y >= seg and y < seg+(1024*GRANULARITY):
            return (seg+(seg+(1024*GRANULARITY)))/2
        seg += (1024*GRANULARITY)
        
def z_segment(z):
    if isnan(z):
        return z
    if z > 2044 or z < 0:
        print("z:", z)
        OBz.append(z)
        raise OutofBounds
    seg = 0
    while(True):
        if z >= seg and z < seg+(292*GRANULARITY):
            return (seg+(seg+(292*GRANULARITY)))/2
        seg += (292*GRANULARITY)

## segmenting df ##
print("NORMALIZING...")
rootdir = '/home/zach/Files/ReplayModels/ReplayDataProcessing/RANKED_STANDARD/Replays/1400-1600/CSVs'
for root, dirs, files in os.walk(rootdir):
    for filename in tqdm(files):
        if not filename.endswith('.csv'):
            print("\n", filename, "not a csv\n")
            continue
        
        if GRANULARITY == 1:
            csv_name = "/home/zach/Files/ReplayModels/ReplayDataProcessing/RANKED_STANDARD/Replays/1400-1600/CSVs_8x_10y_7z/"
        elif GRANULARITY == .5:
            csv_name = "/home/zach/Files/ReplayModels/ReplayDataProcessing/RANKED_STANDARD/Replays/1400-1600/CSVs_16x_20y_14z/"
        else:
            print("BAD GRANULARITY")
            exit()
       
        csv_name = csv_name+filename
        if os.path.exists(csv_name):
            print("\n", csv_name, "exists\n")
            continue
        
        df = pd.read_csv(os.path.join(root, filename), low_memory=False)
        df.drop(columns=['Unnamed: 0'], inplace=True)
        try: 
            for i in df.index: # g1 -> granularity is 1
                if GRANULARITY == 1:
                    df.at[i, '0_pos_x_g1'] = x_segment(df.at[i, '0_pos_x'])
                    df.at[i, '0_pos_y_g1'] = y_segment(df.at[i, '0_pos_y'])
                    df.at[i, '0_pos_z_g1'] = z_segment(df.at[i, '0_pos_z'])
                elif GRANULARITY == .5:
                    df.at[i, '0_pos_x_g1/2'] = x_segment(df.at[i, '0_pos_x'])
                    df.at[i, '0_pos_y_g1/2'] = y_segment(df.at[i, '0_pos_y'])
                    df.at[i, '0_pos_z_g1/2'] = z_segment(df.at[i, '0_pos_z'])
        except OutofBounds:
            print("OB")
            print(OBx)
            print(OBy)
            print(OBz)
            continue

        df.to_csv(csv_name)
    break
