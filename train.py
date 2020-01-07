import pandas as pd
from keras.models import Model
from keras.layers import *

pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

GRANULARITY = 0

if GRANULARITY == 0:
    df = pd.read_csv('exact_train.csv')
if GRANULARITY == .5:
    df = pd.read_csv('train_16x_20y_14z.csv')
if GRANULARITY == 1:
    df = pd.read_csv('train_8x_10y_7z.csv')

df.drop(columns=['Unnamed: 0'], inplace=True)
rem_cols = []
for col in df.columns:
    if 'steer' in col or 'handbrake' in col \
    or 'active' in col or 'collect' in col \
    or 'ball_cam' in col or 'throttle' in col:
        rem_cols.append(col)
df.drop(columns=rem_cols, inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# MOVE MY POSITION TO FRONT OF DATAFRAME
cols = list(df.columns.values)
cols.pop(cols.index('0_pos_x'))
cols.pop(cols.index('0_pos_y'))
cols.pop(cols.index('0_pos_z'))
df = df[['0_pos_x', '0_pos_y', '0_pos_z']+cols]

# NORMALIZING MAKING RANGE -1 TO 1 FOR ALL COLS
for col in df.columns:
    df[col] = (df[col] - df[col].mean()) / df[col].std()

x_train = df.values
x_train = x_train[:, 3:]
y_train = [df['0_pos_x'].values, df['0_pos_y'].values, df['0_pos_z'].values]

inp = Input((x_train.shape[1],))
h = Dense(100, activation='sigmoid', name='h')(inp)
h2 = Dense(100, activation='sigmoid', name='h2')(h)
out1 = Dense(1, name='output_x')(h2)
out2 = Dense(1, name='output_y')(h2)
out3 = Dense(1, name='output_z')(h2)
model = Model(inp, [out1, out2, out3])
model.summary()

model.compile(loss='mean_squared_error', optimizer='rmsprop')
model.fit(x_train, y_train, epochs=60, batch_size=128)
