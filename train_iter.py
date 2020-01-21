#import dask.dataframe as dd
import pandas as pd
import numpy as np
from keras.models import Model, Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.layers import *
import joblib

'''
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
'''

GRANULARITY = 0
FRAMES_AHEAD = 15

if GRANULARITY == 0:
    #df = dd.read_csv("exact_train_"+str(FRAMES_AHEAD)+"_frames"+".csv")
    df = pd.read_csv("exact_train_"+str(FRAMES_AHEAD)+"_frames"+".csv")
if GRANULARITY == .5:
    #df = dd.read_csv("train_16x_20y_14z_"+str(FRAMES_AHEAD)+"_frames"+".csv")
    df = pd.read_csv("train_16x_20y_14z_"+str(FRAMES_AHEAD)+"_frames"+".csv")
if GRANULARITY == 1:
    #df = dd.read_csv("train_8x_10y_7z_"+str(FRAMES_AHEAD)+"_frames"+".csv")
    df = pd.read_csv("train_8x_10y_7z_"+str(FRAMES_AHEAD)+"_frames"+".csv")

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

input_cols = [x for x in df.columns if x != '0_pos_x' \
              and x != '0_pos_y' and x != '0_pos_z']
x_train = df.filter(items=input_cols)
y_train = df.filter(items=['0_pos_x', '0_pos_y', '0_pos_z'])
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

def create_model(layers=[45], activation='sigmoid', optimizer='rmsprop'):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i == 0:
            model.add(Dense(nodes, input_dim=x_train.shape[1]))
        else:
            model.add(Dense(nodes))
        model.add(Activation(activation))
    model.add(Dense(3))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

print("CONSTRUCTING...")
model = KerasRegressor(build_fn=create_model, verbose=2)

layers = [[46], [68, 23]]
activations = ['relu']
batch_size = [128]
epochs = [90]
optimizers = ['rmsprop']

param_grid = dict(layers=layers, activation=activations, batch_size=batch_size,
                  epochs=epochs, optimizer=optimizers)

grid = GridSearchCV(estimator=model, param_grid=param_grid,
                    scoring='neg_mean_squared_error',
                    verbose=5, cv=5)

print("FITTING...")
grid_result = grid.fit(x_train, y_train)
print("SAVING...")
joblib.dump(grid_result, 'gd_obj_'+str(x_train.shape[0])+'_frames.pkl')

print(grid_result.best_score_)
print()
print(grid_result.best_params_)
