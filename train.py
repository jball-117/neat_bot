import pandas as pd
from keras.models import Model, Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.layers import *

GRANULARITY = 0

if GRANULARITY == 0:
    df = pd.read_csv('exact_train.csv')
if GRANULARITY == .5:
    df = pd.read_csv('train_16x_20y_14z.csv')
if GRANULARITY == 1:
    df = pd.read_csv('train_8x_10y_7z.csv')

df.drop(columns=['Unnamed: 0'], inplace=True)

# NORMALIZING MAKING RANGE -1 TO 1 FOR ALL COLS
for col in df.columns:
    df[col] = (df[col] - df[col].mean()) / df[col].std()

x_train = df.values
x_train = x_train[:, 3:]
y_train = [df['0_pos_x'].values, df['0_pos_y'].values, df['0_pos_z'].values]
print(len(x_train))
print(len(y_train))
exit()

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
