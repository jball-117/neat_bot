import pandas as pd
from keras.models import Model
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.layers import *

df = pd.read_csv('train_8x_10y_7z.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)

# NORMALIZING MAKING RANGE -1 TO 1 FOR ALL COLS
for col in df.columns:
    df[col] = (df[col] - df[col].mean()) / df[col].std()

x_train = df.values[:, 3:]
#x_train = x_train[:, 3:]
y_train = [df['0_pos_x'].values, df['0_pos_y'].values, df['0_pos_z'].values]

def create_model(layers_nodes, activation):
    model = Model(inputs=Input((x_train.shape[1],)))
    for i, nodes in enumerate(layers_nodes):
        last_layer = Dense(nodes)
        model.add(last_layer)
        model.add(Activation(activation))
    model.outputs = [Dense(1, name='output_x')(last_layer),
                     Dense(1, name='output_y')(last_layer),
                     Dense(1, name='output_z')(last_layer)]

    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model

model = KerasRegressor(build_fn=create_model, verbose=0)

layers_nodes = [[45], [55, 35], [60, 35, 20], [65, 50, 33, 15], [67, 52, 35, 21, 12],
                [70, 55, 34, 22, 13, 6]]
activations = ['sigmoid', 'relu']
param_grid = dict(layers=layers_nodes, activation=activations, batch_size=[128,256],
                  epochs=[30, 45, 60, 75, 90, 105])
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error')

print(len(x_train))
print(len(y_train))
exit()
grid_result = grid.fit(x_train, y_train)

print(grid_result.best_score_)
print()
print(grid_result.best_params_)


# inp = Input((x_train.shape[1],))
# h = Dense(100, activation='sigmoid', name='h')(inp)
# h2 = Dense(100, activation='sigmoid', name='h2')(h)
# out1 = Dense(1, name='output_x')(h2)
# out2 = Dense(1, name='output_y')(h2)
# out3 = Dense(1, name='output_z')(h2)
# model = Model(inp, [out1, out2, out3])
# model.summary()
#
# model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=[metrics.cosine_similarity])
# model.fit(x_train, y_train, epochs=60, batch_size=128)