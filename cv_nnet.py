import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from NnetModel import *
from GridSearchCV_func import *

df = pd.read_csv("dataQ2.csv")
# remove 2011
df = df.drop(df.loc[(df['V1'] == 2011) | (df['V1'] == 1924)].index)
X = df.drop('V1', axis=1)
y = df['V1']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = std_scale.transform(X_train)

X_test_scaled = std_scale.transform(X_test)

y_train_mlp = (y_train.values - np.mean(y_train)) / np.std(y_train)
y_test_mlp = (y_test.values - np.mean(y_train)) / np.std(y_train)



params_grid = {'n_neurons': [[100], [50, 30, 10], [10 for k in range(8)]],
               'lr': np.logspace(-5, 0, 5),
               'alpha': np.logspace(-4, 0, 5)}

best_score, best_params, Scores_CV, mesh_grid_df = GridSearch_CV(
    X_train_scaled, y_train_mlp, NN, params_grid, n_fold=5)

print(best_score, best_params)
mesh_grid_df['results'] = Scores_CV
mesh_grid_df.to_csv("nn_results.csv")

with open('cv_nnet_results', 'w') as f:
    f.write(str(best_score))
    f.write(str(best_params))
