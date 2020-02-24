import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from GPRegressionModel import *
from GridSearchCV_func import *

df = pd.read_csv("dataQ3.csv")
df = df.rename(columns={'Months.since.Jan.1960': 'm'})
df['year'] = (df.m - 1) // 12 + 1960
df['month'] = (df['m'] - 1) % 12 + 1
df['month_name'] = df['month'].map({1: 'Jan', 2: 'Feb', 3: "Mar", 4: 'Apr', 5: 'May', 6: 'Jun',
                                    7: "Jul", 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'})

X = df.drop(['temp'], axis=1)
y = df['temp'].values
enc = OneHotEncoder(sparse=False)
enc.fit(X['month_name'].values.reshape((-1, 1)))
X_enc = enc.transform(X['month_name'].values.reshape((-1, 1)))
df_enc = pd.DataFrame(data=X_enc, columns=enc.categories_[0])

X = pd.concat([X, df_enc], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, )

X_train_1 = X_train.drop(['month', 'month_name'], axis=1)

X_test_1 = X_test.drop(['month', 'month_name'], axis=1)

X_train_1 = X_train_1.values
X_test_1 = X_test_1.values

y_train_1 = (y_train - np.mean(y_train)) / np.std(y_train)
y_test_1 = (y_test - np.mean(y_train)) / np.std(y_train)


params_grid = {'kernel': ['gaussian', 'matern', "periodic"],
               "amplitude": [0.1, 1, 5, 10, 40],
               "length": [0.1, 1, 10, 20],
               "periodicity": [0.1, 1, 50, 100]}


best_score, best_params, Scores_CV, mesh_grid_df = GridSearch_CV(
    X_train_1, y_train_1, GPRegression, params_grid)

mesh_grid_df['results'] = Scores_CV
mesh_grid_df.to_csv('gp_regression_results.csv')

with open('GPRegr_results.txt', 'w') as f:
    f.write(str(best_score))
    f.write(str(best_params))
