import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error


def cross_validation_score(X, y, Model_cls, params, score_func=mean_squared_error,
                           n_fold=5, stratified=False, shuffle=True):
    scores = []
    if stratified:
        kfold = StratifiedKFold(n_splits=n_fold, shuffle=shuffle)
        k = 1
        for train_index, val_index in kfold.split(X, y):
            print('training : ', k)
            k += 1
            model = Model_cls(**params)
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            scores.append(score_func(y_val, y_pred))
    else:
        kfold = KFold(n_splits=n_fold, shuffle=shuffle)
        k = 1
        for train_index, val_index in kfold.split(X):
            print('training : ', k)
            k += 1
            model = Model_cls(**params)
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            scores.append(score_func(y_val, y_pred))

    return np.mean(scores)


def GridSearch_CV(X, y, Model_cls, parameters, score_func=mean_squared_error, n_fold=5,
                  stratified=False, shuffle=True):

    log_file = open('log_file', 'w')
    # Create Mesh grid
    params_name = list(parameters.keys())
    n_params = len(params_name)
    params_values = np.array([v for v in parameters.values()])
    mesh_grid = np.array(np.meshgrid(*params_values)).T.reshape(-1, n_params)
    mesh_grid_df = pd.DataFrame(data=mesh_grid, columns=params_name)

    Scores_CV = []
    print('NB of Cross Validation: ', len(mesh_grid_df))
    log_file.write('NB Cross Validation : {}'.format(len(mesh_grid_df)))
    log_file.close()
    for i in range(len(mesh_grid_df)):
        log_file = open('log_file', 'a')
        params = mesh_grid_df.loc[i].to_dict()

        score_cv = cross_validation_score(
            X, y, Model_cls, params, score_func, n_fold, stratified, shuffle)
        Scores_CV.append(score_cv)

        print('CV {} / {} done. Score: {}'.format(i, len(mesh_grid_df), score_cv))
        log_file.write('CV {} / {} done. Score: {} \n'.format(i,
                                                              len(mesh_grid_df), score_cv))
        log_file.close()
    best_score = np.min(Scores_CV)
    best_params = mesh_grid_df.loc[np.argmin(Scores_CV)]

    return best_score, best_params, Scores_CV, mesh_grid_df
