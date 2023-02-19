import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer



# Transformer for categorical variable
# LabelEncoder
class NewLabelEncoder(LabelEncoder):
    
    def fit(self, X, y=None):
        return super().fit(X)
    
    def transform(self, X, y=None):
        return pd.DataFrame({'label': super().transform(X)})
    
    def fit_transform(self, X, y=None):
        return pd.DataFrame({'label': super().fit_transform(X)})


# Model comparison: using metrics MAE, MSE, r2
def model_metrics(models, X_train, X_test, y_train, y_test):
    '''
    
    '''
    res = {}
    for i in models:
        models[i].fit(X_train, y_train)
        res[i] = [
            mean_absolute_error(y_train, models[i].predict(X_train)),
            mean_squared_error(y_train, models[i].predict(X_train)),
            r2_score(y_train, models[i].predict(X_train)),
            mean_absolute_error(y_test, models[i].predict(X_test)),
            mean_squared_error(y_test, models[i].predict(X_test)),
            r2_score(y_test, models[i].predict(X_test))
       ]
    res = pd.DataFrame(res)
    res.index = ['MAE_train', 'MSE_train', 'r2_train', 'MAE_test', 'MSE_test', 'r2_test']

    return res

# Model feature importance analysis
# plot the feature importance
def model_fi(models, X_train, y_train, feature_name=None):
    res = {}
    figi = 1
    n = len(models)

    plt.figure( layout='tight')
    for name in models:
        models[name].fit(X_train, y_train)
        
        if 'Lasso' in name or 'Ridge' in name:
            res[name] = [
                models[name]['regressor'].coef_
            ]

            plt.subplot(n*100+10+figi)
            plt.barh(models[name]['poly'].get_feature_names_out(feature_name),
                     models[name]['regressor'].coef_)
            plt.title('Feature importance of ' + name)
            if len(models[name]['poly'].get_feature_names_out(feature_name)) > 10:
                plt.tick_params(axis='y', labelsize=2)
            figi += 1

        if 'LR' in name:        
            res[name] = [
                models[name]['regressor'].coef_
            ]
            plt.subplot(n*100+10+figi)
                 
            if len(models[name]['scaler'].get_feature_names_out()) == len(models[name]['preprocessor'].transformers_[0][2]):
                plt.bar(models[name]['preprocessor'].transformers_[0][2],
                        models[name]['regressor'].coef_)
                plt.title('Feature importance of ' + name)
            else:
                plt.bar(range(len(models[name]['scaler'].get_feature_names_out())),
                        models[name]['regressor'].coef_)
                plt.title('Feature importance of ' + name)
            figi += 1

        if 'RF' in name or 'XGB' in name:
            res[name] = [
                models[name]['regressor'].feature_importances_
            ]

            plt.subplot(n*100+10+figi)
            plt.barh(feature_name,
                    models[name]['regressor'].feature_importances_)
            plt.title('Feature importance of ' + name)
            #plt.xticks(rotation=90)
            figi += 1
    
    plt.tight_layout()
       # print(i, models[i]['regressor'].coef_)
    res = pd.DataFrame(res)
    res.index = ['Feature Importance']

    return res

# Model hyperparameter tune
def model_tune(model, param, train, y_train):
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    model_tuned = GridSearchCV(
            estimator=Pipeline(model),
            param_grid=param,
            scoring=scorer,
            n_jobs = -1,
            cv=3
    ).fit(train, y_train)
    
    print('Tunning results: ', model_tuned.best_params_)
    return model_tuned

# plot and compare model performance
def performance_compare(model_performance, model_name):
    column_name = []
    for i in model_name:
        if 'LR_onehot' in i:
            column_name.append('LR_2')
        if 'Lasso' in i:
            column_name.append('Lasso_2_tuned')
        if 'Ridge' in i:
            column_name.append('Ridge_2_tuned')
        if 'Random' in i:
            column_name.append('RF_Tuned')
        if 'XGB' in i:
            column_name.append('XGB_Tuned')

    plt.figure(figsize=[7, 8], layout='constrained')
    plt.subplot(3,1,1)
    plt.bar(np.arange(len(model_name))-0.15, 
            model_performance.loc['MAE_train'][column_name],
            width=0.3, label='MAE_train')
    plt.bar(np.arange(len(model_name))+0.15,
            model_performance.loc['MAE_test'][column_name],
            width=0.3, label='MAE_test')
    plt.xticks(np.arange(len(model_name)), model_name)
    plt.legend(loc='lower right')

    plt.subplot(3,1,2)
    plt.bar(np.arange(len(model_name))-0.15,
            model_performance.loc['MSE_train'][column_name],
            width=0.3, label='MSE_train')
    plt.bar(np.arange(len(model_name))+0.15,
            model_performance.loc['MSE_test'][column_name],
            width=0.3, label='MSE_test')
    plt.xticks(np.arange(len(model_name)), model_name)
    plt.legend(loc='lower right')

    plt.subplot(3,1,3)
    plt.bar(np.arange(len(model_name))-0.15,
            model_performance.loc['r2_train'][column_name],
            width=0.3, label='r2_train')
    plt.bar(np.arange(len(model_name))+0.15,
            model_performance.loc['r2_test'][column_name],
            width=0.3, label='r2_test')
    plt.xticks(np.arange(len(model_name)), model_name)
    plt.legend(loc='lower right')


class TypeDummyTransformer(BaseEstimator):

    def __init__(self):
        self.keys = ['C', 'B,A', 'N', 'H', 'Q,H', 'V', 'K', 'U,H', 'K,S', 'I,M', 'S,N', 'K,H']

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        res = {}
        for key in self.keys:
            res[key] = [0] * len(X)
        
        for i, x in enumerate(X):
            if x in self.keys:
                res[x][i] = 1
        return pd.DataFrame(res)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

class MakerDummyTransformer(BaseEstimator):

    def __init__(self):
        self.keys = ['M144M145M147', 'M145M145', 'M145M144', 'M142', 'M146M149','M149', 'M146M147M145', 'M141M145']

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        res = {}
        for key in self.keys:
            res[key] = [0] * len(X)
        
        for i, x in enumerate(X):
            if x in self.keys:
                res[x][i] = 1
        return pd.DataFrame(res)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

