#%%
#Importing the modules
import pandas as pd
import numpy as np
#%%
#Importing the data:
data = pd.read_csv('D:\ML_Comps\Exam_mark_prediction\Analysis\\refined_data.csv')
#%%
#Separating the target and predictor variables:
dep_vars = data.loc[:,'math_score'].values
del data['math_score']
ind_vars = data.iloc[:,:].values
#%%
#Preparing the predictor variables:
# from statsmodels.api import add_constant
# ind_vars = add_constant(ind_vars)
# ind_vars = ind_vars[:,[0,1,2,3,4,7,8,10,13,14]]

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# ind_vars = scaler.fit_transform(ind_vars)
#%%
#Utility Functions:

# from statsmodels.api import OLS,add_constant
# ind_vars = add_constant(ind_vars)

# history = OLS(dep_vars,ind_vars[:,[0,1,2,3,4,7,8,10,13,14]]).fit()
# print(history.summary())

from sklearn.model_selection import cross_val_score
def evaluate(regressor,ind_vars,dep_vars,name):
    scores=cross_val_score(regressor,X=ind_vars,y=dep_vars,cv=10,n_jobs=-1,scoring='neg_root_mean_squared_error')
    print(f'\n{name}:')
    print(f'The mean rmse is {scores.mean()}')
    print(f'The std.deviaion is {scores.std()}')
# %%
#Linear Regression:
# Linear Regression([0,1,2,3,4,7,8,10,13,14] & without standard scaling):
# The mean rmse is -5.274109116220796
# The std.deviaion is 0.29506857712292833

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

evaluate(regressor,ind_vars,dep_vars,'Linear Regression')
#%%
#Non Linear Regression:
#Without Standard scaling:
# GradBoostReg:
# The mean rmse is -5.886446354895976
# The std.deviaion is 0.4812067457593804

# ExtraTreeReg:
# The mean rmse is -6.183243616269783
# The std.deviaion is 0.4930689191196667

# AdaBoostReg:
# The mean rmse is -6.18099109866586
# The std.deviaion is 0.5566837976091937

# XGBReg:
# The mean rmse is -6.523939998283884
# The std.deviaion is 0.5337694227116682

# XGBRFReg:
# The mean rmse is -5.946635212672028
# The std.deviaion is 0.4730614220761634

# from sklearn.ensemble import \
#     RandomForestRegressor,GradientBoostingRegressor,ExtraTreesRegressor,AdaBoostRegressor
# from xgboost import XGBRegressor,XGBRFRegressor

# regressors=[
#     GradientBoostingRegressor(n_estimators=500,random_state=0),
#     ExtraTreesRegressor(n_estimators=500,random_state=0),
#     AdaBoostRegressor(n_estimators=200,random_state=0),
#     XGBRegressor(n_estimators=500),
#     XGBRFRegressor(n_estimators=500)
# ]
# names=['GradBoostReg','ExtraTreeReg','AdaBoostReg','XGBReg','XGBRFReg']

# for reg,name in zip(regressors,names):
#     evaluate(reg,ind_vars,dep_vars,name)