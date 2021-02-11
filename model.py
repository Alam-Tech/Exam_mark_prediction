#%%
#Importing packages:
import pandas as pd
import numpy as np
#%%
#Importing the data:
train_data = pd.read_csv('D:\ML_Comps\Exam_mark_prediction\Data\\train.csv')
test_data = pd.read_csv('D:\ML_Comps\Exam_mark_prediction\Data\\test.csv')
train_len = len(train_data)
data = pd.concat((train_data,test_data),axis=0)
del data['Unnamed: 0']
#%%
#Transforming the data:
data = pd.get_dummies(data,columns=['gender','ethnicity','parental level of education','lunch','test preparation course'],drop_first=True)
# %%
#Preparing the predictor and the target variables:
dep_train = data.iloc[:train_len,2].values
del data['math score']
ind_vars = data.iloc[:,:].values
#%%
#Preparing the train and test vars:
from statsmodels.api import add_constant
ind_vars = add_constant(ind_vars)

target_vars = [0,1,2,3,4,7,8,10,13,14]
ind_train = ind_vars[:train_len,target_vars]
ind_test = ind_vars[train_len:,target_vars]
#%%
#Applying the model and getting predictions:
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(ind_train,dep_train)
results = regressor.predict(ind_test)
#%%
#Preparing the results:
results = np.array(results).ravel()
results = results.reshape(-1,1)
results = pd.DataFrame(results,columns=['math score'])
results.to_csv('D:\ML_Comps\Exam_mark_prediction\\results\submission_1.csv')
#Submission_1 - test_rmse = 5.4972
# %%
