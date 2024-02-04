# this script trains and tests a regression model on rime series data with different types of sparsity contraints 

import numpy as np
import pandas as pd
import sys
import os
from scipy.optimize import minimize
from scipy.integrate import quad
from statsmodels.tsa.tsatools import lagmat
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score 
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
from sparse_reg_models import pairwise_gc, pairwise_wf, lasso_objective, group_lasso_objective, freq_lasso_objective, sparse_regression_scipy, simplify_parameters

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()



def predict_linear(X_test, Y_coeffs):    
    predicted_values = np.dot(X_test, Y_coeffs)
    return predicted_values

def r2_from_df(dfTest, dfTestLags, paramdf):
    r2_all =[]
    X_test = dfTestLags.values
    for tick in tickList:
        y_true = dfTest[tick].values
        Y_coeffs = paramdf.loc[tick].values
        y_test = predict_linear(X_test, Y_coeffs)
        r2_tick = r2_score(y_true, y_test)
        r2_all.append(r2_tick)
    return r2_all

####################################################################################################################################################################################

data_file = sys.argv[1]  # file name (without extension), to be stored in ./Data folder
# the remaining parameters may be passed while running the script or changed in the script itself  
#nLag = int(sys.argv[2])
#trainTestRatio =float(sys.argv[3]) 
# alpha_lasso = float(sys.argv[4])
# alpha_glasso = float(sys.argv[5])
# epsilon_pw = float(sys.argv[6])
# pw_lasso = float(sys.argv[7])
# pw_glasso = float(sys.argv[8])
# for fin_data....
# nLag = 5
# trainTestRatio = 4
# alpha_lasso = 0.0005
# alpha_glasso = 0.0005
# alpha_flasso = 0.01
# epsilon_pw =0.0000000000001
# pw_lasso = 0.000001
# pw_glasso = 0.000001
# pw_flasso = 0.00001

nLag = 5
trainTestRatio = 4
alpha_lasso = 0.05
alpha_glasso = 0.05
alpha_flasso = 0.01
epsilon_pw =0.01
pw_lasso = 0.2
pw_glasso = 0.2
pw_flasso = 0.00001
normalize = True
#Evaluate differences for time step Tstep
Tstep = 0

#p_sign = float(sys.argv[4])   

MainPath = os.path.normpath(os.getcwd() + os.sep + os.pardir)
DataPath = str(MainPath) + "\\Data\\"
#print("DataPath = ",  DataPath)


df = pd.read_csv(DataPath + data_file + ".csv", index_col=0).reset_index()
tickList = [col for col in df.columns.tolist() if col not in ["Date", "date", "index"]]
nTicks = len(tickList)


df1 = df[tickList]
if Tstep>0:
    df1 = df[tickList].pct_change(Tstep)[Tstep:].iloc[::Tstep, :]
df1 = df1.fillna(0)
nData = df1.shape[0]
nTrain = int(nData*trainTestRatio/(trainTestRatio+1))
dfTrain = df1.loc[0:nTrain].copy(deep=True)
print("dfTrain.shape - ", dfTrain.shape)
dfTest = df1.loc[nTrain+1:nData]
print("dfTest.shape - ", dfTest.shape)
dfTrain1 = dfTrain
dfTest1 = dfTest

# Normalize data 
if normalize:
    meanData = dfTrain.mean()
    stdData = dfTrain.std()
    dfTrain1 = (dfTrain - meanData)/stdData
    dfTest1 = (dfTest - meanData)/stdData


# generate lags 

dfTrainLags = lagmat(dfTrain1[tickList], maxlag=nLag, use_pandas=True)
dfTrainLags  = dfTrainLags.reindex(sorted(dfTrainLags.columns), axis=1)
dfTestLags = lagmat(dfTest1[tickList], maxlag=nLag, use_pandas=True)
dfTestLags  = dfTestLags.reindex(sorted(dfTestLags.columns), axis=1)
feature_names = dfTrainLags.columns.tolist()

parameters = []
edges = []
model_names = []

#### VAR Model #####
print("VAR Parameters")
model_name = "VAR"
data = dfTrain[tickList].copy(deep=True)
#data.index = df["Date"]
# make a VAR model
model = VAR(data)
results = model.fit(nLag)

VAR_param_df = results.params.transpose()
old_cols = VAR_param_df.columns.tolist()[1:]
new_cols = [f"{part[3:]}.{part[0]}.{part[1]}" for part in old_cols]
VAR_param_df.rename(columns=dict(zip(old_cols, new_cols)), inplace=True)
VAR_param_df = VAR_param_df[new_cols].reindex(sorted(new_cols), axis=1)

#VAR_param_df.reset_index(inplace=True)
#VAR_param_df.rename(columns={'index': 'target'}, inplace=True)
#print(VAR_param_df)

adj_df = simplify_parameters(VAR_param_df,  tickList, feature_names) 
np.fill_diagonal(adj_df.values, 0)
#print(adj_df)
print("Number of edges = " , 1*(adj_df>0).sum().sum())
parameters.append(VAR_param_df)
edges.append(adj_df)
model_names.append(model_name)

#### pairwise tests ######
print("+" * 50)
print("Pairwise Tests")
pairwise_df = pairwise_wf(dfTrain1, dfTrainLags, epsilon_pw)
edges.append(pairwise_df)
#model_names.append("pairwise")
print("Number of edges = " , 1*(pairwise_df>0).sum().sum())
#print(pairwise_df)

# Sparsifying directly on VAR model
for objective_func in [lasso_objective, group_lasso_objective, freq_lasso_objective]: #freq_lasso_objective
    print("+" * 50)
    print( objective_func.__name__)
    model_name ="VAR_" + objective_func.__name__
    param_df = pd.DataFrame(data = [], columns = dfTrainLags.columns.tolist())
    for tick in tickList:
        Y = dfTrain[tick].values        
        X = dfTrainLags.values
        Y_coeffs = sparse_regression_scipy(X, Y, objective_func, alpha_lasso, alpha_glasso, alpha_flasso, nLag)
        #print(Y_coeffs.shape, param_df.shape)
        param_df.loc[len(param_df)] = Y_coeffs
       # print(tick, Y_coeffs)
    param_df["target"] = tickList
    param_df.set_index('target', inplace=True)    
   # print(param_df.round(2))
    adj_df = simplify_parameters(param_df,  tickList, feature_names) 
    parameters.append(param_df)
    edges.append(adj_df)
    model_names.append(model_name)
    print("Number of edges = ",  1*(adj_df>0).sum().sum())
    #print(simplify_parameters(param_df))

# Sparsifying after initializing from pairwise test
for objective_func in [lasso_objective, group_lasso_objective, freq_lasso_objective]: #lasso_objective, group_lasso_objective, freq_lasso_objective
    print("+" * 50)
    print( objective_func.__name__)
    model_name ="PW_" + objective_func.__name__
    param_df = pd.DataFrame(data = [], columns = dfTrainLags.columns.tolist())
    all_lag_cols =  dfTrainLags.columns.tolist()
    for tick in tickList:
        #print(tick)
        Y = dfTrain[tick].values 
        pairwise_row = pairwise_df.loc[tick]             
        input_ticks = [tick] +  pairwise_row.index[pairwise_row == 1].tolist() 
        #print("tick = ", tick, " pw_row = ", pairwise_row, " ip_tks = ", input_ticks)
        input_lags = [s1 for s1 in all_lag_cols for s2 in input_ticks if s1.startswith(s2 + '.L')]
        #print(tick, input_lags)
        if len(input_lags)==0:
           param_df.loc[len(param_df)] = np.zeros(param_df.shape[1]) 
        else:   
            #print("in else")
            X = dfTrainLags[input_lags].values
            #print(X.shape, Y.shape)
            Y_coeffs = sparse_regression_scipy(X, Y, objective_func, pw_lasso, pw_glasso, pw_flasso, nLag)
            #print(Y_coeffs.shape, param_df.shape)
            row_Y = pd.DataFrame([Y_coeffs], columns=input_lags)
            missing_ticks = [t for t in all_lag_cols if t not in input_lags]
            missing_coefs = np.zeros(len(missing_ticks))
            missing_coefs_row = row_Y_new = pd.DataFrame([missing_coefs], columns=missing_ticks)

            # Concatenate DataFrames along columns (axis=1)
            new_row_Y = pd.concat([row_Y, missing_coefs_row], axis=1)
            # print("row_Y = ", input_lags)
            # print("missing_coefs_row", missing_ticks)
            # print("new_row_Y ", new_row_Y)
            #param_df = param_df.append(new_row_Y, ignore_index=True)
            param_df  = pd.concat([param_df, new_row_Y], ignore_index=True)
        #print(tick) 
       # print(tick, Y_coeffs)
    param_df["target"] = tickList
    param_df.set_index('target', inplace=True)    
    # print(param_df.round(2))
    adj_df = simplify_parameters(param_df, tickList, feature_names) 
    parameters.append(param_df)
    edges.append(adj_df)
    model_names.append(model_name)
    print("Number of edges = ",  1*(adj_df>0).sum().sum())
    

#pairwise_gc( df1, p_sign, nLag)
    # coeffs = sparse_regression_scipy(X, Y, objective_func, alpha, group_size)
    # print(coeffs)
    # parameters.append(coeffs)


for model_i in range(len(model_names)):
    model_name = model_names[model_i]
    
    param_df = parameters[model_i]
    #print(model_name, param_df.head())
    r2_train = 100*np.mean(np.array(r2_from_df(dfTrain, dfTrainLags, param_df)))
    r2_test = 100*np.mean(np.array(r2_from_df(dfTest, dfTestLags, param_df)))
    print(model_name + ", R2-train = "+ str(np.round(r2_train, decimals = 2)) + ", R2-test = "+ str(np.round(r2_test, decimals = 2)))


# Open the file in binary write mode and use pickle to dump the list
# with open(DataPath + data_file +"_res_aL_" + str(alpha_lasso) +"_res_aG_" + str(alpha_glasso) +"_epspw_" +
#            str(epsilon_pw) + '_pwL_'+ str(pw_lasso) +"_pwG_" + str(pw_glasso) +".pkl", 'wb') as file:
#     pickle.dump(parameters, file)



