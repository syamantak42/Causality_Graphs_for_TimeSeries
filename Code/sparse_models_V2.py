# this script finds optimal (sparse) causality graphs using a combination of pairwise Wiener filters and group lasso 

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
#from sklearn.metrics import r2_score 
from sklearn.preprocessing import StandardScaler
import pickle
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


def pairwise_gc( df, p_sign, nLag):
    test   = 'ssr_chi2test'
    for col1 in df:
        for col2 in [col for col in df if col != col1]:
            data = df[[col1, col2]]
            gc_res = grangercausalitytests(data, nLag, verbose = False)
           #print(dir(gc_res))
            p_values = [round(gc_res[i+1][0][test][1],4) for i in range(nLag)]
            #print(col1, col2, p_values)
            print(p_values)
            if np.min(p_values)>=p_sign:
                print(col2 + " ----> GC ----> " + col1)
    return 


def pairwise_wf( df1, dflags, epsilon = 0.001):
    nSeries = df1.shape[1]
    AdjMat = np.empty([nSeries, nSeries])
    Names = df1.columns.tolist()
    #print(df1.columns, dflags.columns)
    for col1 in Names:
        for col2 in [col for col in df1 if col != col1]:
            features = [c for c in dflags if col2 in c]
            #print(col1, col2, features)
            X = dflags[features].values
            Y = df1[col1].values
            model = LinearRegression()
            model.fit(X,Y)
            Y_pred = model.predict(X)
            mse = np.mean((Y-Y_pred)**2)
            error_gain = 1 -mse/np.var(Y)
            #ratio = r2_score(Y, Y_pred)
            AdjMat[Names.index(col1), Names.index(col2)] = error_gain
    
    AdjMat[AdjMat < epsilon ] = 0
    mask = AdjMat > AdjMat.T  # Create a boolean mask for values where AdjMat(i, j) > AdjMat(j, i)

    AdjMat[mask] = 1   # Set values where A(i, j) > A(j, i) to 0
    AdjMat[~mask] = 0   # Set values where A(i, j) <= A(j, i) to 1
    np.fill_diagonal(AdjMat, 0)
            #print(col2+" --> "+ col1 + " :  "+ str(np.round(mse, decimals = 2)))
    results = pd.DataFrame(data = AdjMat, columns = Names, index = Names)
    return results

def lasso_objective(coeffs, X, Y, alpha, group_size):
    predicted_values = np.dot(X, coeffs)
    mse = np.mean((predicted_values - Y) ** 2)

    # Add L1 regularization term (lasso penalty)
    l1_penalty = alpha * np.sum(np.abs(coeffs))

    return mse + l1_penalty

def group_lasso_objective(params, X, Y, alpha, group_size):
    num_groups = len(params) // group_size
    coefficients = params.reshape((num_groups, group_size))

    predicted_values = np.dot(X, coefficients.flatten())
    mse = np.mean((predicted_values - Y) ** 2)

    # Add group L1 regularization term (group lasso penalty)
    group_lasso_penalty = alpha * np.sum(np.linalg.norm(coefficients, ord=2, axis=1))

    return mse + group_lasso_penalty

def phi(x, group_size):
    phi_x = []
    for t in range(group_size):
        phi_x.append(np.exp(2*(np.pi)*x*t*1j))
    phi_x = np.array(phi_x)
    return phi_x 

def group_integral_objective(params, X, Y, alpha, group_size):
    num_groups = len(params) // group_size
    coefficients = params.reshape((num_groups, group_size))

    predicted_values = np.dot(X, coefficients.flatten())
    mse = np.mean((predicted_values - Y) ** 2)

    # Add group integral regularization term
    group_integral_penalty = 0.0
    for group_coefficients in coefficients:
        group_function = lambda x: abs(np.dot(group_coefficients, phi(x, group_size)))
        integral_result, _ = quad(group_function, -0.5, 0.5)
        group_integral_penalty += integral_result

    group_integral_penalty *= alpha

    return mse + group_integral_penalty



    

def sparse_regression_scipy(X, Y, objective_func, alphaL, alphaG, group_size):
    num_features = X.shape[1]
    num_groups = num_features // group_size
    initial_params = np.zeros(num_groups * group_size)
    if objective_func.__name__ == "lasso_objective":
        alpha = alphaL
    if objective_func.__name__ == "group_lasso_objective":
        alpha = alphaG    
    objective_function = lambda params: objective_func(params, X, Y, alpha, group_size)

    result = minimize(objective_function, initial_params, method='L-BFGS-B')

    optimal_params = result.x
    #coefficients = optimal_params.reshape((num_groups, group_size))

    return optimal_params


def simplify_parameters(parameter_df) :
    abs_para = parameter_df.abs()
    adj_df = pd.DataFrame()
    feature_names = dftest.columns.tolist()
    for tick in tickList:
        lag_features = [c for c in feature_names if tick in c]
        adj_df[tick] = abs_para[lag_features].sum(axis=1)
    adj_df.index = tickList 
    return adj_df

####################################################################################################################################################################################

data_file = sys.argv[1]
nLag = int(sys.argv[2])
#trainTestRatio =float(sys.argv[3]) 
# alpha_lasso = float(sys.argv[4])
# alpha_glasso = float(sys.argv[5])
# epsilon_pw = float(sys.argv[6])
# pw_lasso = float(sys.argv[7])
# pw_glasso = float(sys.argv[8])

alpha_lasso = float(sys.argv[3])
alpha_glasso = float(sys.argv[4])
epsilon_pw = float(sys.argv[5])
pw_lasso = float(sys.argv[6])
pw_glasso = float(sys.argv[7])

 

MainPath = os.path.normpath(os.getcwd() + os.sep + os.pardir)
DataPath = str(MainPath) + "\\Data\\"
#print("DataPath = ",  DataPath)


df = pd.read_csv(DataPath + data_file + ".csv", index_col=0).reset_index()
tickList = [col for col in df.columns.tolist() if col not in ["Date", "date", "index"]]

#Evaluate differences for time step Tstep
Tstep = 1
#df1 = df[tickList]
df1 = df[tickList].pct_change(Tstep)[Tstep:].iloc[::Tstep, :]
df1 = df1.fillna(0)

# Normalize data 
scaler = StandardScaler()
# Fit and transform the DataFrame
df1 = pd.DataFrame(scaler.fit_transform(df1), columns=df1.columns)

group_size = nLag

dftest = lagmat(df1[tickList], maxlag=nLag, use_pandas=True)
dftest  = dftest.reindex(sorted(dftest.columns), axis=1)


#### VAR Model #####
print("VAR Parameters")
parameters = []
data = df1[tickList].copy(deep=True)
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
# print(VAR_param_df)

adj_df = simplify_parameters(VAR_param_df) 
np.fill_diagonal(adj_df.values, 0)
#print(adj_df)
print("Number of edges = " , 1*(adj_df>0).sum().sum())
parameters.append(adj_df)

print("+" * 50)
print("Pairwise Tests")
pairwise_df = pairwise_wf(df1, dftest, epsilon_pw)
parameters.append(pairwise_df)
print("Number of edges = " , 1*(pairwise_df>0).sum().sum())
#print(pairwise_df)

for objective_func in [lasso_objective, group_lasso_objective]: #group_integral_objective
    print("+" * 50)
    print( objective_func.__name__)
    param_df = pd.DataFrame(data = [], columns = dftest.columns.tolist())
    for tick in tickList:
        Y = df1[tick].values        
        X = dftest.values
        Y_coeffs = sparse_regression_scipy(X, Y, objective_func, alpha_lasso, alpha_glasso, group_size)
        #print(Y_coeffs.shape, param_df.shape)
        param_df.loc[len(param_df)] = Y_coeffs
       # print(tick, Y_coeffs)
    param_df["target"] = tickList
    param_df.set_index('target', inplace=True)    
    # print(param_df.round(2))
    adj_df = simplify_parameters(param_df) 
    parameters.append(adj_df)
    print("Number of edges = ",  1*(adj_df>0).sum().sum())
    #print(simplify_parameters(param_df))

for objective_func in [lasso_objective, group_lasso_objective]: #group_integral_objective
    print("+" * 50)
    print( objective_func.__name__)
    param_df = pd.DataFrame(data = [], columns = dftest.columns.tolist())
    all_lag_cols =  dftest.columns.tolist()
    for tick in tickList:
        #print(tick)
        Y = df1[tick].values 
        pairwise_row = pairwise_df.loc[tick]             
        input_ticks =  pairwise_row.index[pairwise_row == 1].tolist()
        #print("tick = ", tick, " pw_row = ", pairwise_row, " ip_tks = ", input_ticks)
        input_lags = [s1 for s1 in all_lag_cols for s2 in input_ticks if s1.startswith(s2 + '.L')]
        #print(tick, input_lags)
        if len(input_lags)==0:
           param_df.loc[len(param_df)] = np.zeros(param_df.shape[1]) 
        else:   
            #print("in else")
            X = dftest[input_lags].values
            #print(X.shape, Y.shape)
            Y_coeffs = sparse_regression_scipy(X, Y, objective_func, pw_lasso, pw_glasso, group_size)
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
    adj_df = simplify_parameters(param_df) 
    parameters.append(adj_df)
    print("Number of edges = ",  1*(adj_df>0).sum().sum())
    #print(simplify_parameters(param_df))

#pairwise_gc( df1, p_sign, nLag)
    # coeffs = sparse_regression_scipy(X, Y, objective_func, alpha, group_size)
    # print(coeffs)
    # parameters.append(coeffs)





# Open the file in binary write mode and use pickle to dump the list
with open(DataPath + data_file +"_res_aL_" + str(alpha_lasso) +"_res_aG_" + str(alpha_glasso) +"_epspw_" +
           str(epsilon_pw) + '_pwL_'+ str(pw_lasso) +"_pwG_" + str(pw_glasso) +".pkl", 'wb') as file:
    pickle.dump(parameters, file)



